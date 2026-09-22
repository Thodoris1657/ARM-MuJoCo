#!/usr/bin/env python
"""Train the 3-joint reacher with TorchRL (clipped PPO).

This is the "read every line" version: the collector, the GAE computation, the
PPO epochs and the optimiser step are all spelled out rather than hidden behind
a ``.learn()`` call.

Every ``--eval-every`` batches the deterministic policy is scored on held-out
cases from the frozen benchmark set, and the best one is kept as
``best_model.pt`` -- PPO's final iterate is often not its best.

Examples
--------
    python scripts/train_torchrl.py
    python scripts/train_torchrl.py --total-frames 400000 --seed 1
    python scripts/train_torchrl.py --target 0.3 0.0 0.35
    python scripts/train_torchrl.py --env-kwarg control_mode=torque
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from mujoco_arm_rl.utils import load_config, repo_root, set_seed  # noqa: E402

# ------------------------------------------------------------------ version shims
# TorchRL's public API still moves between releases:
#   * SyncDataCollector was renamed Collector (>= 0.11)
#   * entropy_coef / critic_coef were renamed entropy_coeff / critic_coeff (>= 0.13)
# Both are resolved at import time rather than pinning an exact version.

_ALIASES = {
    "entropy_coef": ("entropy_coeff",),
    "entropy_coeff": ("entropy_coef",),
    "critic_coef": ("critic_coeff",),
    "critic_coeff": ("critic_coef",),
}


def get_collector_cls():
    import torchrl.collectors as collectors

    for name in ("SyncDataCollector", "Collector"):
        cls = getattr(collectors, name, None)
        if cls is not None:
            return cls
    raise ImportError("no synchronous collector found in torchrl.collectors")


def compat_kwargs(cls, **kwargs):
    """Keep only kwargs this TorchRL version accepts, translating renamed ones."""
    import inspect

    params = inspect.signature(cls.__init__).parameters
    out = {}
    for key, value in kwargs.items():
        if key in params:
            out[key] = value
            continue
        for alias in _ALIASES.get(key, ()):
            if alias in params:
                out[alias] = value
                break
    return out


def parse_env_kwargs(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        key, raw = item.split("=", 1)
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


# --------------------------------------------------------------------------- env


def make_env(env_id: str, env_kwargs: dict, device: torch.device, obs_norm_steps: int = 2000):
    from torchrl.envs import Compose, DoubleToFloat, ObservationNorm, StepCounter, TransformedEnv
    from torchrl.envs.libs.gym import GymEnv
    from torchrl.envs.utils import check_env_specs

    import mujoco_arm_rl.envs  # noqa: F401  (registers the env IDs)

    base = GymEnv(env_id, device=device, **env_kwargs)
    env = TransformedEnv(base, Compose(ObservationNorm(in_keys=["observation"]),
                                       DoubleToFloat(), StepCounter()))
    env.transform[0].init_stats(num_iter=obs_norm_steps, reduce_dim=0, cat_dim=0)
    # Features that never vary (e.g. the target under --target) get std 0; clamp
    # the scale so normalisation cannot blow them up to inf/NaN.
    with torch.no_grad():
        env.transform[0].scale.clamp_(max=1e3)
    check_env_specs(env)
    return env


# ------------------------------------------------------------------------ models


def build_actor_critic(env, hidden: int, device: torch.device):
    from tensordict.nn import TensorDictModule
    from tensordict.nn.distributions import NormalParamExtractor
    from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

    n_obs = env.observation_spec["observation"].shape[-1]
    n_act = env.action_spec.shape[-1]

    actor_net = nn.Sequential(
        nn.Linear(n_obs, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 2 * n_act),
        NormalParamExtractor(),
    ).to(device)
    policy = ProbabilisticActor(
        module=TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"]),
        spec=env.action_spec, in_keys=["loc", "scale"],
        distribution_class=TanhNormal, return_log_prob=True,
    )
    value_net = nn.Sequential(
        nn.Linear(n_obs, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 1),
    ).to(device)
    value = ValueOperator(module=value_net, in_keys=["observation"])

    with torch.no_grad():  # lazy-init pass so parameter shapes exist
        td = env.reset()
        policy(td)
        value(td)
    return policy, value, actor_net, value_net


def build_loss(policy, value, clip_epsilon: float, entropy_eps: float):
    from torchrl.objectives import ClipPPOLoss

    return ClipPPOLoss(**compat_kwargs(
        ClipPPOLoss,
        actor_network=policy, critic_network=value,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps), entropy_coef=entropy_eps,
        critic_coef=1.0, loss_critic_type="smooth_l1",
        normalize_advantage=True,
    ))


# ------------------------------------------------------------------ evaluation


def checkpoint_blob(actor_net, value_net, obs_norm, hidden, n_obs, n_act, logs) -> dict:
    return {
        "policy_state_dict": {k: v.detach().cpu().clone() for k, v in actor_net.state_dict().items()},
        "value_state_dict": {k: v.detach().cpu().clone() for k, v in value_net.state_dict().items()},
        "obs_loc": obs_norm.loc.detach().cpu().clone(),
        "obs_scale": obs_norm.scale.detach().cpu().clone(),
        # ObservationNorm computes (obs - loc)/scale if standard_normal, else obs*scale + loc.
        "obs_standard_normal": bool(getattr(obs_norm, "standard_normal", False)),
        "hidden": hidden, "n_obs": n_obs, "n_act": n_act, "activation": "tanh",
        "logs": {k: list(map(float, v)) for k, v in logs.items()},
    }


def evaluate_blob(blob: dict, env_id: str, env_kwargs: dict, cases) -> dict:
    """Deterministic rollout of a checkpoint blob on held-out benchmark cases."""
    import tempfile

    from mujoco_arm_rl.benchmark import rollout
    from mujoco_arm_rl.policies import load_torchrl

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eval.pt"
        torch.save(blob, path)
        act, _ = load_torchrl(path)   # the exact loader evaluate.py uses
    res = rollout(act, env_id, env_kwargs, cases)
    return {"success": float(res["success_5cm"].mean()),
            "success_2cm": float(res["success_2cm"].mean()),
            "median_dist": float(np.median(res["final_dist"]))}


# -------------------------------------------------------------------------- main


def main() -> None:
    cfg = load_config(repo_root() / "configs" / "torchrl_ppo.yaml")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-id", default=cfg.get("env_id", "MujocoArm3Reach-v1"))
    p.add_argument("--env-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="train on ONE fixed target instead of random ones")
    p.add_argument("--total-frames", type=int, default=cfg.get("total_frames", 300_000))
    p.add_argument("--frames-per-batch", type=int, default=cfg.get("frames_per_batch", 2_000))
    p.add_argument("--sub-batch", type=int, default=cfg.get("sub_batch_size", 250))
    p.add_argument("--epochs", type=int, default=cfg.get("num_epochs", 10))
    p.add_argument("--lr", type=float, default=cfg.get("lr", 3e-4))
    p.add_argument("--hidden", type=int, default=cfg.get("hidden", 256))
    p.add_argument("--gamma", type=float, default=cfg.get("gamma", 0.95))
    p.add_argument("--lmbda", type=float, default=cfg.get("lmbda", 0.95))
    p.add_argument("--clip-epsilon", type=float, default=cfg.get("clip_epsilon", 0.2))
    p.add_argument("--entropy-eps", type=float, default=cfg.get("entropy_eps", 1e-4))
    p.add_argument("--max-grad-norm", type=float, default=cfg.get("max_grad_norm", 1.0))
    p.add_argument("--eval-every", type=int, default=cfg.get("eval_every", 10),
                   help="evaluate on held-out cases every N batches (0 = only at the end)")
    p.add_argument("--eval-cases", type=int, default=cfg.get("eval_cases", 30))
    p.add_argument("--seed", type=int, default=cfg.get("seed", 0))
    p.add_argument("--threads", type=int, default=cfg.get("threads", 0))
    p.add_argument("--device", default=cfg.get("device", "cpu"))
    p.add_argument("--out", default=None, help="output directory (default: runs/torchrl)")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out) if args.out else repo_root() / "runs" / "torchrl"
    out.mkdir(parents=True, exist_ok=True)

    from torchrl.data.replay_buffers import ReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.data.replay_buffers.storages import LazyTensorStorage
    from torchrl.objectives.value import GAE

    from mujoco_arm_rl.benchmark import load_eval_set
    from mujoco_arm_rl.policies import save_env_config

    env_kwargs = parse_env_kwargs(args.env_kwarg)
    if args.target is not None:
        env_kwargs["fixed_target"] = tuple(args.target)
    save_env_config(out, args.env_id, env_kwargs)
    # held-out cases: the *last* N of the frozen set, so the headline benchmark
    # (which reports on all 100) is not the exact set used for model selection
    eval_cases = load_eval_set()[-args.eval_cases:]
    if args.target is not None:
        eval_cases = [dict(c, target=list(args.target)) for c in eval_cases]

    print(f"[train_torchrl] PPO on {args.env_id} {env_kwargs or ''} | {args.total_frames} frames, "
          f"seed {args.seed}, device {device} -> {out}", flush=True)
    env = make_env(args.env_id, env_kwargs, device)
    env.set_seed(args.seed)
    policy, value, actor_net, value_net = build_actor_critic(env, args.hidden, device)
    n_obs = int(env.observation_spec["observation"].shape[-1])
    n_act = int(env.action_spec.shape[-1])

    collector_cls = get_collector_cls()
    collector = collector_cls(env, policy, **compat_kwargs(
        collector_cls, frames_per_batch=args.frames_per_batch, total_frames=args.total_frames,
        split_trajs=False, device=device, auto_register_policy_transforms=False))
    buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=args.frames_per_batch, device=device),
                          sampler=SamplerWithoutReplacement())
    advantage = GAE(gamma=args.gamma, lmbda=args.lmbda, value_network=value, average_gae=True)
    loss_module = build_loss(policy, value, args.clip_epsilon, args.entropy_eps)
    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr)
    n_batches = max(args.total_frames // args.frames_per_batch, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_batches, 0.0)

    logs = defaultdict(list)
    best = {"success": -1.0, "median_dist": np.inf}
    t0 = time.time()

    for i, data in enumerate(collector):
        for _ in range(args.epochs):
            with torch.no_grad():
                advantage(data)
            buffer.extend(data.reshape(-1))
            for _ in range(args.frames_per_batch // args.sub_batch):
                losses = loss_module(buffer.sample(args.sub_batch).to(device))
                loss = (losses["loss_objective"] + losses["loss_critic"]
                        + losses.get("loss_entropy", torch.zeros((), device=device)))
                loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), args.max_grad_norm)
                optim.step()
                optim.zero_grad(set_to_none=True)
        scheduler.step()

        rew = data["next", "reward"].mean().item()
        logs["reward"].append(rew)
        frames = (i + 1) * args.frames_per_batch
        line = (f"[{frames:>7d}/{args.total_frames}] mean step reward {rew: .4f} | "
                f"lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - t0:5.0f}s")

        last = (i + 1) == n_batches
        if (args.eval_every and (i + 1) % args.eval_every == 0) or last:
            blob = checkpoint_blob(actor_net, value_net, env.transform[0], args.hidden, n_obs, n_act, logs)
            ev = evaluate_blob(blob, args.env_id, env_kwargs, eval_cases)
            logs["eval_success"].append(ev["success"])
            line += (f" | eval success {100 * ev['success']:5.1f}% "
                     f"(@2cm {100 * ev['success_2cm']:5.1f}%) median {1000 * ev['median_dist']:.0f} mm")
            if (ev["success"], -ev["median_dist"]) > (best["success"], -best["median_dist"]):
                best = ev
                torch.save(blob, out / "best_model.pt")
                line += "  <- best"
        print(line, flush=True)

    collector.shutdown()
    torch.save(checkpoint_blob(actor_net, value_net, env.transform[0], args.hidden, n_obs, n_act, logs),
               out / "ppo_final.pt")

    print(f"[train_torchrl] done in {(time.time() - t0) / 60:.1f} min")
    print(f"[train_torchrl] best model : {out / 'best_model.pt'} "
          f"(held-out success {100 * best['success']:.1f}%)")
    print(f"[train_torchrl] watch it   : python scripts/evaluate.py --ckpt \"{out / 'best_model.pt'}\" --render human")


if __name__ == "__main__":
    main()
