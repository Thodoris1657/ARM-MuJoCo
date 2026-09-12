#!/usr/bin/env python
"""Train the 3-joint reacher with TorchRL (clipped PPO).

This is the "read every line" version: the collector, the GAE computation, the
PPO epochs and the optimiser step are all spelled out rather than hidden behind
a ``.learn()`` call.

Examples
--------
    python scripts/train_torchrl.py
    python scripts/train_torchrl.py --total-frames 500000 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from mujoco_arm_rl.utils import load_config, repo_root, run_dir, set_seed  # noqa: E402

# ------------------------------------------------------------------ version shims
# TorchRL's public API still moves between releases. Two things changed recently:
#   * SyncDataCollector was renamed Collector (>= 0.11)
#   * entropy_coef / critic_coef were renamed entropy_coeff / critic_coeff (>= 0.13)
# Rather than pinning an exact version, resolve both at import time.

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


# --------------------------------------------------------------------------- env


def make_env(device: torch.device, obs_norm_steps: int = 2000, fixed_target=None):
    from torchrl.envs import (
        Compose,
        DoubleToFloat,
        ObservationNorm,
        StepCounter,
        TransformedEnv,
    )
    from torchrl.envs.libs.gym import GymEnv
    from torchrl.envs.utils import check_env_specs

    import mujoco_arm_rl.envs  # noqa: F401  (registers MujocoArm3Reach-v0)

    env_kwargs = {} if fixed_target is None else {"fixed_target": tuple(fixed_target)}
    base = GymEnv("MujocoArm3Reach-v0", device=device, **env_kwargs)
    env = TransformedEnv(
        base,
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    env.transform[0].init_stats(num_iter=obs_norm_steps, reduce_dim=0, cat_dim=0)
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

    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )

    value_net = nn.Sequential(
        nn.Linear(n_obs, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 1),
    ).to(device)
    value = ValueOperator(module=value_net, in_keys=["observation"])

    # Lazy-init sanity pass so parameter shapes exist before the optimiser is built.
    with torch.no_grad():
        td = env.reset()
        policy(td)
        value(td)
    return policy, value, actor_net, value_net


def build_loss(policy, value, clip_epsilon: float, entropy_eps: float):
    """ClipPPOLoss, tolerant of the coefficient renames across TorchRL versions."""
    from torchrl.objectives import ClipPPOLoss

    return ClipPPOLoss(
        **compat_kwargs(
            ClipPPOLoss,
            actor_network=policy,
            critic_network=value,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
        )
    )


# -------------------------------------------------------------------------- main


def main() -> None:
    cfg = load_config(repo_root() / "configs" / "torchrl_ppo.yaml")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--total-frames", type=int, default=cfg.get("total_frames", 200_000))
    p.add_argument("--frames-per-batch", type=int, default=cfg.get("frames_per_batch", 2_000))
    p.add_argument("--sub-batch", type=int, default=cfg.get("sub_batch_size", 250))
    p.add_argument("--epochs", type=int, default=cfg.get("num_epochs", 10))
    p.add_argument("--lr", type=float, default=cfg.get("lr", 3e-4))
    p.add_argument("--hidden", type=int, default=cfg.get("hidden", 256))
    p.add_argument("--gamma", type=float, default=cfg.get("gamma", 0.98))
    p.add_argument("--lmbda", type=float, default=cfg.get("lmbda", 0.95))
    p.add_argument("--clip-epsilon", type=float, default=cfg.get("clip_epsilon", 0.2))
    p.add_argument("--entropy-eps", type=float, default=cfg.get("entropy_eps", 1e-4))
    p.add_argument("--max-grad-norm", type=float, default=cfg.get("max_grad_norm", 1.0))
    p.add_argument("--seed", type=int, default=cfg.get("seed", 0))
    p.add_argument("--device", default=cfg.get("device", "cpu"))
    p.add_argument("--out", default=None, help="output directory (default: runs/torchrl)")
    p.add_argument("--target", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="train/evaluate on ONE fixed target instead of random ones, e.g. --target 0.3 0.0 0.35")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out) if args.out else run_dir("torchrl")

    from torchrl.data.replay_buffers import ReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.data.replay_buffers.storages import LazyTensorStorage
    from torchrl.objectives.value import GAE

    print(f"[train_torchrl] device={device} total_frames={args.total_frames} -> {out}")
    if args.target is not None:
        print(f"[train_torchrl] fixed target at {tuple(args.target)} (no target randomisation)")
    env = make_env(device, fixed_target=args.target)
    env.set_seed(args.seed)

    policy, value, actor_net, value_net = build_actor_critic(env, args.hidden, device)

    collector_cls = get_collector_cls()
    collector = collector_cls(
        env,
        policy,
        **compat_kwargs(
            collector_cls,
            frames_per_batch=args.frames_per_batch,
            total_frames=args.total_frames,
            split_trajs=False,
            device=device,
            # The policy is a plain MLP, so no InitTracker transform is needed.
            auto_register_policy_transforms=False,
        ),
    )

    buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=args.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
    )

    advantage = GAE(gamma=args.gamma, lmbda=args.lmbda, value_network=value, average_gae=True)
    loss_module = build_loss(policy, value, args.clip_epsilon, args.entropy_eps)

    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr)
    n_batches = max(args.total_frames // args.frames_per_batch, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_batches, 0.0)

    logs = defaultdict(list)
    t0 = time.time()

    for i, data in enumerate(collector):
        for _ in range(args.epochs):
            with torch.no_grad():
                advantage(data)
            buffer.extend(data.reshape(-1))
            for _ in range(args.frames_per_batch // args.sub_batch):
                sub = buffer.sample(args.sub_batch).to(device)
                losses = loss_module(sub)
                loss = (
                    losses["loss_objective"]
                    + losses["loss_critic"]
                    + losses.get("loss_entropy", torch.zeros((), device=device))
                )
                loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), args.max_grad_norm)
                optim.step()
                optim.zero_grad(set_to_none=True)

        scheduler.step()

        rew = data["next", "reward"].mean().item()
        steps = data["step_count"].max().item()
        logs["reward"].append(rew)
        logs["step_count"].append(steps)
        frames = (i + 1) * args.frames_per_batch
        print(
            f"[{frames:>7d}/{args.total_frames}] "
            f"mean step reward {rew: .4f} | max episode len {steps:3.0f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:5.0f}s",
            flush=True,
        )

    collector.shutdown()

    obs_norm = env.transform[0]
    ckpt = {
        "policy_state_dict": actor_net.state_dict(),
        "value_state_dict": value_net.state_dict(),
        "obs_loc": obs_norm.loc.detach().cpu(),
        "obs_scale": obs_norm.scale.detach().cpu(),
        # ObservationNorm applies (obs - loc) / scale when standard_normal is set,
        # and obs * scale + loc otherwise. evaluate.py has to replay the same one.
        "obs_standard_normal": bool(getattr(obs_norm, "standard_normal", False)),
        "hidden": args.hidden,
        "n_obs": int(env.observation_spec["observation"].shape[-1]),
        "n_act": int(env.action_spec.shape[-1]),
        "logs": {k: list(map(float, v)) for k, v in logs.items()},
    }
    ckpt_path = out / "ppo_final.pt"
    torch.save(ckpt, ckpt_path)

    print(f"[train_torchrl] done in {(time.time()-t0)/60:.1f} min")
    print(f"[train_torchrl] checkpoint : {ckpt_path}")
    print(f"[train_torchrl] mean step reward: first={np.mean(logs['reward'][:3]):.4f} "
          f"last={np.mean(logs['reward'][-3:]):.4f}")
    print(f"[train_torchrl] evaluate   : python scripts/evaluate.py --algo torchrl "
          f"--ckpt \"{ckpt_path}\" --render human")


if __name__ == "__main__":
    main()
