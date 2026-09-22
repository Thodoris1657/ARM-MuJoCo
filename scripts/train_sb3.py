#!/usr/bin/env python
"""Train the 3-joint reacher with Stable-Baselines3 (SAC by default).

Examples
--------
    python scripts/train_sb3.py                                  # v1 task, config defaults
    python scripts/train_sb3.py --timesteps 30000 --seed 1
    python scripts/train_sb3.py --target 0.3 0.0 0.35            # one fixed goal
    python scripts/train_sb3.py --algo tqc                       # needs sb3-contrib
    python scripts/train_sb3.py --env-id MujocoArm3Reach-v0      # the original torque task

    # ablations: override any env constructor argument
    python scripts/train_sb3.py --env-kwarg control_mode=torque --env-kwarg reward_mode=legacy

The env ID and kwargs are written to <out>/env_config.json, so evaluate.py and
benchmark.py always roll the policy out on the task it was trained on.
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mujoco_arm_rl.utils import load_config, repo_root, set_seed  # noqa: E402


def parse_env_kwargs(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--env-kwarg expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw  # plain string, e.g. control_mode=torque
    return out


def build_env(env_id: str, n_envs: int, seed: int, env_kwargs: dict):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    import mujoco_arm_rl.envs  # noqa: F401  (registers the env IDs)

    return make_vec_env(env_id, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs,
                        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv)


def main() -> None:
    cfg = load_config(repo_root() / "configs" / "sb3_sac.yaml")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", default=cfg.get("algo", "sac"), choices=["sac", "tqc", "td3", "ppo"])
    p.add_argument("--env-id", default=cfg.get("env_id", "MujocoArm3Reach-v1"))
    p.add_argument("--env-kwarg", action="append", default=[], metavar="KEY=VALUE",
                   help="override an env constructor argument (repeatable)")
    p.add_argument("--target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="train on ONE fixed target instead of random ones")
    p.add_argument("--timesteps", type=int, default=cfg.get("total_timesteps", 60_000))
    p.add_argument("--n-envs", type=int, default=cfg.get("n_envs", 1))
    p.add_argument("--seed", type=int, default=cfg.get("seed", 0))
    p.add_argument("--lr", type=float, default=cfg.get("learning_rate", 1e-3))
    p.add_argument("--gamma", type=float, default=cfg.get("gamma", 0.95))
    p.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 256))
    p.add_argument("--eval-episodes", type=int, default=cfg.get("eval_episodes", 20))
    p.add_argument("--eval-freq", type=int, default=cfg.get("eval_freq", 5_000))
    p.add_argument("--threads", type=int, default=cfg.get("threads", 0),
                   help="torch CPU threads (0 = torch default); use 1 when running seeds in parallel")
    p.add_argument("--quiet", action="store_true", help="no per-rollout log tables")
    p.add_argument("--out", default=None, help="output directory (default: runs/sb3)")
    p.add_argument("--device", default=cfg.get("device", "auto"))
    args = p.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    out = Path(args.out) if args.out else repo_root() / "runs" / "sb3"
    out.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import EvalCallback

    from mujoco_arm_rl.policies import save_env_config

    env_kwargs = parse_env_kwargs(args.env_kwarg)
    if args.target is not None:
        env_kwargs["fixed_target"] = tuple(args.target)
        print(f"[train_sb3] fixed target at {tuple(args.target)} (no target randomisation)")
    save_env_config(out, args.env_id, env_kwargs)

    env = build_env(args.env_id, args.n_envs, args.seed, env_kwargs)
    eval_env = build_env(args.env_id, 1, args.seed + 10_000, env_kwargs)

    try:
        import tensorboard  # noqa: F401
        tb_log = str(out / "tb")
    except ImportError:
        tb_log = None

    net_arch = cfg.get("net_arch", [256, 256])
    common = dict(policy="MlpPolicy", env=env, verbose=0 if args.quiet else 1, seed=args.seed,
                  device=args.device, tensorboard_log=tb_log, learning_rate=args.lr,
                  gamma=args.gamma)
    off_policy = dict(batch_size=args.batch_size,
                      buffer_size=cfg.get("buffer_size", 300_000),
                      learning_starts=cfg.get("learning_starts", 1_000),
                      train_freq=cfg.get("train_freq", 1),
                      gradient_steps=cfg.get("gradient_steps", 1),
                      tau=cfg.get("tau", 0.02))

    if args.algo == "sac":
        model = SAC(**common, **off_policy, policy_kwargs=dict(net_arch=net_arch))
    elif args.algo == "tqc":
        try:
            from sb3_contrib import TQC
        except ImportError:
            raise SystemExit("--algo tqc needs sb3-contrib: pip install sb3-contrib") from None
        model = TQC(**common, **off_policy, top_quantiles_to_drop_per_net=2,
                    policy_kwargs=dict(net_arch=net_arch, n_critics=2))
    elif args.algo == "td3":
        model = TD3(**common, **off_policy, policy_kwargs=dict(net_arch=net_arch))
    else:
        model = PPO(**common, batch_size=args.batch_size, n_steps=cfg.get("n_steps", 1024),
                    gae_lambda=cfg.get("gae_lambda", 0.95), ent_coef=cfg.get("ent_coef", 0.0),
                    policy_kwargs=dict(net_arch=net_arch))

    eval_cb = EvalCallback(eval_env, best_model_save_path=str(out), log_path=str(out / "eval"),
                           eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
                           n_eval_episodes=args.eval_episodes, deterministic=True, verbose=1)

    print(f"[train_sb3] {args.algo} on {args.env_id} {env_kwargs or ''} | "
          f"{args.timesteps} steps, seed {args.seed} -> {out}", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=False)
    model.save(str(out / f"{args.algo}_final"))
    env.close()
    eval_env.close()

    print(f"[train_sb3] done in {(time.time() - t0) / 60:.1f} min "
          f"({args.timesteps / (time.time() - t0):.0f} steps/s)")
    print(f"[train_sb3] best model : {out / 'best_model.zip'}")
    print(f"[train_sb3] watch it   : python scripts/evaluate.py --ckpt \"{out / 'best_model.zip'}\" --render human")
    print(f"[train_sb3] benchmark  : python scripts/benchmark.py --entry \"mine={out / 'best_model.zip'}\" --entry oracle")


if __name__ == "__main__":
    main()
