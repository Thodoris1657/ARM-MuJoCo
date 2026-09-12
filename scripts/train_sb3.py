#!/usr/bin/env python
"""Train the 3-joint reacher with Stable-Baselines3 (SAC by default).

Examples
--------
    python scripts/train_sb3.py                       # defaults from configs/sb3_sac.yaml
    python scripts/train_sb3.py --timesteps 300000
    python scripts/train_sb3.py --algo ppo --n-envs 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mujoco_arm_rl.utils import load_config, repo_root, run_dir, set_seed  # noqa: E402


def build_env(n_envs: int, seed: int, env_kwargs: dict):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor  # noqa: F401
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    import mujoco_arm_rl.envs  # noqa: F401  (registers MujocoArm3Reach-v0)

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return make_vec_env(
        "MujocoArm3Reach-v0",
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_cls,
    )


def main() -> None:
    cfg = load_config(repo_root() / "configs" / "sb3_sac.yaml")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", default=cfg.get("algo", "sac"), choices=["sac", "ppo", "td3"])
    p.add_argument("--timesteps", type=int, default=cfg.get("total_timesteps", 200_000))
    p.add_argument("--n-envs", type=int, default=cfg.get("n_envs", 1))
    p.add_argument("--seed", type=int, default=cfg.get("seed", 0))
    p.add_argument("--lr", type=float, default=cfg.get("learning_rate", 3e-4))
    p.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 256))
    p.add_argument("--eval-episodes", type=int, default=cfg.get("eval_episodes", 10))
    p.add_argument("--eval-freq", type=int, default=cfg.get("eval_freq", 10_000))
    p.add_argument("--out", default=None, help="output directory (default: runs/sb3)")
    p.add_argument("--device", default=cfg.get("device", "auto"))
    p.add_argument("--target", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="train/evaluate on ONE fixed target instead of random ones, e.g. --target 0.3 0.0 0.35")
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.out) if args.out else run_dir("sb3")

    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    env_kwargs = {} if args.target is None else {"fixed_target": tuple(args.target)}
    if args.target is not None:
        print(f"[train_sb3] fixed target at {tuple(args.target)} (no target randomisation)")
    env = build_env(args.n_envs, args.seed, env_kwargs=env_kwargs)
    eval_env = build_env(1, args.seed + 10_000, env_kwargs=env_kwargs)

    # TensorBoard is optional: log to it when it is installed, stay quiet otherwise.
    try:
        import tensorboard  # noqa: F401

        tb_log = str(out / "tb")
    except ImportError:
        tb_log = None
        print("[train_sb3] tensorboard not installed -- skipping TB logging "
              "(pip install tensorboard to enable)")

    common = dict(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=tb_log,
        learning_rate=args.lr,
    )

    if args.algo == "sac":
        model = SAC(
            **common,
            batch_size=args.batch_size,
            buffer_size=cfg.get("buffer_size", 300_000),
            learning_starts=cfg.get("learning_starts", 2_000),
            train_freq=cfg.get("train_freq", 1),
            gradient_steps=cfg.get("gradient_steps", 1),
            gamma=cfg.get("gamma", 0.98),
            tau=cfg.get("tau", 0.02),
            policy_kwargs=dict(net_arch=cfg.get("net_arch", [256, 256])),
        )
    elif args.algo == "td3":
        model = TD3(
            **common,
            batch_size=args.batch_size,
            buffer_size=cfg.get("buffer_size", 300_000),
            learning_starts=cfg.get("learning_starts", 2_000),
            gamma=cfg.get("gamma", 0.98),
            policy_kwargs=dict(net_arch=cfg.get("net_arch", [256, 256])),
        )
    else:
        model = PPO(
            **common,
            batch_size=args.batch_size,
            n_steps=cfg.get("n_steps", 1024),
            gamma=cfg.get("gamma", 0.99),
            gae_lambda=cfg.get("gae_lambda", 0.95),
            ent_coef=cfg.get("ent_coef", 0.0),
            policy_kwargs=dict(net_arch=cfg.get("net_arch", [256, 256])),
        )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(out),
            log_path=str(out / "eval"),
            eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
        ),
        CheckpointCallback(
            save_freq=max(50_000 // max(args.n_envs, 1), 1),
            save_path=str(out / "checkpoints"),
            name_prefix=args.algo,
        ),
    ]

    print(f"[train_sb3] algo={args.algo} timesteps={args.timesteps} n_envs={args.n_envs} -> {out}")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    dt = time.time() - t0

    final_path = out / f"{args.algo}_final"
    model.save(str(final_path))
    env.close()
    eval_env.close()

    print(f"[train_sb3] done in {dt/60:.1f} min")
    print(f"[train_sb3] final model : {final_path}.zip")
    print(f"[train_sb3] best model  : {out / 'best_model.zip'}")
    print(f"[train_sb3] evaluate    : python scripts/evaluate.py --algo sb3 "
          f"--ckpt \"{out / 'best_model.zip'}\" --render human")


if __name__ == "__main__":
    main()
