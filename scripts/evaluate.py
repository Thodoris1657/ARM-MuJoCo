#!/usr/bin/env python
"""Roll out a trained policy (SB3 .zip or TorchRL .pt) and watch or measure it.

The env it trained on is read from env_config.json next to the checkpoint, so
you never have to repeat --target or control-mode flags. For a rigorous number
with confidence intervals use scripts/benchmark.py instead.

Examples
--------
    python scripts/evaluate.py --ckpt runs/sb3/best_model.zip --render human
    python scripts/evaluate.py --ckpt runs/torchrl/best_model.pt --episodes 50
    python scripts/evaluate.py --ckpt runs/sb3/best_model.zip --video runs/sb3/demo.mp4
    python scripts/evaluate.py --ckpt oracle --render human     # scripted IK reference
    python scripts/evaluate.py --ckpt random --episodes 20      # untrained baseline
    python scripts/evaluate.py --ckpt runs/sb3/best_model.zip --seed 124 --episodes 1  # replay one
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from mujoco_arm_rl.envs import Reach3Env  # noqa: E402
from mujoco_arm_rl.policies import load_policy, resolve_env_kwargs  # noqa: E402
from mujoco_arm_rl.utils import describe_rollout, set_seed  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None,
                   help='checkpoint path (.zip = SB3, .pt = TorchRL), or "oracle" / "random"')
    p.add_argument("--algo", default=None, choices=["sb3", "torchrl", "random", "oracle"],
                   help="optional: the checkpoint type is detected automatically")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--render", default="none", choices=["none", "human"])
    p.add_argument("--video", default=None, help="write an mp4 to this path (off-screen render)")
    p.add_argument("--fps", type=int, default=None, help="video fps (default: the control rate)")
    p.add_argument("--fast", action="store_true", help="with --render human, skip realtime pacing")
    p.add_argument("--target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="override the target (default: whatever the policy trained on)")
    args = p.parse_args()

    spec = args.ckpt or (args.algo if args.algo in ("random", "oracle") else None)
    if spec is None:
        p.error("--ckpt is required (a checkpoint path, or 'oracle' / 'random')")
    set_seed(args.seed)

    act, env_id, env_kwargs, desc = load_policy(spec)
    kwargs = resolve_env_kwargs(env_id, env_kwargs)
    if args.target is not None:
        kwargs["fixed_target"] = tuple(args.target)
    print(f"[evaluate] {desc}")
    print(f"[evaluate] env {env_id} {env_kwargs or ''}")

    render_mode = "rgb_array" if args.video else (None if args.render == "none" else "human")
    env = Reach3Env(render_mode=render_mode, **kwargs)
    realtime = render_mode == "human" and not args.fast

    frames: list[np.ndarray] = []
    returns, successes, final_distances = [], [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total, done, info = 0.0, False, {}
        while not done:
            obs, reward, terminated, truncated, info = env.step(act(obs, env))
            total += reward
            if args.video:
                frames.append(env.render())
            if realtime:
                time.sleep(env.dt)
            done = terminated or truncated
            if render_mode == "human" and not env.viewer_is_running:
                done = True
        returns.append(total)
        successes.append(info["is_success"])
        final_distances.append(info["distance"])
        reach = info.get("time_to_reach")
        print(f"  episode {ep + 1:3d} (seed {args.seed + ep}): return {total:7.2f} | "
              f"final distance {1000 * info['distance']:6.1f} mm | "
              f"{f'reached in {reach:.2f} s' if reach else 'never within 5 cm'}")
        if render_mode == "human" and not env.viewer_is_running:
            break

    print("\n[evaluate] " + describe_rollout(returns, successes, final_distances))
    env.wait_for_viewer()
    env.close()

    if args.video:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise SystemExit("writing video needs imageio: pip install 'imageio[ffmpeg]'") from exc
        out = Path(args.video)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out), frames, fps=args.fps or int(round(1.0 / env.dt)))
        print(f"[evaluate] wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
