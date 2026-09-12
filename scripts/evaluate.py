#!/usr/bin/env python
"""Roll out a trained policy (SB3 or TorchRL) and report how well it reaches.

Examples
--------
    # watch it live in the MuJoCo viewer
    python scripts/evaluate.py --algo sb3 --ckpt runs/sb3/best_model.zip --render human

    # headless benchmark over 50 episodes
    python scripts/evaluate.py --algo torchrl --ckpt runs/torchrl/ppo_final.pt --episodes 50

    # write an mp4
    python scripts/evaluate.py --algo sb3 --ckpt runs/sb3/best_model.zip --video runs/sb3/demo.mp4

    # untrained baseline, for comparison
    python scripts/evaluate.py --algo random --episodes 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from mujoco_arm_rl.envs import Reach3Env  # noqa: E402
from mujoco_arm_rl.utils import describe_rollout, set_seed  # noqa: E402

# ------------------------------------------------------------------- policies


def load_sb3_policy(ckpt: str):
    from stable_baselines3 import PPO, SAC, TD3

    last_err = None
    for cls in (SAC, PPO, TD3):
        try:
            model = cls.load(ckpt, device="cpu")
            print(f"[evaluate] loaded {cls.__name__} from {ckpt}")
            return lambda obs: model.predict(obs, deterministic=True)[0]
        except Exception as err:  # noqa: BLE001 - try the next algorithm class
            last_err = err
    raise RuntimeError(f"could not load {ckpt} as a SAC/PPO/TD3 model: {last_err}")


def load_torchrl_policy(ckpt: str):
    import torch
    from torch import nn

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    hidden, n_obs, n_act = blob["hidden"], blob["n_obs"], blob["n_act"]

    # Same trunk as train_torchrl.py, minus the NormalParamExtractor tail:
    # the network outputs [loc | raw_scale]; the deterministic action is tanh(loc).
    net = nn.Sequential(
        nn.Linear(n_obs, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 2 * n_act),
    )
    # NormalParamExtractor has no parameters, so the layer indices line up exactly.
    net.load_state_dict(blob["policy_state_dict"])
    net.eval()

    # Replay the training-time ObservationNorm exactly: TorchRL computes
    # (obs - loc) / scale when standard_normal is set, and obs * scale + loc otherwise.
    loc = blob["obs_loc"].float().numpy()
    scale = blob["obs_scale"].float().numpy()
    standard_normal = bool(blob.get("obs_standard_normal", False))
    print(f"[evaluate] loaded TorchRL PPO policy from {ckpt} "
          f"(obs norm: {'standard_normal' if standard_normal else 'affine'})")

    def policy(obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        x = (x - loc) / scale if standard_normal else x * scale + loc
        with torch.no_grad():
            out = net(torch.as_tensor(x, dtype=torch.float32))
        mean = out[..., :n_act]
        return np.tanh(mean.numpy())

    return policy


# ----------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", default="sb3", choices=["sb3", "torchrl", "random"])
    p.add_argument("--ckpt", default=None, help="path to the saved model")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--render", default="none", choices=["none", "human"])
    p.add_argument("--video", default=None, help="write an mp4 to this path (off-screen render)")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--fast", action="store_true",
                   help="with --render human, skip realtime pacing")
    p.add_argument("--target", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="train/evaluate on ONE fixed target instead of random ones, e.g. --target 0.3 0.0 0.35")
    args = p.parse_args()

    set_seed(args.seed)

    if args.algo == "random":
        policy = None
    elif args.ckpt is None:
        p.error("--ckpt is required unless --algo random")
    elif args.algo == "sb3":
        policy = load_sb3_policy(args.ckpt)
    else:
        policy = load_torchrl_policy(args.ckpt)

    render_mode = "rgb_array" if args.video else (args.render if args.render != "none" else None)
    env = Reach3Env(render_mode=render_mode,
                    fixed_target=None if args.target is None else tuple(args.target))
    # Without pacing, 200 steps flash past in a fraction of a second.
    realtime = render_mode == "human" and not args.fast

    frames: list[np.ndarray] = []
    returns, successes, final_distances = [], [], []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total, done, hits = 0.0, False, 0
        while not done:
            action = env.action_space.sample() if policy is None else policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            hits += int(info["is_success"])
            if args.video:
                frames.append(env.render())
            if realtime:
                time.sleep(env.dt)
            done = terminated or truncated
            if render_mode == "human" and not env.viewer_is_running:
                done = True  # the user closed the window
        returns.append(total)
        successes.append(hits > 0)
        final_distances.append(info["distance"])
        print(f"  episode {ep + 1:3d}: return {total:8.2f} | "
              f"steps on target {hits:3d} | final distance {info['distance']:.4f} m")

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
        imageio.mimsave(str(out), frames, fps=args.fps)
        print(f"[evaluate] wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
