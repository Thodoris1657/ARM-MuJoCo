#!/usr/bin/env python
"""Open the MuJoCo viewer and drive the arm with random torques.

Use this first: it proves MuJoCo, the MJCF model and the environment all work
on your machine before any RL is involved.

Playback runs at realtime and the window stays open when the rollout ends, so
you actually get to watch it. Episodes repeat until you close the window.

    python scripts/view_random.py              # loop until you close the window
    python scripts/view_random.py --episodes 3 # stop after 3, then hold the window
    python scripts/view_random.py --fast       # no pacing (finishes in a blink)
    python scripts/view_random.py --headless   # no window at all (CI / servers)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mujoco_arm_rl.envs import Reach3Env  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=0,
                   help="number of episodes; 0 (default) means loop until the window closes")
    p.add_argument("--headless", action="store_true", help="no viewer window (CI / servers)")
    p.add_argument("--fast", action="store_true",
                   help="run as fast as possible instead of at realtime speed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="train/evaluate on ONE fixed target instead of random ones, e.g. --target 0.3 0.0 0.35")
    args = p.parse_args()

    if args.headless and args.episodes == 0:
        args.episodes = 3  # never loop forever without a window to close

    env = Reach3Env(render_mode=None if args.headless else "human",
                    fixed_target=None if args.target is None else tuple(args.target))
    realtime = not args.fast and not args.headless

    print(f"observation space {env.observation_space.shape}  "
          f"action space {env.action_space.shape}  dt {env.dt:.3f}s")
    if args.episodes == 0:
        print("Looping episodes -- close the viewer window to stop.")

    ep = 0
    while args.episodes == 0 or ep < args.episodes:
        obs, info = env.reset(seed=args.seed + ep)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total += reward
            done = terminated or truncated
            if realtime:
                time.sleep(env.dt)
            if not args.headless and not env.viewer_is_running:
                done = True  # the user closed the window mid-episode
        ep += 1
        print(f"episode {ep}: return {total:8.2f}  final distance {info['distance']:.3f} m")
        if not args.headless and not env.viewer_is_running:
            break

    print("\nRandom torques give roughly this return; a trained policy should be far higher.")
    env.wait_for_viewer()
    env.close()


if __name__ == "__main__":
    main()
