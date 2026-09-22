#!/usr/bin/env python
"""Open the MuJoCo viewer and drive the arm with random actions.

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
                   help="pin the target, e.g. --target 0.3 0.0 0.35 (handy for checking reachability)")
    p.add_argument("--torque", action="store_true",
                   help="use the legacy torque-control interface instead of the position servo")
    args = p.parse_args()

    if args.headless and args.episodes == 0:
        args.episodes = 3  # never loop forever without a window to close

    env = Reach3Env(render_mode=None if args.headless else "human",
                    control_mode="torque" if args.torque else "position",
                    fixed_target=None if args.target is None else tuple(args.target))
    realtime = not args.fast and not args.headless

    print(f"{env.control_mode} control at {1 / env.dt:.0f} Hz | obs {env.observation_space.shape} "
          f"| action {env.action_space.shape}")
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
        print(f"episode {ep}: return {total:7.2f}  final distance {1000 * info['distance']:.0f} mm")
        if not args.headless and not env.viewer_is_running:
            break

    print("\nRandom actions give roughly this; compare: python scripts/evaluate.py --ckpt oracle --render human")
    env.wait_for_viewer()
    env.close()


if __name__ == "__main__":
    main()
