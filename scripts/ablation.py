#!/usr/bin/env python
"""Reproduce the README ablation: what each design change is worth.

Trains every (config, seed) pair with SAC at an identical step budget, running
``--jobs`` processes in parallel (one torch thread each), then scores all of
them on the frozen held-out set. Finished runs are skipped, so an interrupted
ablation resumes where it stopped.

    python scripts/ablation.py                    # 5 configs x 3 seeds x 30k steps
    python scripts/ablation.py --seeds 0 1 --steps 20000 --jobs 4
    python scripts/ablation.py --only E_v1        # just the final recipe
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A..E are cumulative: each row adds one change on top of the previous one.
# X_* rows are side experiments that each change one thing relative to E.
V0 = ["--env-id", "MujocoArm3Reach-v0"]
CONFIGS = {
    "A_original": ("original recipe (v0 task, lr 3e-4, gamma 0.98)",
                   [*V0, "--lr", "3e-4", "--gamma", "0.98"]),
    "B_hparams": ("+ tuned SAC (lr 1e-3, gamma 0.95)",
                  [*V0, "--lr", "1e-3", "--gamma", "0.95"]),
    "C0_rate": ("+ 25 Hz decisions instead of 100 Hz",
                [*V0, "--env-kwarg", "control_hz=25.0"]),
    "C_task": ("+ reachable goals, richer obs (still torque)",
               ["--env-kwarg", "control_mode=torque", "--env-kwarg", "reward_mode=legacy"]),
    "D_servo": ("+ position servo control",
                ["--env-kwarg", "reward_mode=legacy"]),
    "E_v1": ("+ shaped reward  (= v1 default)", []),
    "X_torque": ("side: v1 but raw torque control",
                 ["--env-kwarg", "control_mode=torque"]),
    "X_fast": ("side: v1 with a 6 rad/s servo limit",
               ["--env-kwarg", "max_joint_speed=6.0"]),
}


def train(name: str, seed: int, steps: int, out_root: Path) -> str:
    out = out_root / f"{name}_s{seed}"
    if (out / "best_model.zip").exists() and any(out.glob("*_final.zip")):
        return f"skip   {out.name} (already trained)"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "train_sb3.py"), *CONFIGS[name][1],
           "--timesteps", str(steps), "--seed", str(seed), "--threads", "1", "--quiet",
           "--eval-freq", str(max(steps // 6, 1000)), "--out", str(out)]
    t0 = time.time()
    with (out / "train.log").open("w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    status = "ok    " if rc == 0 else f"FAILED ({rc})"
    return f"{status} {out.name} in {(time.time() - t0) / 60:.1f} min"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--jobs", type=int, default=2, help="parallel training processes")
    p.add_argument("--only", nargs="+", choices=list(CONFIGS), default=None)
    p.add_argument("--out", default=str(ROOT / "runs" / "ablation"))
    p.add_argument("--skip-train", action="store_true", help="only (re)run the benchmark")
    args = p.parse_args()

    names = args.only or list(CONFIGS)
    out_root = Path(args.out)
    if not args.skip_train:
        jobs = [(n, s) for s in args.seeds for n in names]  # seed-major: early rows fill in first
        print(f"[ablation] {len(jobs)} runs, {args.steps} steps each, {args.jobs} in parallel", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for msg in pool.map(lambda j: train(j[0], j[1], args.steps, out_root), jobs):
                print(f"[ablation] {msg}", flush=True)

    entries = []
    for n in names:
        entries += ["--entry", f"{n}: {CONFIGS[n][0]}={out_root}/{n}_s*/best_model.zip"]
    entries += ["--entry", "oracle", "--entry", "random"]
    subprocess.run([sys.executable, str(ROOT / "scripts" / "benchmark.py"), *entries,
                    "--out", str(out_root / "results.json")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
