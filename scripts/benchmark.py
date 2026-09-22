#!/usr/bin/env python
"""Score policies on the frozen 100-case held-out set, with confidence intervals.

Each --entry is LABEL=GLOB; every checkpoint the glob matches counts as one
training seed of that entry. "oracle" and "random" are built-in references.

Examples
--------
    python scripts/benchmark.py --entry oracle --entry random
    python scripts/benchmark.py --entry "SAC v1=runs/sb3/best_model.zip"
    python scripts/benchmark.py \\
        --entry "v0 recipe=runs/ablation/A_*/best_model.zip" \\
        --entry "v1 recipe=runs/ablation/D_*/best_model.zip" \\
        --entry oracle --out runs/ablation/results.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mujoco_arm_rl.benchmark import format_table, load_eval_set, rollout, summarise  # noqa: E402
from mujoco_arm_rl.policies import load_policy  # noqa: E402


def parse_entry(text: str) -> tuple[str, list[str]]:
    if text in ("oracle", "random"):
        return {"oracle": "IK oracle (scripted)", "random": "random actions"}[text], [text]
    if "=" not in text:
        label, pattern = Path(text).parent.name or text, text
    else:
        label, pattern = text.split("=", 1)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"--entry {text!r}: no checkpoint matches {pattern!r}")
    return label.strip(), paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entry", action="append", required=True, help='"LABEL=GLOB", "oracle" or "random"')
    p.add_argument("--cases", type=int, default=None, help="use only the first N eval cases (quick look)")
    p.add_argument("--out", default=None, help="also write raw + summary results to this JSON file")
    args = p.parse_args()

    cases = load_eval_set()
    if args.cases:
        cases = cases[: args.cases]

    rows, raw = [], {}
    for text in args.entry:
        label, specs = parse_entry(text)
        per_seed = []
        for spec in specs:
            act, env_id, env_kwargs, desc = load_policy(spec)
            res = rollout(act, env_id, env_kwargs, cases)
            per_seed.append(res)
            print(f"  {label:<28} {desc:<60} success@5cm {100*res['success_5cm'].mean():5.1f}%  "
                  f"@2cm {100*res['success_2cm'].mean():5.1f}%", flush=True)
        summary = summarise(per_seed)
        rows.append((label, summary))
        raw[label] = {"checkpoints": specs, "summary": summary,
                      "per_seed": [{k: v.tolist() for k, v in r.items()} for r in per_seed]}

    print(f"\nHeld-out set: {len(cases)} (start pose, target) pairs, 2 s episodes, "
          f"metrics at episode end. Brackets = 95% stratified-bootstrap CI.\n")
    print(format_table(rows))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(raw, indent=1, default=float))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
