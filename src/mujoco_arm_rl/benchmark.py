"""Frozen held-out evaluation for the reach task.

Why this exists: the training-time numbers SB3 prints are the mean over whatever
targets happened to be sampled, on one seed, mixed with exploration noise. That
is fine for watching a run but useless for deciding whether change A beat change
B. Here every policy -- whatever env version or control mode it trained on -- is
rolled out from the *same* 100 (start pose, target) pairs, stored in
``benchmarks/eval_set_v1.json`` and never regenerated, and several training
seeds are aggregated with a stratified bootstrap confidence interval.

Metrics (all measured at the end of the 2 s episode, i.e. "reach AND hold"):
    success@5cm, success@2cm   final tip-target distance under the threshold
    final distance             median, in mm
    time to reach              first time inside 5 cm (successful episodes)
    hold                       fraction of the final 0.5 s spent inside 5 cm
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_arm_rl.envs import Reach3Env
from mujoco_arm_rl.policies import Policy, resolve_env_kwargs
from mujoco_arm_rl.utils import repo_root

EVAL_SET_PATH = repo_root() / "benchmarks" / "eval_set_v1.json"
EVAL_SEED = 20260921
EVAL_SIZE = 100


def build_eval_set(n: int = EVAL_SIZE, seed: int = EVAL_SEED) -> list[dict[str, list[float]]]:
    env = Reach3Env()
    env.reset(seed=seed)
    cases = []
    for _ in range(n):
        cases.append({"qpos": env._sample_start().tolist(), "target": env._sample_target().tolist()})
    env.close()
    return cases


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict[str, list[float]]]:
    """The frozen eval set; created on first use, then only ever read."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        cases = build_eval_set()
        path.write_text(json.dumps({"seed": EVAL_SEED, "cases": cases}, indent=1))
    return json.loads(path.read_text())["cases"]


def rollout(policy: Policy, env_id: str, env_kwargs: dict[str, Any],
            cases: list[dict[str, list[float]]]) -> dict[str, np.ndarray]:
    kwargs = resolve_env_kwargs(env_id, env_kwargs)
    kwargs.pop("fixed_target", None)            # the eval set supplies targets
    env = Reach3Env(**kwargs)
    hold_steps = max(1, int(round(0.5 / env.dt)))
    out: dict[str, list[float]] = {k: [] for k in
                                   ("final_dist", "success_5cm", "success_2cm", "time_to_reach", "hold")}
    for i, case in enumerate(cases):
        obs, _ = env.reset(seed=i, options=case)
        dists, done, info = [], False, {}
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs, env))
            dists.append(info["distance"])
            done = term or trunc
        d = np.asarray(dists)
        inside = np.flatnonzero(d < 0.05)
        out["final_dist"].append(d[-1])
        out["success_5cm"].append(float(d[-1] < 0.05))
        out["success_2cm"].append(float(d[-1] < 0.02))
        out["time_to_reach"].append((inside[0] + 1) * env.dt if inside.size else np.nan)
        out["hold"].append(float(np.mean(d[-hold_steps:] < 0.05)))
    env.close()
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def stratified_bootstrap(per_seed: list[np.ndarray], stat=np.nanmean, n_boot: int = 2000,
                         rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Point estimate and 95 % CI, resampling seeds and then episodes within each seed."""
    rng = rng or np.random.default_rng(0)
    with warnings.catch_warnings():            # all-NaN columns (e.g. never reached) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        point = float(stat(np.concatenate(per_seed)))
        if len(per_seed) == 1 and per_seed[0].size < 2:
            return point, point, point
        boots = np.empty(n_boot)
        for b in range(n_boot):
            seeds = rng.integers(0, len(per_seed), size=len(per_seed))
            sample = [per_seed[s][rng.integers(0, per_seed[s].size, size=per_seed[s].size)]
                      for s in seeds]
            boots[b] = stat(np.concatenate(sample))
        if np.all(np.isnan(boots)):
            return point, float("nan"), float("nan")
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def summarise(per_seed: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    def col(key):
        return [r[key] for r in per_seed]
    s5 = stratified_bootstrap(col("success_5cm"))
    s2 = stratified_bootstrap(col("success_2cm"))
    fd = stratified_bootstrap(col("final_dist"), stat=np.nanmedian)
    tr = stratified_bootstrap(col("time_to_reach"))
    hold = stratified_bootstrap(col("hold"))
    per_seed_s5 = [float(np.mean(r["success_5cm"])) for r in per_seed]
    return {"n_seeds": len(per_seed), "success_5cm": s5, "success_2cm": s2,
            "final_dist_median": fd, "time_to_reach": tr, "hold": hold,
            "per_seed_success_5cm": per_seed_s5}


def format_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    def pct(t):
        p, lo, hi = t
        return f"{100*p:5.1f}% [{100*lo:.0f}–{100*hi:.0f}]"

    def mm(t):
        p, lo, hi = t
        return f"{1000*p:6.1f} [{1000*lo:.0f}–{1000*hi:.0f}]"

    def sec(t):
        p, lo, hi = t
        return "—" if np.isnan(p) else f"{p:.2f} [{lo:.2f}–{hi:.2f}]"

    lines = ["| Policy | Seeds | Success @5 cm | Success @2 cm | Median final dist (mm) | Time to reach (s) | Hold |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for label, s in rows:
        lines.append(f"| {label} | {s['n_seeds']} | {pct(s['success_5cm'])} | {pct(s['success_2cm'])} | "
                     f"{mm(s['final_dist_median'])} | {sec(s['time_to_reach'])} | {pct(s['hold'])} |")
    return "\n".join(lines)
