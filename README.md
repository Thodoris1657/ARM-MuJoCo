# mujoco-arm-rl

Reinforcement learning on a **3-joint MuJoCo manipulator** that must bring its
end-effector onto a random target anywhere in its workspace **and hold it
there**. One environment, two trainers:

| Trainer | Script | What it gives you |
| --- | --- | --- |
| **Stable-Baselines3** (SAC / TQC) | `scripts/train_sb3.py` | Batteries included; the measured-best recipe. |
| **TorchRL** (clipped PPO) | `scripts/train_torchrl.py` | Every piece explicit — collector, GAE, PPO epochs, optimiser step. |

**Headline** (held-out benchmark, 100 fixed start/target pairs): SAC reaches
**100 % success at 5 cm and 95–98 % at 2 cm, with a median final error of ~6 mm**,
after 30k steps (~7 min on 2 CPU cores). The analytic-IK reference controller
scores 5.7 mm. The original version of this repo scored **0–1 %** on the same
benchmark.

---

## Quick start

```bash
conda create -n mujoco-rl python=3.11 -y && conda activate mujoco-rl
pip install -r requirements.txt

python scripts/view_random.py                                   # does MuJoCo work?
python scripts/evaluate.py --ckpt oracle --render human         # what "good" looks like
python scripts/train_sb3.py --timesteps 30000 --out runs/sb3_v1 # ~7 min on CPU
python scripts/evaluate.py --ckpt runs/sb3_v1/best_model.zip --render human
python scripts/benchmark.py --entry "mine=runs/sb3_v1/best_model.zip" --entry oracle
```

`pytest -q` runs 16 environment contract tests. On a headless machine, set
`MUJOCO_GL=disable` for anything that does not open a window.

---

## The robot

`assets/arm3.xml`: shoulder pan (z), shoulder pitch (y), elbow (y). The pitch
joint sits at **(0, 0, 0.22)** and the tip reaches **0.50 m** from it; the elbow
limit of ±2.6 rad means nothing closer than ~0.13 m to the shoulder is reachable.
Gravity is on. The green sphere is a mocap body used as the target.

## The task: `MujocoArm3Reach-v1`

| | |
| --- | --- |
| **Control** | Policy outputs *changes* to joint targets, rate-limited to 3 rad/s; a PD servo inside MuJoCo (kp = 200/300/150, critically damped from the joint-space inertia) turns them into torque, capped at the motor limits. |
| **Rate** | 25 Hz decisions, 2 s episodes (50 steps), never terminated early — reach *and hold*. |
| **Goals** | Uniform by volume in the workspace, kept only if the analytic IK finds a collision-free solution inside the joint limits. |
| **Observation** (24) | `cos q`, `sin q`, `q̇/10`, servo error, previous action, tip position, target position, `10·(target − tip)` |
| **Reward** | `−d + (1 − tanh(d / 5 cm)) − 0.02‖a‖² − 0.02‖a − a_prev‖²` |

The linear term pulls from far away; the `tanh` term gives a strong gradient in
the last few centimetres, which is where precision comes from; the two small
action penalties make the arm settle instead of dither.

`MujocoArm3Reach-v0` is the original task (raw torques at 100 Hz, 200-step
episodes, the old sampler and reward, 18-dim obs). It is kept so old checkpoints
still load — `evaluate.py` recognises them automatically — and so the ablation
is reproducible. `--env-kwarg control_mode=torque` etc. switches any piece back.

---

## What changed, and what each change was worth

### Two bugs found by auditing the task, not by tuning

1. **A permanent contact on the pan joint.** link1's capsule sat 4.5 cm inside
   the base cylinder. MuJoCo skips parent–child collisions, *except* when the
   parent is welded to the world — which the base is. So every step of every
   episode had a penetrating contact adding friction to joint 1. Fixed with a
   `<contact><exclude>` in the MJCF; regression-tested.
2. **4.3 % of training targets were physically unreachable.** The old sampler
   centred its shell on joint 1 (z = 0.10) instead of the pitch joint
   (z = 0.22), so some targets landed inside the elbow's minimum reach. Found by
   writing an analytic IK (verified against MuJoCo FK to 1e-15 m) and checking
   20,000 samples. The new sampler only produces targets IK says are reachable.

### Ablation

SAC, 30k agent steps per run (equal gradient updates = equal compute), scored on
the held-out benchmark. **Seed 0 only** — the multi-seed run was stopped part way,
so treat single-point differences of a few percent as noise. Rows are cumulative.

| Config | Success @5 cm | Success @2 cm | Median final dist | Time to reach |
| --- | --- | --- | --- | --- |
| A — original recipe (v0 task, lr 3e-4, γ 0.98) | 1 % | 0 % | 165 mm | — |
| B — + tuned SAC (lr 1e-3, γ 0.95) | 0 % | 0 % | 249 mm | — |
| C — + 25 Hz, reachable goals, richer obs (still torque) | 97 % | 49 % | 20 mm | 0.30 s |
| D — + position servo | 98 % | 48 % | 20 mm | 0.53 s |
| **E — + shaped reward (= v1)** | **100 %** | **95 %** | **5.8 mm** | 0.52 s |
| IK oracle (scripted reference) | 100 % | 100 % | 5.7 mm | 0.52 s |
| random actions | 0 % | 0 % | 539 mm | — |

What the data says — including where it contradicted the plan:

- **The task formulation was the big lever (C), not the algorithm (B).** Tuning
  SAC on the old task did nothing. Changing the control rate, goal distribution
  and observation took success from ~0 % to 97 %. Row C bundles three changes;
  `scripts/ablation.py` has a `C0_rate` config that isolates the 25 Hz change,
  not yet run.
- **The position servo did not help accuracy** (D ≈ C), and it made reaching
  *slower* (0.53 s vs 0.30 s), because the 3 rad/s rate limit is the bottleneck.
  It is still the default because it gives smooth, bounded motion and it is how
  real arms are commanded, but that is a design choice, not a measured win.
  `X_fast` (6 rad/s limit) and `X_torque` (v1 with raw torques) in the ablation
  script test this directly; not yet run.
- **The shaped reward is what buys precision (E):** success at 2 cm 48 % → 95 %,
  median error 20 mm → 5.8 mm, essentially matching the IK reference.

### Hyperparameters (SAC on v1, 30k steps, seed 0)

| lr | γ | @5 cm | @2 cm | Median |
| --- | --- | --- | --- | --- |
| 3e-4 | 0.98 | 97 % | 71 % | 14.6 mm |
| **1e-3** | **0.95** | **100 %** | **98 %** | **6.3 mm** |

At 25 Hz, γ = 0.95 is a ~0.8 s effective horizon, matching a ~0.5 s reach.

---

## Evaluating properly

```bash
python scripts/benchmark.py \
    --entry "SAC=runs/seed*/best_model.zip" \
    --entry oracle --entry random --out results.json
```

- **Frozen held-out set**: `benchmarks/eval_set_v1.json`, 100 (start pose,
  target) pairs, generated once and never regenerated.
- **Metrics at the end of the 2 s episode**: success at 5 cm and 2 cm, median
  final distance, time to first reach, fraction of the last 0.5 s held inside 5 cm.
- **Multiple seeds**: every checkpoint matched by one `--entry` glob is a seed;
  brackets are 95 % stratified-bootstrap CIs (seeds, then episodes within seeds).
- **The IK oracle** — analytic IK driven through the same rate-limited action
  interface — is the reference ceiling: if a policy is far from it, the problem
  is learning, not physics.

Training-time numbers from SB3's logger are a single seed on random targets
with exploration noise. Use them to watch a run, not to compare runs.

## Training

```bash
python scripts/train_sb3.py                       # v1, 60k steps (configs/sb3_sac.yaml)
python scripts/train_sb3.py --algo tqc            # TQC from sb3-contrib
python scripts/train_sb3.py --target 0.3 0.0 0.35 # one fixed goal
python scripts/train_torchrl.py                   # PPO, 300k frames, keeps best held-out checkpoint
python scripts/ablation.py --jobs 2               # reproduce the table above (all seeds)
```

Each run writes `env_config.json` beside its checkpoints recording the env ID
and kwargs, and `evaluate.py` / `benchmark.py` read it back. A policy is always
evaluated on the task it trained on — no repeating `--target` by hand.

`--threads 1` when running several seeds in parallel; on 2 cores, two
single-threaded runs give ~2× the throughput of one.

**TorchRL status:** the rewritten PPO trainer (held-out evaluation, best
checkpoint, advantage normalisation, v1 task) is smoke-tested end to end but has
**not** had a full-length run yet, so there is no measured PPO number on v1.

### A note on TorchRL versions

TorchRL renames public API between releases (`SyncDataCollector` → `Collector`,
`entropy_coef` → `entropy_coeff`). `train_torchrl.py` resolves both at import
time rather than pinning a version. TorchRL checkpoints are reloaded in plain
PyTorch by replaying the saved `ObservationNorm` statistics; the checkpoint
records which convention was used (`obs * scale + loc` by default), because
getting it backwards silently yields a policy that scores worse than random.

---

## Layout

```
mujoco-arm-rl/
├── assets/arm3.xml                 MJCF model (with the base/link1 contact fix)
├── benchmarks/eval_set_v1.json     frozen held-out evaluation set
├── configs/                        trainer defaults (measured, see above)
├── src/mujoco_arm_rl/
│   ├── envs/reach3.py              env: servo, IK, goal sampling, reward
│   ├── envs/__init__.py            registers MujocoArm3Reach-v1 and -v0
│   ├── policies.py                 load SB3 / TorchRL / oracle / random behind one API
│   ├── benchmark.py                rollouts, metrics, stratified bootstrap
│   └── utils.py
├── scripts/
│   ├── view_random.py              viewer + random actions (start here)
│   ├── train_sb3.py                SAC / TQC / TD3 / PPO via Stable-Baselines3
│   ├── train_torchrl.py            clipped PPO written out in full
│   ├── evaluate.py                 watch a policy, record mp4
│   ├── benchmark.py                score policies with confidence intervals
│   └── ablation.py                 reproduce the ablation table
└── tests/test_env.py               16 contract tests
```

## License

MIT — see [LICENSE](LICENSE).
