# mujoco-arm-rl

Reinforcement learning on a **3-joint MuJoCo manipulator** that has to drive its
end-effector onto a randomly placed target. The same environment is trained two
ways so you can compare the ecosystems side by side:

| Trainer | Script | What it gives you |
| --- | --- | --- |
| **Stable-Baselines3** (SAC) | `scripts/train_sb3.py` | Batteries included. One `.learn()` call, solid defaults, TensorBoard logging. |
| **TorchRL** (clipped PPO) | `scripts/train_torchrl.py` | Every piece is explicit — collector, GAE, PPO epochs, optimiser step. |

---

## The robot

`assets/arm3.xml` is a 3-DOF arm, torque-controlled:

| Joint | Axis | Range (rad) | Role |
| --- | --- | --- | --- |
| `joint1` | z | ±3.14 | shoulder pan |
| `joint2` | y | ±2.0 | shoulder lift |
| `joint3` | y | ±2.6 | elbow |

Links are 0.25 m each, so the reachable workspace is a spherical shell of radius
≈ 0.5 m around the shoulder. Gravity is on, so holding a pose costs torque —
the policy has to learn to fight it, not just point in the right direction.
The translucent green sphere is a **mocap body** used as the target; it is moved
by writing to `data.mocap_pos[0]`, which is the cheapest way to reposition
something between episodes without touching the physics state.

## The task

| | |
| --- | --- |
| **Observation** (18) | `cos(q)`, `sin(q)`, `qvel/10`, end-effector position, target position, target − end-effector |
| **Action** (3) | normalised joint torques in `[-1, 1]`, scaled by the MJCF `gear` values |
| **Reward** | `−distance + 1.0·[distance < 5 cm] − 0.01‖a‖² − 0.001‖q̇‖²` |
| **Episode** | 200 control steps at `dt = 0.01 s` (2 s), always truncated, never terminated early |

Angles enter the observation as `cos`/`sin` rather than raw radians so the
network never sees a discontinuity at ±π. The success bonus is what turns a
"get close" policy into a "park on the target and stay there" policy: it pays
per timestep spent inside the 5 cm ball, so a good policy reaches fast and holds.

### Measured results

Short smoke runs on 2 CPU cores, evaluated deterministically over 20 held-out
episodes (seeds 123–142). Both trainers clearly beat the baseline; neither run
was long enough to converge, so treat these as a floor, not a ceiling.

| Policy | Training budget | Return (20 eval episodes) | Mean final distance |
| --- | --- | --- | --- |
| random torques | — | **−91.4** ± 21.5 | 0.399 m |
| SB3 SAC | 25k steps (6 min) | **−57.4** ± 16.9 | 0.253 m |
| TorchRL PPO | 120k frames (5 min) | **−53.5** ± 18.9 | 0.221 m |

Both configs default to longer runs (200k steps / 200k frames), which is where
the arm starts parking inside the 5 cm ball instead of merely drifting toward it.

### One fixed target instead of random ones

Pass `--target X Y Z` to any script to pin the goal to a single point. The task
gets dramatically easier, because the policy no longer has to generalise across
the whole workspace:

```bash
python scripts/train_sb3.py --timesteps 20000 --target 0.3 0.0 0.35
python scripts/evaluate.py --algo sb3 --ckpt runs/sb3/best_model.zip \
    --target 0.3 0.0 0.35 --render human
```

| Task | Budget | Eval return | Success rate |
| --- | --- | --- | --- |
| random targets | 25k steps | −57.4 | 0% |
| **fixed target** `0.3 0.0 0.35` | 20k steps | **+48.2** | **60%** |

Same budget, same algorithm. Use `--out runs/sb3_fixed` to keep the checkpoints
separate from a random-target run.

Coordinates are metres in world frame, with the shoulder at `(0, 0, 0.10)` and a
maximum reach of 0.535 m. An unreachable point is rejected at startup with an
explanation rather than quietly training on an impossible goal. **Whatever
`--target` you train with, pass the same one to `evaluate.py`** — the target
position is part of the observation, so a policy trained on one point will not
transfer to another.

---

## Install

```bash
git clone <your-repo-url> mujoco-arm-rl
cd mujoco-arm-rl

# conda (matches the VS Code setting already in your MuJoCo folder)
conda create -n mujoco-rl python=3.11 -y
conda activate mujoco-rl

# ...or plain venv
# python -m venv .venv && .venv\Scripts\activate      # Windows
# python -m venv .venv && source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
pip install -e .            # optional: makes `import mujoco_arm_rl` work anywhere
```

> On a GPU machine, install the CUDA build of PyTorch **first**
> (see pytorch.org), then `pip install -r requirements.txt`.

## Sanity check — no RL yet

```bash
python scripts/view_random.py
```

A viewer window opens and the arm flails under random torques, at realtime
speed, looping until you close the window. If that works, MuJoCo, the model and
the environment are all fine. Use `--episodes 3` to stop after a few (the window
still stays open), or `--fast` to drop the pacing.

```bash
pytest -q          # 10 environment contract tests
```

On a headless machine (server, CI, WSL without a display) set `MUJOCO_GL=disable`
before any command that does not need a window — MuJoCo otherwise tries to open
an OpenGL context at import time and fails.

## Train

```bash
# Stable-Baselines3, SAC  (~200k steps, 20-40 min on CPU)
python scripts/train_sb3.py

# TorchRL, PPO
python scripts/train_torchrl.py

# a fast smoke run of either
python scripts/train_sb3.py --timesteps 20000
python scripts/train_torchrl.py --total-frames 20000
```

Both read their defaults from `configs/`, and every key is overridable on the
command line (`--lr`, `--seed`, `--device`, …).

Watch SB3 learn in TensorBoard:

```bash
tensorboard --logdir runs/sb3/tb
```

## Evaluate and watch

```bash
# live viewer
python scripts/evaluate.py --algo sb3     --ckpt runs/sb3/best_model.zip   --render human
python scripts/evaluate.py --algo torchrl --ckpt runs/torchrl/ppo_final.pt --render human

# headless benchmark
python scripts/evaluate.py --algo sb3 --ckpt runs/sb3/best_model.zip --episodes 50

# record an mp4
python scripts/evaluate.py --algo sb3 --ckpt runs/sb3/best_model.zip --video runs/sb3/demo.mp4

# untrained baseline for comparison
python scripts/evaluate.py --algo random --episodes 20
```

---

## Layout

```
mujoco-arm-rl/
├── assets/arm3.xml                 MJCF model: arm, target mocap body, scene
├── configs/
│   ├── sb3_sac.yaml                defaults for the SB3 trainer
│   └── torchrl_ppo.yaml            defaults for the TorchRL trainer
├── src/mujoco_arm_rl/
│   ├── envs/reach3.py              the environment — physics, obs, reward, render
│   ├── envs/__init__.py            registers "MujocoArm3Reach-v0" with Gymnasium
│   └── utils.py                    seeding, config loading, run directories
├── scripts/
│   ├── view_random.py              viewer + random torques (start here)
│   ├── train_sb3.py                SAC / PPO / TD3 via Stable-Baselines3
│   ├── train_torchrl.py            clipped PPO written out in full
│   └── evaluate.py                 rollouts, metrics, viewer, mp4 export
├── tests/test_env.py               spaces, seeding, truncation, reward monotonicity
├── .github/workflows/ci.yml        lint + tests on every push
├── requirements.txt
└── pyproject.toml
```

The environment is written directly against the `mujoco` bindings instead of
subclassing `gymnasium.envs.mujoco.MujocoEnv`, so the whole loop — stepping
physics, building the observation, shaping the reward — lives in one readable
file and does not break when Gymnasium reshuffles its internals.

## Tuning notes

- **`gamma = 0.98`** rather than 0.99: episodes are only 200 steps, so a shorter
  effective horizon learns faster here.
- **SAC over PPO** as the default: this is a low-dimensional continuous-control
  task with cheap simulation, exactly where off-policy sample efficiency wins.
  PPO needs roughly 5× the environment steps for the same policy.
- **Reward scale** matters more than reward *shape*. If you raise
  `success_bonus`, drop `ctrl_cost_weight` to match or the policy will learn to
  go limp and let gravity park the arm.
- **Harder variants** to try: shrink `success_threshold` to 2 cm, widen
  `target_radius_range` toward the edge of the workspace, or add an obstacle geom
  to `arm3.xml` and a collision penalty to the reward.

### A note on TorchRL versions

TorchRL renames public API between releases — `SyncDataCollector` became
`Collector`, and `entropy_coef` / `critic_coef` became `entropy_coeff` /
`critic_coeff`. `train_torchrl.py` resolves both at import time (see the
`compat_kwargs` shim at the top of the file) rather than pinning a version, so
it runs on 0.4 through 0.14 without edits.

`evaluate.py` reloads the TorchRL policy with plain PyTorch — no TorchRL import
— by replaying the saved `ObservationNorm` statistics by hand. The checkpoint
records which normalisation convention was used (`obs * scale + loc` by default,
`(obs - loc) / scale` when `standard_normal` is set); getting that backwards
silently produces a policy that scores *worse* than random while training logs
look fine.

## License

MIT — see [LICENSE](LICENSE).
