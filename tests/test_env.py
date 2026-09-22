"""Contract tests for the Reach3 environment. Run with: pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np
import pytest

from mujoco_arm_rl.envs import ENV_ID, LEGACY_ENV_ID, LEGACY_KWARGS, Reach3Env, analytic_ik
from mujoco_arm_rl.policies import oracle_policy


@pytest.fixture()
def env():
    e = Reach3Env()
    yield e
    e.close()


# ----------------------------------------------------------------- basic API

def test_spaces(env):
    assert env.observation_space.shape == (24,)
    assert env.action_space.shape == (3,)
    assert np.allclose(env.action_space.low, -1.0) and np.allclose(env.action_space.high, 1.0)


def test_timing_defaults(env):
    assert env.dt == pytest.approx(0.04)            # 25 Hz decisions
    assert env.max_episode_steps == 50              # 2 s episodes


def test_legacy_mode_matches_v0():
    e = Reach3Env(**LEGACY_KWARGS)
    assert e.observation_space.shape == (18,)
    assert e.dt == pytest.approx(0.01) and e.max_episode_steps == 200
    assert e.control_mode == "torque"
    e.close()


def test_step_shapes_and_finiteness(env):
    env.reset(seed=0)
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert not terminated
    assert {"distance", "is_success", "is_success_2cm", "time_to_reach"} <= set(info)


def test_episode_truncates_at_limit(env):
    env.reset(seed=1)
    steps, done = 0, False
    while not done and steps < 1000:
        _, _, terminated, truncated, _ = env.step(np.zeros(3))
        steps += 1
        done = terminated or truncated
    assert steps == env.max_episode_steps


def test_seeding_is_reproducible():
    a, b = Reach3Env(), Reach3Env()
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    assert np.allclose(oa, ob)
    for _ in range(10):
        oa, ra, *_ = a.step(np.full(3, 0.3))
        ob, rb, *_ = b.step(np.full(3, 0.3))
    assert np.allclose(oa, ob) and ra == pytest.approx(rb)
    a.close()
    b.close()


def test_reset_options_pin_start_and_target(env):
    obs, _ = env.reset(seed=0, options={"qpos": [0.1, 0.2, -0.3], "target": [0.3, 0.0, 0.35]})
    assert np.allclose(env.data.qpos, [0.1, 0.2, -0.3])
    assert np.allclose(env.target_pos, [0.3, 0.0, 0.35])


# ------------------------------------------------------------ model / physics

def test_no_contacts_at_home_pose(env):
    """Regression: link1 used to sit 4.5 cm inside the base, a permanent contact."""
    env.reset(seed=0, options={"qpos": [0.0, 0.0, 0.0]})
    assert env.data.ncon == 0


def test_analytic_ik_matches_mujoco_fk(env):
    rng = np.random.default_rng(0)
    m, d = env.model, mujoco.MjData(env.model)
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
    for _ in range(200):
        d.qpos[:] = rng.uniform(env.jnt_lo, env.jnt_hi)
        mujoco.mj_kinematics(m, d)
        p = d.site_xpos[tip].copy()
        errs = []
        for q in analytic_ik(p):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            errs.append(np.linalg.norm(d.site_xpos[tip] - p))
        assert min(errs) < 1e-9


def test_sampled_targets_are_all_reachable(env):
    for seed in range(40):
        env.reset(seed=seed)
        assert env.solve_ik(env.target_pos) is not None


def test_servo_tracks_ik_goal(env):
    """The scripted IK oracle must reach within 2 cm in under a second."""
    act = oracle_policy()
    for tgt in ([0.3, 0.0, 0.35], [-0.2, 0.3, 0.12], [0.0, -0.25, 0.6]):
        obs, _ = env.reset(seed=0, options={"qpos": [0, 0, 0], "target": tgt})
        info = {}
        for _ in range(env.max_episode_steps):
            obs, _, _, _, info = env.step(act(obs, env))
        assert info["distance"] < 0.02
        assert info["time_to_reach"] is not None and info["time_to_reach"] < 1.0


def test_reward_improves_as_the_tip_approaches(env):
    env.reset(seed=3)
    _, r_far, *_ = env.step(np.zeros(3))
    env.data.mocap_pos[0] = env.ee_pos
    _, r_near, *_ = env.step(np.zeros(3))
    assert r_near > r_far


def test_torque_mode_still_works():
    e = Reach3Env(control_mode="torque")
    e.reset(seed=0)
    for _ in range(20):
        obs, r, term, trunc, info = e.step(e.action_space.sample())
        assert np.isfinite(obs).all() and not term
    e.close()


# ------------------------------------------------------------------ targets

def test_fixed_target_never_moves():
    e = Reach3Env(fixed_target=(0.30, 0.0, 0.35))
    for seed in range(5):
        e.reset(seed=seed)
        assert np.allclose(e.target_pos, [0.30, 0.0, 0.35])
    e.close()


def test_unreachable_fixed_target_is_rejected():
    with pytest.raises(ValueError, match="unreachable"):
        Reach3Env(fixed_target=(0.9, 0.0, 0.3))       # beyond max reach
    with pytest.raises(ValueError, match="unreachable"):
        Reach3Env(fixed_target=(0.0, 0.0, 0.30))      # inside the elbow's minimum reach
    with pytest.raises(ValueError, match="3 numbers"):
        Reach3Env(fixed_target=(0.3, 0.0))


def test_gymnasium_registration():
    gym = pytest.importorskip("gymnasium")
    import mujoco_arm_rl.envs  # noqa: F401

    for env_id, dim in ((ENV_ID, 24), (LEGACY_ENV_ID, 18)):
        e = gym.make(env_id)
        obs, _ = e.reset(seed=0)
        assert obs.shape == (dim,)
        e.close()
