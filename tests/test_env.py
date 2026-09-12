"""Contract tests for the Reach3 environment. Run with: pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from mujoco_arm_rl.envs import ENV_ID, Reach3Env


@pytest.fixture()
def env():
    e = Reach3Env()
    yield e
    e.close()


def test_spaces(env):
    assert env.observation_space.shape == (18,)
    assert env.action_space.shape == (3,)
    assert np.allclose(env.action_space.low, -1.0)
    assert np.allclose(env.action_space.high, 1.0)


def test_reset_returns_valid_obs(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert np.isfinite(obs).all()
    assert "distance" in info


def test_step_shapes_and_finiteness(env):
    env.reset(seed=0)
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert not terminated, "physics should not diverge under bounded torques"
    assert set(info) >= {"distance", "is_success"}


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
    obs_a, _ = a.reset(seed=42)
    obs_b, _ = b.reset(seed=42)
    assert np.allclose(obs_a, obs_b)
    for _ in range(10):
        act = a.action_space.sample() * 0 + 0.3
        obs_a, r_a, *_ = a.step(act)
        obs_b, r_b, *_ = b.step(act)
    assert np.allclose(obs_a, obs_b)
    assert r_a == pytest.approx(r_b)
    a.close()
    b.close()


def test_targets_are_reachable(env):
    """Every sampled target must sit inside the arm's workspace."""
    max_reach = 0.25 + 0.25 + 0.035  # link2 + link3 + end-effector radius
    shoulder = env._shoulder_pos
    for seed in range(30):
        env.reset(seed=seed)
        assert np.linalg.norm(env.target_pos - shoulder) <= max_reach
        assert env.target_pos[2] >= env.target_min_height


def test_reward_improves_as_the_tip_approaches(env):
    """Reward must be monotone in distance, all else equal."""
    env.reset(seed=3)
    _, r_far, *_ = env.step(np.zeros(3))
    far = env._get_obs()
    d_far = np.linalg.norm(env.target_pos - env.ee_pos)
    # Move the target onto the tip; the same zero action must now score better.
    env.data.mocap_pos[0] = env.ee_pos
    _, r_near, *_ = env.step(np.zeros(3))
    assert r_near > r_far
    assert d_far > 0 and far.shape == (18,)


def test_fixed_target_never_moves():
    e = Reach3Env(fixed_target=(0.30, 0.0, 0.35))
    for seed in range(5):
        e.reset(seed=seed)
        assert np.allclose(e.target_pos, [0.30, 0.0, 0.35])
    e.close()


def test_unreachable_fixed_target_is_rejected():
    with pytest.raises(ValueError, match="unreachable"):
        Reach3Env(fixed_target=(0.9, 0.0, 0.3))
    with pytest.raises(ValueError, match="3 numbers"):
        Reach3Env(fixed_target=(0.3, 0.0))


def test_gymnasium_registration():
    gym = pytest.importorskip("gymnasium")
    import mujoco_arm_rl.envs  # noqa: F401

    e = gym.make(ENV_ID)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (18,)
    e.close()
