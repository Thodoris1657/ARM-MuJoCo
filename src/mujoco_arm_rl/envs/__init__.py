"""Environment package. Importing it registers the Gymnasium environment IDs."""

from gymnasium.envs.registration import register, registry

from mujoco_arm_rl.envs.reach3 import DEFAULT_MODEL_PATH, Reach3Env

ENV_ID = "MujocoArm3Reach-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="mujoco_arm_rl.envs.reach3:Reach3Env",
        max_episode_steps=200,
        reward_threshold=100.0,
    )

__all__ = ["Reach3Env", "ENV_ID", "DEFAULT_MODEL_PATH"]
