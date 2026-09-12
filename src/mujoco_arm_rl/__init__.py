"""mujoco_arm_rl -- reinforcement learning on a 3-joint MuJoCo manipulator."""

__version__ = "0.1.0"

from mujoco_arm_rl.envs import ENV_ID, Reach3Env  # noqa: F401  (registers the env)

__all__ = ["ENV_ID", "Reach3Env", "__version__"]
