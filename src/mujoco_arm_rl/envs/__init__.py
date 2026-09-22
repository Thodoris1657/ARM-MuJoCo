"""Environment package. Importing it registers the Gymnasium environment IDs.

MujocoArm3Reach-v1   position-servo control at 25 Hz, reachable goals, shaped reward
MujocoArm3Reach-v0   the original torque-control task, kept for old checkpoints/ablations

The env truncates its own episodes, so no TimeLimit wrapper is registered: that
keeps ``episode_seconds`` / ``control_hz`` overrides from fighting a fixed limit.
"""

from gymnasium.envs.registration import register, registry

from mujoco_arm_rl.envs.reach3 import (
    DEFAULT_MODEL_PATH,
    LEGACY_KWARGS,
    Reach3Env,
    analytic_ik,
)

ENV_ID = "MujocoArm3Reach-v1"
LEGACY_ENV_ID = "MujocoArm3Reach-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="mujoco_arm_rl.envs.reach3:Reach3Env")
if LEGACY_ENV_ID not in registry:
    register(id=LEGACY_ENV_ID, entry_point="mujoco_arm_rl.envs.reach3:Reach3Env",
             kwargs=dict(LEGACY_KWARGS))

__all__ = ["Reach3Env", "ENV_ID", "LEGACY_ENV_ID", "LEGACY_KWARGS",
           "DEFAULT_MODEL_PATH", "analytic_ik"]
