"""Reach-a-random-target task for a 3-DOF MuJoCo manipulator.

The environment is written directly against the ``mujoco`` bindings (rather than
``gymnasium.envs.mujoco.MujocoEnv``) so that every part of the loop -- stepping
physics, building the observation, shaping the reward -- is visible in one file
and does not depend on Gymnasium internals that change between releases.

Observation (18,)
    [0:3]   cos(q)          joint angles, cosine part
    [3:6]   sin(q)          joint angles, sine part
    [6:9]   qvel / 10       joint velocities, roughly unit-scaled
    [9:12]  ee_pos          end-effector position, world frame
    [12:15] target_pos      target position, world frame
    [15:18] target - ee     error vector (the thing the policy must zero out)

Action (3,)
    Normalised joint torques in [-1, 1], scaled by the ``gear`` values in the
    MJCF model.

Reward
    -distance  +  success_bonus  -  control_cost  -  velocity_cost
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np

try:  # Gymnasium is the only hard RL dependency of the env itself.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required: pip install 'gymnasium>=0.29'"
    ) from exc


ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "assets",
)
DEFAULT_MODEL_PATH = os.path.join(ASSETS_DIR, "arm3.xml")


class Reach3Env(gym.Env):
    """3-joint arm that must drive its end-effector onto a random target."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        frame_skip: int = 5,
        max_episode_steps: int = 200,
        success_threshold: float = 0.05,
        success_bonus: float = 1.0,
        ctrl_cost_weight: float = 0.01,
        vel_cost_weight: float = 0.001,
        target_radius_range: tuple[float, float] = (0.18, 0.45),
        target_min_height: float = 0.10,
        fixed_target: Sequence[float] | None = None,
        render_mode: str | None = None,
        width: int = 640,
        height: int = 480,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MJCF model not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.success_threshold = float(success_threshold)
        self.success_bonus = float(success_bonus)
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        self.vel_cost_weight = float(vel_cost_weight)
        self.target_radius_range = target_radius_range
        self.target_min_height = float(target_min_height)
        self.fixed_target = (
            None if fixed_target is None else np.asarray(fixed_target, dtype=np.float64)
        )

        self.render_mode = render_mode
        self.width, self.height = int(width), int(height)
        self._viewer = None
        self._renderer = None

        self._tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self._target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site"
        )
        self._shoulder_pos = np.array([0.0, 0.0, 0.10])  # base of the kinematic chain

        self.nq, self.nv, self.nu = self.model.nq, self.model.nv, self.model.nu
        assert self.nu == 3, f"expected a 3-actuator model, got {self.nu}"

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(18,), dtype=np.float32
        )

        self.metadata["render_fps"] = int(round(1.0 / self.dt))

        if self.fixed_target is not None:
            if self.fixed_target.shape != (3,):
                raise ValueError(f"fixed_target must be 3 numbers, got {self.fixed_target!r}")
            ok, why = self.check_target_reachable(self.fixed_target)
            if not ok:
                pretty = tuple(round(float(v), 3) for v in self.fixed_target)
                shoulder = tuple(round(float(v), 3) for v in self._shoulder_pos)
                raise ValueError(
                    f"fixed_target {pretty} is unreachable: {why}. "
                    f"The shoulder sits at {shoulder}."
                )

        self._elapsed_steps = 0
        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------ utils

    @property
    def dt(self) -> float:
        """Duration of one agent step (seconds)."""
        return self.model.opt.timestep * self.frame_skip

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tip_id].copy()

    @property
    def target_pos(self) -> np.ndarray:
        return self.data.mocap_pos[0].copy()

    # Kinematic reach: link2 + link3 + the end-effector sphere radius.
    MAX_REACH = 0.25 + 0.25 + 0.035

    def check_target_reachable(self, point: np.ndarray) -> tuple[bool, str]:
        """Is this point inside the arm's workspace? Returns (ok, explanation)."""
        point = np.asarray(point, dtype=np.float64)
        radius = float(np.linalg.norm(point - self._shoulder_pos))
        if radius > self.MAX_REACH:
            return False, (f"{radius:.3f} m from the shoulder, but the arm only reaches "
                           f"{self.MAX_REACH:.3f} m")
        if point[2] < 0.0:
            return False, f"z = {point[2]:.3f} m is below the floor"
        return True, f"{radius:.3f} m from the shoulder (max {self.MAX_REACH:.3f} m)"

    def _sample_target(self) -> np.ndarray:
        """The fixed target if one was given, else a random reachable point."""
        if self.fixed_target is not None:
            return self.fixed_target.copy()

        lo, hi = self.target_radius_range
        for _ in range(100):
            direction = self.np_random.normal(size=3)
            direction /= np.linalg.norm(direction) + 1e-9
            radius = self.np_random.uniform(lo, hi)
            point = self._shoulder_pos + direction * radius
            if point[2] >= self.target_min_height:
                return point
        return self._shoulder_pos + np.array([lo, 0.0, 0.0])

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[: self.nq]
        qvel = self.data.qvel[: self.nv]
        ee, target = self.ee_pos, self.target_pos
        return np.concatenate(
            [
                np.cos(qpos),
                np.sin(qpos),
                qvel / 10.0,
                ee,
                target,
                target - ee,
            ]
        ).astype(np.float32)

    # ------------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        # Small random start pose so the policy cannot memorise a single trajectory.
        self.data.qpos[: self.nq] = self.np_random.uniform(-0.3, 0.3, size=self.nq)
        self.data.qvel[: self.nv] = self.np_random.uniform(-0.05, 0.05, size=self.nv)
        self.data.mocap_pos[0] = self._sample_target()
        mujoco.mj_forward(self.model, self.data)

        self._elapsed_steps = 0
        obs = self._get_obs()
        info = {"distance": float(np.linalg.norm(self.target_pos - self.ee_pos))}

        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[: self.nu] = action
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_rnePostConstraint(self.model, self.data)  # keeps site data fresh

        distance = float(np.linalg.norm(self.target_pos - self.ee_pos))
        success = distance < self.success_threshold

        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(action)))
        vel_cost = self.vel_cost_weight * float(np.sum(np.square(self.data.qvel)))
        reward = -distance + (self.success_bonus if success else 0.0) - ctrl_cost - vel_cost

        self._elapsed_steps += 1
        terminated = not np.isfinite(self.data.qpos).all()  # physics blew up
        truncated = self._elapsed_steps >= self.max_episode_steps

        info = {
            "distance": distance,
            "is_success": bool(success),
            "ctrl_cost": ctrl_cost,
            "vel_cost": vel_cost,
        }

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ render

    def render(self):
        if self.render_mode == "human":
            import mujoco.viewer

            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data, show_left_ui=False, show_right_ui=False
                )
            if self._viewer.is_running():
                self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model, height=self.height, width=self.width
                )
            self._renderer.update_scene(self.data, camera=-1)
            return self._renderer.render()

        return None

    @property
    def viewer_is_running(self) -> bool:
        """True while a human-mode viewer window is open (False if there is none)."""
        return self._viewer is not None and self._viewer.is_running()

    def wait_for_viewer(self, fps: int = 50) -> None:
        """Hold the viewer window open until the user closes it.

        Without this the process exits as soon as the rollout ends -- which, with
        no realtime pacing, is well under a second -- and the window vanishes
        before you have seen anything.
        """
        if self._viewer is None:
            return
        print("Viewer is open -- close the window (or press Ctrl+C) to exit.")
        try:
            while self._viewer.is_running():
                self._viewer.sync()
                time.sleep(1.0 / max(fps, 1))
        except KeyboardInterrupt:
            pass

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
