"""Load any policy this repo can produce behind one interface: ``act(obs, env)``.

Every trainer writes ``env_config.json`` next to its checkpoints recording the env
ID and constructor kwargs it trained on. Evaluation reads it back, so a policy is
always rolled out on the task it learned -- no remembering to repeat ``--target``
or which control mode was used. Checkpoints from before that file existed are
recognised by their observation size (18 = the legacy v0 task).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_arm_rl.envs import ENV_ID, LEGACY_ENV_ID, LEGACY_KWARGS

Policy = Callable[[np.ndarray, Any], np.ndarray]
ENV_CONFIG_NAME = "env_config.json"


# ----------------------------------------------------------------- env config


def save_env_config(out_dir: str | Path, env_id: str, env_kwargs: dict[str, Any]) -> Path:
    path = Path(out_dir) / ENV_CONFIG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: (list(v) if isinstance(v, (tuple, np.ndarray)) else v) for k, v in env_kwargs.items()}
    path.write_text(json.dumps({"env_id": env_id, "env_kwargs": clean}, indent=2))
    return path


def resolve_env_kwargs(env_id: str, env_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Full constructor kwargs for an env ID (the v0 ID carries its legacy defaults)."""
    base = dict(LEGACY_KWARGS) if env_id == LEGACY_ENV_ID else {}
    base.update(env_kwargs)
    return base


def find_env_config(ckpt: str | Path, obs_dim: int | None = None) -> tuple[str, dict[str, Any]]:
    """env_config.json beside the checkpoint (or one level up), else infer from obs size."""
    ckpt = Path(ckpt)
    for folder in (ckpt.parent, ckpt.parent.parent):
        cfg = folder / ENV_CONFIG_NAME
        if cfg.exists():
            blob = json.loads(cfg.read_text())
            return blob["env_id"], blob.get("env_kwargs", {})
    if obs_dim == 18:
        return LEGACY_ENV_ID, {}
    return ENV_ID, {}


# ------------------------------------------------------------------- policies


def random_policy() -> Policy:
    return lambda obs, env: env.action_space.sample()


def oracle_policy() -> Policy:
    """Analytic IK + rate-limited servo targets: a scripted reference controller.

    Uses the same action interface as the learned policies (delta joint targets,
    same speed limit), so it is an honest point of comparison rather than a
    teleport. It cannot compensate servo sag, so it settles a few mm short.
    """
    cache: dict[str, Any] = {"goal_for": None, "goal": None}

    def act(obs, env):
        if env.control_mode != "position":
            raise ValueError("the IK oracle needs a position-control env")
        key = tuple(np.round(env.target_pos, 9))
        if cache["goal_for"] != key:
            goal = env.solve_ik(env.target_pos, near=env.data.qpos[: env.nq])
            cache.update(goal_for=key, goal=goal)
        goal = cache["goal"]
        if goal is None:
            return np.zeros(env.nu)
        return np.clip((goal - env.q_target) / env.max_step, -1.0, 1.0)

    return act


def load_sb3(ckpt: str | Path) -> tuple[Policy, int]:
    from stable_baselines3 import PPO, SAC, TD3

    classes = [SAC, PPO, TD3]
    try:
        from sb3_contrib import TQC
        classes.insert(0, TQC)
    except ImportError:
        pass
    last_err = None
    for cls in classes:
        try:
            model = cls.load(str(ckpt), device="cpu")

            def act(obs, env, _model=model):
                return _model.predict(obs, deterministic=True)[0]

            return act, int(model.observation_space.shape[0])
        except Exception as err:  # noqa: BLE001 -- try the next algorithm class
            last_err = err
    raise RuntimeError(f"could not load {ckpt} with any SB3 algorithm: {last_err}")


def load_torchrl(ckpt: str | Path) -> tuple[Policy, int]:
    """Rebuild the TorchRL PPO actor in plain PyTorch (no TorchRL import needed)."""
    import torch
    from torch import nn

    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    hidden, n_obs, n_act = blob["hidden"], blob["n_obs"], blob["n_act"]
    act_fn = nn.ReLU if blob.get("activation", "tanh") == "relu" else nn.Tanh
    net = nn.Sequential(
        nn.Linear(n_obs, hidden), act_fn(),
        nn.Linear(hidden, hidden), act_fn(),
        nn.Linear(hidden, 2 * n_act),
    )
    net.load_state_dict(blob["policy_state_dict"])  # NormalParamExtractor has no params
    net.eval()

    loc = blob["obs_loc"].float().numpy()
    scale = blob["obs_scale"].float().numpy()
    standard_normal = bool(blob.get("obs_standard_normal", False))

    def act(obs, env):
        x = np.asarray(obs, dtype=np.float32)
        # replay TorchRL's ObservationNorm exactly (see README: getting this
        # backwards silently yields a policy worse than random)
        x = (x - loc) / scale if standard_normal else x * scale + loc
        with torch.no_grad():
            out = net(torch.as_tensor(x, dtype=torch.float32))
        return np.tanh(out[..., :n_act].numpy())

    return act, int(n_obs)


def load_policy(spec: str) -> tuple[Policy, str, dict[str, Any], str]:
    """``spec`` is "random", "oracle", or a checkpoint path (.zip = SB3, .pt = TorchRL).

    Returns (act, env_id, env_kwargs, description).
    """
    if spec == "random":
        return random_policy(), ENV_ID, {}, "random actions"
    if spec == "oracle":
        return oracle_policy(), ENV_ID, {}, "analytic IK oracle"
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if path.suffix == ".pt":
        act, obs_dim = load_torchrl(path)
        kind = "TorchRL PPO"
    else:
        act, obs_dim = load_sb3(path)
        kind = "SB3"
    env_id, env_kwargs = find_env_config(path, obs_dim)
    return act, env_id, env_kwargs, f"{kind} {path}"
