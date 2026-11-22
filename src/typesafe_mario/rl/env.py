from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import NUM_ACTIONS, action_from_index
from .adapter import EmulatorBackend, MarioEmulatorAdapter
from .observations import OBSERVATION_DIM, encode_observation
from .rewards import MarioRewardCalculator, RewardConfig


class MarioRLEnv(gym.Env[np.ndarray, int]):
    """Gymnasium environment for RAM-based RL training on Super Mario Bros."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        env_id: str = "SuperMarioBros-1-1-v0",
        *,
        frames_per_decision: int = 4,
        max_steps: int = 1500,
        max_stall_steps: int = 60,
        random_noop_max: int = 0,
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
        adapter: EmulatorBackend | None = None,
        capture_step_frames: bool = False,
    ) -> None:
        super().__init__()
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if max_stall_steps < 1:
            raise ValueError("max_stall_steps must be at least 1")

        self.env_id = env_id
        self.frames_per_decision = frames_per_decision
        self.max_steps = max_steps
        self.max_stall_steps = max_stall_steps
        self.random_noop_max = random_noop_max
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

        adapter_render_mode = render_mode if render_mode in {"rgb_array", "human"} else "rgb_array"
        self._adapter: EmulatorBackend = adapter or MarioEmulatorAdapter(
            env_id=env_id,
            frames_per_decision=frames_per_decision,
            render_mode=adapter_render_mode,
            random_noop_max=random_noop_max,
            capture_step_frames=capture_step_frames,
        )
        self._reward_calc = MarioRewardCalculator(reward_config)
        self._step_count = 0
        self._episode_reward = 0.0
        self._best_x = 0
        self._last_snapshot = None

    @property
    def last_snapshot(self) -> Any:
        return self._last_snapshot

    @property
    def last_frame(self) -> Any:
        return self._adapter.last_frame

    @property
    def last_step_frames(self) -> tuple[Any, ...]:
        frames = getattr(self._adapter, "last_step_frames", None)
        if frames:
            return tuple(frames)
        frame = self._adapter.last_frame
        return (frame,) if frame is not None else ()

    @property
    def area_type(self) -> int | None:
        return getattr(self._adapter, "area_type", None)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        snapshot, raw_info = self._adapter.reset(seed=seed)
        self._reward_calc.reset(snapshot)
        self._step_count = 0
        self._episode_reward = 0.0
        self._best_x = snapshot.x
        self._last_snapshot = snapshot

        obs = encode_observation(snapshot)
        info = {
            "env_id": self.env_id,
            "x": snapshot.x,
            "best_x": self._best_x,
            "clear": snapshot.clear,
            "dead": snapshot.dead,
            "score": snapshot.score,
            "coins": snapshot.coins,
            "time_left": snapshot.time_left,
            "steps": self._step_count,
            "episode_reward": self._episode_reward,
            "raw_info": raw_info,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._last_snapshot is None:
            raise RuntimeError("Environment must be reset before calling step().")

        action_idx = int(action)
        chosen_action = action_from_index(action_idx)
        previous_snapshot = self._last_snapshot

        result = self._adapter.step(action_idx)
        snapshot = result.snapshot
        breakdown = self._reward_calc.compute(
            previous_snapshot,
            snapshot,
            died=result.died,
            cleared=result.cleared,
        )

        self._step_count += 1
        self._episode_reward += breakdown.total
        self._best_x = max(self._best_x, snapshot.x)
        self._last_snapshot = snapshot

        terminated = bool(result.terminated)
        stalled_out = (not terminated) and (snapshot.stalled_steps >= self.max_stall_steps)
        truncated = bool(
            result.truncated
            or (not terminated and self._step_count >= self.max_steps)
            or stalled_out
        )

        obs = encode_observation(snapshot)
        info = {
            "env_id": self.env_id,
            "action": chosen_action.value,
            "x": snapshot.x,
            "best_x": self._best_x,
            "clear": result.cleared,
            "dead": result.died,
            "stalled_out": stalled_out,
            "score": snapshot.score,
            "coins": snapshot.coins,
            "time_left": snapshot.time_left,
            "steps": self._step_count,
            "frames_executed": result.frames_executed,
            "native_reward": result.native_reward,
            "episode_reward": self._episode_reward,
            "reward_breakdown": breakdown.to_dict(),
        }
        return obs, float(breakdown.total), terminated, truncated, info

    def step_clear_animation(self, *, max_frames: int = 900) -> Any | None:
        step_anim = getattr(self._adapter, "step_clear_animation", None)
        if not callable(step_anim):
            return None
        frame = step_anim(max_frames=max_frames)
        if frame is not None and self._adapter.last_snapshot is not None:
            self._last_snapshot = self._adapter.last_snapshot
            self._best_x = max(self._best_x, self._last_snapshot.x)
        return frame

    def render(self) -> Any:
        return self._adapter.render()

    def close(self) -> None:
        self._adapter.close()


def world_stage_env_ids(world: int, *, version: int = 0) -> tuple[str, ...]:
    """Return all 4 stage gym env IDs for a given Super Mario Bros. world (1..8)."""
    if not 1 <= world <= 8:
        raise ValueError(f"world must be in [1, 8], got {world}")
    return tuple(f"SuperMarioBros-{world}-{stage}-v{version}" for stage in range(1, 5))


class MultiStageMarioEnv(gym.Env[np.ndarray, int]):
    """Gymnasium environment that cycles or samples across multiple Mario stages."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        env_ids: tuple[str, ...] | list[str],
        *,
        frames_per_decision: int = 4,
        max_steps: int = 1500,
        max_stall_steps: int = 60,
        random_noop_max: int = 0,
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
        sampling: str = "round_robin",
        stage_envs: dict[str, MarioRLEnv] | None = None,
    ) -> None:
        super().__init__()
        if not env_ids:
            raise ValueError("env_ids must contain at least one environment ID")
        if sampling not in {"round_robin", "random"}:
            raise ValueError("sampling must be 'round_robin' or 'random'")

        self.env_ids = tuple(env_ids)
        self.frames_per_decision = frames_per_decision
        self.max_steps = max_steps
        self.max_stall_steps = max_stall_steps
        self.random_noop_max = random_noop_max
        self.render_mode = render_mode
        self.reward_config = reward_config
        self.sampling = sampling

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

        self._stage_envs: dict[str, MarioRLEnv] = dict(stage_envs) if stage_envs else {}
        self._active_env_id: str | None = None
        self._active_env: MarioRLEnv | None = None
        self._cycle_index = 0
        self._rng = np.random.default_rng()

    @property
    def active_env_id(self) -> str | None:
        return self._active_env_id

    @property
    def last_snapshot(self) -> Any:
        return self._active_env.last_snapshot if self._active_env is not None else None

    @property
    def last_frame(self) -> Any:
        return self._active_env.last_frame if self._active_env is not None else None

    def _get_or_create_stage_env(self, env_id: str) -> MarioRLEnv:
        if env_id not in self._stage_envs:
            self._stage_envs[env_id] = MarioRLEnv(
                env_id=env_id,
                frames_per_decision=self.frames_per_decision,
                max_steps=self.max_steps,
                max_stall_steps=self.max_stall_steps,
                random_noop_max=self.random_noop_max,
                render_mode=self.render_mode,
                reward_config=self.reward_config,
            )
        return self._stage_envs[env_id]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options and "env_id" in options:
            chosen_id = str(options["env_id"])
            if chosen_id not in self.env_ids:
                raise ValueError(f"Requested env_id {chosen_id!r} not in {self.env_ids}")
        elif self.sampling == "random":
            chosen_id = self.env_ids[int(self._rng.integers(0, len(self.env_ids)))]
        else:
            chosen_id = self.env_ids[self._cycle_index % len(self.env_ids)]
            self._cycle_index += 1

        self._active_env_id = chosen_id
        self._active_env = self._get_or_create_stage_env(chosen_id)
        return self._active_env.reset(seed=seed)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._active_env is None:
            raise RuntimeError("Environment must be reset before calling step().")
        return self._active_env.step(action)

    def render(self) -> Any:
        if self._active_env is not None:
            return self._active_env.render()
        return None

    def close(self) -> None:
        for env in self._stage_envs.values():
            env.close()
        self._stage_envs.clear()
        self._active_env = None
