from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from ..actions import Action
from ..runner import _unwrap_ram, create_mario_env
from ..state import MarioSnapshot, MarioStateParser
from .actions import action_from_index, expand_macro_frames, index_from_action


@dataclass(frozen=True)
class AdapterStepResult:
    snapshot: MarioSnapshot
    native_reward: float
    terminated: bool
    truncated: bool
    died: bool
    cleared: bool
    info: dict[str, Any]
    frames_executed: int


class EmulatorBackend(Protocol):
    @property
    def last_snapshot(self) -> MarioSnapshot | None: ...

    @property
    def last_frame(self) -> Any: ...

    def reset(self, *, seed: int | None = None) -> tuple[MarioSnapshot, dict[str, Any]]: ...

    def step(self, action_index: int) -> AdapterStepResult: ...

    def render(self) -> Any: ...

    def close(self) -> None: ...


class MarioEmulatorAdapter:
    """Adapter bridging gym-super-mario-bros, RAM parsing, and RL macro actions."""

    def __init__(
        self,
        env_id: str = "SuperMarioBros-1-1-v0",
        *,
        frames_per_decision: int = 4,
        render_mode: str = "rgb_array",
        random_noop_max: int = 0,
        goal: str = "Reach the flag in World 1-1 without dying.",
        capture_step_frames: bool = False,
    ) -> None:
        if frames_per_decision < 1:
            raise ValueError("frames_per_decision must be at least 1")
        if random_noop_max < 0:
            raise ValueError("random_noop_max must be non-negative")

        self.env_id = env_id
        self.frames_per_decision = frames_per_decision
        self.render_mode = render_mode
        self.random_noop_max = random_noop_max
        self.capture_step_frames = capture_step_frames
        self.parser = MarioStateParser(
            goal=goal,
            decision_horizon_frames=frames_per_decision,
        )
        self._env: Any | None = None
        self._rng = np.random.default_rng()
        self._last_frame: Any = None
        self._last_step_frames: list[Any] = []
        self._last_snapshot: MarioSnapshot | None = None
        self._previous_action: Action | None = None
        self._initial_lives: int = 0
        self._clear_anim_frames: int = 0
        self._clear_anim_origin: tuple[int, int] | None = None

    @property
    def last_snapshot(self) -> MarioSnapshot | None:
        return self._last_snapshot

    @property
    def last_frame(self) -> Any:
        return self._last_frame

    @property
    def last_step_frames(self) -> tuple[Any, ...]:
        return tuple(self._last_step_frames)

    @property
    def area_type(self) -> int | None:
        if self._env is None:
            return None
        ram = _unwrap_ram(self._env)
        if ram is None or len(ram) <= 0x074E:
            return None
        return int(ram[0x074E])

    def _ensure_env(self) -> Any:
        if self._env is None:
            self._env = create_mario_env(self.env_id, render_mode=self.render_mode)
        return self._env

    @staticmethod
    def _extract_lives(info: dict[str, Any]) -> int:
        val = info.get("life", info.get("lives", 0))
        return int(val) if val is not None else 0

    def reset(self, *, seed: int | None = None) -> tuple[MarioSnapshot, dict[str, Any]]:
        env = self._ensure_env()
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        frame, info = env.reset(seed=seed)
        self.parser.reset()
        self._last_frame = frame
        self._previous_action = None
        self._initial_lives = self._extract_lives(info)
        self._clear_anim_frames = 0
        self._clear_anim_origin = None

        if self.random_noop_max > 0:
            noop_count = int(self._rng.integers(0, self.random_noop_max + 1))
            noop_idx = index_from_action(Action.NOOP)
            for _ in range(noop_count):
                frame, _, terminated, truncated, info = env.step(noop_idx)
                self._last_frame = frame
                if terminated or truncated:
                    frame, info = env.reset(seed=seed)
                    self._last_frame = frame
                    self._initial_lives = self._extract_lives(info)
                    break

        snapshot = self.parser.parse(
            info,
            _unwrap_ram(env),
            previous_action=None,
            previous_reward=0.0,
            previous_response_delay_frames=0,
        )
        self._last_snapshot = snapshot
        if self.capture_step_frames and self._last_frame is not None:
            self._last_step_frames = [np.array(self._last_frame, copy=True)]
        else:
            self._last_step_frames = []
        return snapshot, dict(info)

    def step(self, action_index: int) -> AdapterStepResult:
        env = self._ensure_env()
        action = action_from_index(action_index)
        grounded = self._last_snapshot.grounded if self._last_snapshot is not None else True

        frame_actions = expand_macro_frames(
            action,
            frames=self.frames_per_decision,
            previous_action=self._previous_action,
            grounded=grounded,
        )

        total_reward = 0.0
        terminated = False
        truncated = False
        died = False
        cleared = False
        info: dict[str, Any] = {}
        frames_executed = 0
        step_frames: list[Any] = []

        for frame_action in frame_actions:
            frame, reward, step_terminated, step_truncated, step_info = env.step(frame_action)
            self._last_frame = frame
            if self.capture_step_frames and frame is not None:
                step_frames.append(np.array(frame, copy=True))
            info = dict(step_info)
            total_reward += float(reward)
            frames_executed += 1

            lives = self._extract_lives(info)
            step_died = bool(
                info.get("death", False)
                or info.get("is_dead", False)
                or (self._initial_lives > 0 and lives < self._initial_lives)
            )
            step_cleared = bool(info.get("clear", False) or info.get("flag_get", False))

            if step_died:
                died = True
            if step_cleared:
                cleared = True
            if step_terminated:
                terminated = True
            if step_truncated:
                truncated = True

            if terminated or truncated or died or cleared:
                break

        snapshot = self.parser.parse(
            info,
            _unwrap_ram(env),
            previous_action=action.value,
            previous_reward=total_reward,
            previous_response_delay_frames=0,
        )
        if snapshot.dead:
            died = True
        if snapshot.clear:
            cleared = True

        self._last_snapshot = snapshot
        self._previous_action = action
        self._last_step_frames = step_frames

        return AdapterStepResult(
            snapshot=snapshot,
            native_reward=total_reward,
            terminated=bool(terminated or died or cleared),
            truncated=bool(truncated and not (terminated or died or cleared)),
            died=died,
            cleared=cleared,
            info=info,
            frames_executed=frames_executed,
        )

    def step_clear_animation(self, *, max_frames: int = 900) -> Any | None:
        """Advance one NES frame of the post-flagpole clear animation until stage transition."""
        if (
            self._env is None
            or self._last_snapshot is None
            or not self._last_snapshot.clear
            or self._clear_anim_frames >= max_frames
        ):
            return None

        nes = getattr(self._env, "unwrapped", None)
        if nes is None or not hasattr(nes, "_frame_advance") or not hasattr(nes, "screen"):
            return None

        if self._clear_anim_origin is None:
            self._clear_anim_origin = (
                int(self._last_snapshot.world),
                int(self._last_snapshot.stage),
            )

        nes._frame_advance(0)
        self._clear_anim_frames += 1

        raw_info = nes._get_info() if hasattr(nes, "_get_info") else {}
        current_world = int(raw_info.get("world", self._clear_anim_origin[0]))
        current_stage = int(raw_info.get("stage", self._clear_anim_origin[1]))
        if (current_world, current_stage) != self._clear_anim_origin:
            return None

        frame = np.array(nes.screen, copy=True)
        self._last_frame = frame
        new_x = int(raw_info.get("x_pos", self._last_snapshot.x))
        new_y = int(raw_info.get("y_pos", self._last_snapshot.y))
        new_score = int(raw_info.get("score", self._last_snapshot.score))
        new_time = int(raw_info.get("time", self._last_snapshot.time_left))
        self._last_snapshot = replace(
            self._last_snapshot,
            x=new_x,
            y=new_y,
            score=new_score,
            time_left=new_time,
            progress=max(self._last_snapshot.progress, new_x),
            best_progress=max(self._last_snapshot.best_progress, new_x),
            clear=True,
        )
        return frame

    def render(self) -> Any:
        if self.render_mode == "rgb_array":
            return self._last_frame
        if self._env is not None:
            return self._env.render()
        return None

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
