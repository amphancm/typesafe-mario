from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.env_checker import check_env

from typesafe_mario.rl.actions import NUM_ACTIONS
from typesafe_mario.rl.adapter import AdapterStepResult
from typesafe_mario.rl.env import MarioRLEnv, MultiStageMarioEnv, world_stage_env_ids
from typesafe_mario.rl.observations import OBSERVATION_DIM, encode_observation
from typesafe_mario.state import MarioStateParser


class FakeEmulatorBackend:
    """Deterministic in-memory backend for fast Gymnasium environment unit tests."""

    def __init__(self, *, clear_at_x: int = 160, die_on_noop: bool = False) -> None:
        self.clear_at_x = clear_at_x
        self.die_on_noop = die_on_noop
        self.parser = MarioStateParser(decision_horizon_frames=4)
        self.x = 40
        self.time_left = 400
        self.score = 0
        self._last_snapshot = None
        self._last_frame = np.zeros((240, 256, 3), dtype=np.uint8)

    @property
    def last_snapshot(self) -> Any:
        return self._last_snapshot

    @property
    def last_frame(self) -> Any:
        return self._last_frame

    def _ram(self) -> bytearray:
        ram = bytearray(0x0800)
        ground_row = (79 + 16 - 32) // 16
        for col in range(16):
            ram[0x0500 + ground_row * 16 + col] = 1
        return ram

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        del seed
        self.parser.reset()
        self.x = 40
        self.time_left = 400
        self.score = 0
        info = {
            "world": 1,
            "stage": 1,
            "area": 1,
            "x_pos": self.x,
            "y_pos": 79,
            "y_pixel": 79,
            "progress": self.x,
            "time": self.time_left,
            "life": 2,
            "score": self.score,
            "status": "small",
            "death": False,
            "clear": False,
        }
        self._last_snapshot = self.parser.parse(info, self._ram())
        return self._last_snapshot, info

    def step(self, action_index: int) -> AdapterStepResult:
        died = self.die_on_noop and action_index == 0
        if action_index in (1, 2, 3, 4):
            self.x += 12
            self.score += 10
        elif action_index == 6:
            self.x = max(0, self.x - 6)
        self.time_left = max(0, self.time_left - 1)
        cleared = self.x >= self.clear_at_x and not died

        info = {
            "world": 1,
            "stage": 1,
            "area": 1,
            "x_pos": self.x,
            "y_pos": 79,
            "y_pixel": 79,
            "progress": self.x,
            "time": self.time_left,
            "life": 1 if died else 2,
            "score": self.score,
            "status": "small",
            "death": died,
            "clear": cleared,
        }
        snapshot = self.parser.parse(info, self._ram(), previous_action=str(action_index))
        self._last_snapshot = snapshot
        return AdapterStepResult(
            snapshot=snapshot,
            native_reward=1.0,
            terminated=died or cleared,
            truncated=False,
            died=died,
            cleared=cleared,
            info=info,
            frames_executed=4,
        )

    def render(self) -> Any:
        return self._last_frame

    def close(self) -> None:
        return None


def test_observation_vector_shape_dtype_and_bounds() -> None:
    backend = FakeEmulatorBackend()
    snapshot, _ = backend.reset(seed=1)
    obs = encode_observation(snapshot)

    assert obs.shape == (OBSERVATION_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert float(obs.min()) >= -1.0
    assert float(obs.max()) <= 1.0


def test_mario_rl_env_passes_sb3_check_env_with_mock_backend() -> None:
    env = MarioRLEnv(adapter=FakeEmulatorBackend(), max_steps=100, render_mode="rgb_array")
    try:
        assert env.action_space.n == NUM_ACTIONS
        assert env.observation_space.shape == (OBSERVATION_DIM,)
        check_env(env, warn=True, skip_render_check=False)
    finally:
        env.close()


def test_mario_rl_env_episode_clear_and_death_termination() -> None:
    env = MarioRLEnv(adapter=FakeEmulatorBackend(clear_at_x=64), max_steps=100)
    try:
        obs, info = env.reset(seed=42)
        assert obs.shape == (OBSERVATION_DIM,)
        assert info["x"] == 40

        _, reward1, term1, trunc1, info1 = env.step(3)
        assert not term1 and not trunc1
        assert reward1 > 0.0
        assert info1["x"] == 52

        _, reward2, term2, trunc2, info2 = env.step(3)
        assert term2 and not trunc2
        assert info2["clear"] is True
        assert reward2 > 50.0
    finally:
        env.close()

    death_env = MarioRLEnv(adapter=FakeEmulatorBackend(die_on_noop=True), max_steps=100)
    try:
        death_env.reset(seed=42)
        _, reward, terminated, truncated, info = death_env.step(0)
        assert terminated and not truncated
        assert info["dead"] is True
        assert reward < -10.0
    finally:
        death_env.close()


def test_mario_rl_env_truncates_on_stall() -> None:
    env = MarioRLEnv(adapter=FakeEmulatorBackend(), max_steps=100, max_stall_steps=3)
    try:
        env.reset(seed=7)
        truncated = False
        info: dict[str, Any] = {}
        for _ in range(5):
            _, _, terminated, truncated, info = env.step(0)
            assert not terminated
            if truncated:
                break
        assert truncated is True
        assert info["stalled_out"] is True
    finally:
        env.close()


def test_real_mario_rl_env_passes_sb3_check_env() -> None:
    env = MarioRLEnv(
        env_id="SuperMarioBros-1-1-v0",
        frames_per_decision=4,
        max_steps=200,
        render_mode="rgb_array",
    )
    try:
        check_env(env, warn=True, skip_render_check=False)
    finally:
        env.close()


def test_multi_stage_env_cycles_stages_and_passes_check_env() -> None:
    assert world_stage_env_ids(1) == (
        "SuperMarioBros-1-1-v0",
        "SuperMarioBros-1-2-v0",
        "SuperMarioBros-1-3-v0",
        "SuperMarioBros-1-4-v0",
    )
    stage_envs = {
        "SuperMarioBros-1-1-v0": MarioRLEnv(
            env_id="SuperMarioBros-1-1-v0", adapter=FakeEmulatorBackend()
        ),
        "SuperMarioBros-1-2-v0": MarioRLEnv(
            env_id="SuperMarioBros-1-2-v0", adapter=FakeEmulatorBackend()
        ),
    }
    multi = MultiStageMarioEnv(
        env_ids=tuple(stage_envs.keys()),
        stage_envs=stage_envs,
        render_mode="rgb_array",
    )
    try:
        _, info1 = multi.reset(seed=1)
        _, info2 = multi.reset(seed=2)
        assert info1["env_id"] == "SuperMarioBros-1-1-v0"
        assert info2["env_id"] == "SuperMarioBros-1-2-v0"
        check_env(multi, warn=True, skip_render_check=False)
    finally:
        multi.close()
