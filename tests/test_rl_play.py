from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from typesafe_mario.actions import Action
from typesafe_mario.cli import build_parser
from typesafe_mario.dashboard import DashboardCommand, DashboardConfig
from typesafe_mario.policy import Decision
from typesafe_mario.rl.env import MarioRLEnv
from typesafe_mario.rl.play import play_rl


def test_mario_rl_env_captures_distinct_step_frames() -> None:
    env = MarioRLEnv(
        env_id="SuperMarioBros-1-1-v0",
        frames_per_decision=4,
        max_steps=10,
        render_mode="rgb_array",
        capture_step_frames=True,
    )
    try:
        env.reset(seed=123)
        env.step(1)
        frames = env.last_step_frames
        assert len(frames) == 4
        # Each captured frame must be an independent copy, not aliased to the shared nes_py buffer.
        assert frames[0] is not frames[1]
        assert not np.shares_memory(frames[0], frames[1])
    finally:
        env.close()


def test_play_rl_draws_every_nes_frame_at_normal_speed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    drawn_frames: list[Any] = []
    captured_config: list[DashboardConfig | None] = []

    class FakeRLPolicy:
        def __init__(self, model_path: Path, *, deterministic: bool = True, device: str = "cpu") -> None:
            del model_path, deterministic, device

        def choose(self, snapshot: Any, actions: Any) -> Decision:
            del snapshot, actions
            return Decision(action=Action.RIGHT, confidence=1.0, probabilities={"right": 1.0}, latency_ms=1.0)

    class FakeEnv:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["capture_step_frames"] is True
            self.last_snapshot = object()
            self.last_frame = np.zeros((4, 4, 3), dtype=np.uint8)
            self._frames: tuple[Any, ...] = ()
            self._steps = 0

        @property
        def last_step_frames(self) -> tuple[Any, ...]:
            return self._frames

        def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
            del seed
            return np.zeros(1), {"clear": False, "dead": False, "best_x": 40}

        def step(self, action_idx: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            del action_idx
            self._steps += 1
            self._frames = tuple(np.full((4, 4, 3), i, dtype=np.uint8) for i in range(4))
            self.last_frame = self._frames[-1]
            done = self._steps >= 2
            return (
                np.zeros(1),
                1.0,
                done,
                False,
                {"clear": done, "dead": False, "best_x": 100, "frames_executed": 4},
            )

        def close(self) -> None:
            return None

    class FakeDashboard:
        def __init__(self, config: DashboardConfig | None = None) -> None:
            captured_config.append(config)

        def draw(self, frame: Any, snapshot: Any, decision: Any, **kwargs: Any) -> DashboardCommand:
            del snapshot, decision
            drawn_frames.append(frame)
            if kwargs.get("run_ended") and len(drawn_frames) >= 8:
                return DashboardCommand.QUIT
            return DashboardCommand.CONTINUE

        def close(self) -> None:
            return None

    monkeypatch.setattr("typesafe_mario.rl.play.RLPolicy", FakeRLPolicy)
    monkeypatch.setattr("typesafe_mario.rl.play.MarioRLEnv", FakeEnv)
    monkeypatch.setattr("typesafe_mario.rl.play.LiveDashboard", FakeDashboard)

    summary = play_rl(tmp_path / "dummy.zip", display="dashboard", speed=1.0)
    assert summary["steps"] == 2
    assert summary["clear"] is True
    assert captured_config == [DashboardConfig(fps=60)]
    # 2 decisions * 4 NES frames per decision = 8 rendered frames at 60 FPS (normal 1.0x speed).
    assert len(drawn_frames) == 8


def test_rl_play_cli_speed_default_and_validation(tmp_path: Path) -> None:
    args = build_parser().parse_args(["rl-play"])
    assert args.speed == 1.0

    with pytest.raises(ValueError, match="speed must be greater than 0"):
        play_rl(tmp_path / "dummy.zip", speed=0.0)


def test_play_rl_runs_clear_animation_before_showing_finished_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    draw_calls: list[tuple[bool, bool]] = []

    class FakeRLPolicy:
        def __init__(self, model_path: Path, *, deterministic: bool = True, device: str = "cpu") -> None:
            del model_path, deterministic, device

        def choose(self, snapshot: Any, actions: Any) -> Decision:
            del snapshot, actions
            return Decision(action=Action.RIGHT, confidence=1.0, probabilities={"right": 1.0}, latency_ms=1.0)

    class FakeEnv:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.last_snapshot = object()
            self.last_frame = np.zeros((4, 4, 3), dtype=np.uint8)
            self._frames: tuple[Any, ...] = ()
            self._anim_remaining = 3

        @property
        def last_step_frames(self) -> tuple[Any, ...]:
            return self._frames

        def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
            del seed
            return np.zeros(1), {"clear": False, "dead": False, "best_x": 40}

        def step(self, action_idx: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            del action_idx
            self._frames = (np.ones((4, 4, 3), dtype=np.uint8),)
            self.last_frame = self._frames[-1]
            return (
                np.zeros(1),
                10.0,
                True,
                False,
                {"clear": True, "dead": False, "best_x": 3161, "frames_executed": 1},
            )

        def step_clear_animation(self) -> Any | None:
            if self._anim_remaining <= 0:
                return None
            self._anim_remaining -= 1
            self.last_frame = np.full((4, 4, 3), 2 + self._anim_remaining, dtype=np.uint8)
            return self.last_frame

        def close(self) -> None:
            return None

    class FakeDashboard:
        def __init__(self, config: DashboardConfig | None = None) -> None:
            del config

        def draw(self, frame: Any, snapshot: Any, decision: Any, **kwargs: Any) -> DashboardCommand:
            del frame, snapshot, decision
            run_ended = bool(kwargs.get("run_ended"))
            finishing = bool(kwargs.get("finishing"))
            draw_calls.append((run_ended, finishing))
            if run_ended and not finishing:
                return DashboardCommand.QUIT
            return DashboardCommand.CONTINUE

        def close(self) -> None:
            return None

    monkeypatch.setattr("typesafe_mario.rl.play.RLPolicy", FakeRLPolicy)
    monkeypatch.setattr("typesafe_mario.rl.play.MarioRLEnv", FakeEnv)
    monkeypatch.setattr("typesafe_mario.rl.play.LiveDashboard", FakeDashboard)

    summary = play_rl(tmp_path / "dummy.zip", display="dashboard", speed=1.0)
    assert summary["clear"] is True
    # 1 step frame + 3 clear-animation frames with finishing=True, then 1 final finished frame with finishing=False.
    assert draw_calls == [
        (True, True),
        (True, True),
        (True, True),
        (True, True),
        (True, False),
    ]
