from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from typesafe_mario.actions import Action
from typesafe_mario.audio import (
    AudioConfig,
    MarioAudioEngine,
    detect_audio_events,
    infer_music_theme,
    note_frequency,
    synthesize_bgm,
    synthesize_sfx,
)
from typesafe_mario.cli import build_parser
from typesafe_mario.dashboard import DashboardCommand, DashboardConfig
from typesafe_mario.policy import Decision
from typesafe_mario.rl.play import play_rl
from typesafe_mario.state import MarioSnapshot


def _snapshot(**overrides: Any) -> MarioSnapshot:
    base = MarioSnapshot(
        goal="Clear 1-1",
        world=1,
        stage=1,
        area=1,
        x=40,
        y=79,
        dx=0,
        dy=0,
        direction="idle",
        vertical_motion="level_or_grounded",
        airborne=False,
        status="small",
        player_state=8,
        lives=2,
        coins=0,
        score=0,
        time_left=400,
        progress=40,
        best_progress=40,
        stalled_steps=0,
        grounded=True,
        jump_phase="grounded",
    )
    return replace(base, **overrides)


def test_note_frequency_matches_standard_tuning() -> None:
    assert note_frequency(None) == 0.0
    assert note_frequency("-") == 0.0
    assert note_frequency("A4") == pytest.approx(440.0, rel=1e-4)
    assert note_frequency("C5") == pytest.approx(523.251, rel=1e-3)
    assert note_frequency("E6") == pytest.approx(1318.51, rel=1e-3)


@pytest.mark.parametrize("theme", ["overworld", "underground", "castle", "water"])
@pytest.mark.parametrize("fast", [False, True])
def test_synthesize_bgm_returns_valid_stereo_int16(theme: str, fast: bool) -> None:
    pcm = synthesize_bgm(theme, fast=fast, sample_rate=22_050)
    assert pcm.dtype == np.int16
    assert pcm.ndim == 2
    assert pcm.shape[1] == 2
    assert pcm.shape[0] > 22_050
    assert int(np.max(np.abs(pcm))) > 1000


@pytest.mark.parametrize(
    "sfx_name",
    [
        "jump_small",
        "jump_big",
        "coin",
        "stomp",
        "bump",
        "powerup",
        "powerdown",
        "one_up",
        "hurry_up",
        "stage_clear",
        "death",
    ],
)
def test_synthesize_sfx_returns_valid_stereo_int16(sfx_name: str) -> None:
    pcm = synthesize_sfx(sfx_name, sample_rate=22_050)
    assert pcm.dtype == np.int16
    assert pcm.ndim == 2
    assert pcm.shape[1] == 2
    assert pcm.shape[0] > 1000
    assert int(np.max(np.abs(pcm))) > 1000


def test_infer_music_theme_from_ram_and_stage_fallback() -> None:
    snap = _snapshot(world=1, stage=1)
    assert infer_music_theme(snap, area_type=0) == "water"
    assert infer_music_theme(snap, area_type=1) == "overworld"
    assert infer_music_theme(snap, area_type=2) == "underground"
    assert infer_music_theme(snap, area_type=3) == "castle"

    assert infer_music_theme(_snapshot(world=1, stage=1)) == "overworld"
    assert infer_music_theme(_snapshot(world=1, stage=2)) == "underground"
    assert infer_music_theme(_snapshot(world=1, stage=4)) == "castle"


def test_detect_audio_events_covers_core_gameplay_transitions() -> None:
    grounded = _snapshot()
    assert detect_audio_events(None, grounded) == []

    # Small jump takeoff
    jumped = _snapshot(grounded=False, airborne=True, dy=5, jump_phase="rising")
    assert detect_audio_events(grounded, jumped) == ["jump_small"]

    # Big jump takeoff
    big_grounded = _snapshot(status="tall")
    big_jumped = _snapshot(status="tall", grounded=False, airborne=True, dy=5, jump_phase="rising")
    assert detect_audio_events(big_grounded, big_jumped) == ["jump_big"]

    # Coin pickup
    assert detect_audio_events(grounded, _snapshot(coins=1, score=200)) == ["coin"]

    # Enemy stomp (score increases without coin or powerup)
    assert detect_audio_events(grounded, _snapshot(score=100)) == ["stomp"]

    # Powerup & powerdown
    assert detect_audio_events(grounded, _snapshot(status="tall", score=1000)) == ["powerup"]
    assert detect_audio_events(_snapshot(status="tall"), _snapshot(status="small")) == ["powerdown"]

    # 1-UP
    assert detect_audio_events(grounded, _snapshot(lives=3)) == ["one_up"]

    # Hurry up
    assert detect_audio_events(_snapshot(time_left=100), _snapshot(time_left=99)) == ["hurry_up"]

    # Stage clear and death
    assert detect_audio_events(grounded, _snapshot(clear=True)) == ["stage_clear"]
    assert detect_audio_events(grounded, _snapshot(dead=True)) == ["death"]


def test_audio_engine_mute_toggle_and_disabled_mode() -> None:
    engine = MarioAudioEngine(AudioConfig(enabled=False, volume=0.5))
    assert not engine.available
    assert engine.muted is True
    assert engine.toggle_mute() is False
    assert engine.toggle_mute() is True
    assert engine.update(_snapshot()) == []
    engine.close()

    with pytest.raises(ValueError, match="volume must be between"):
        MarioAudioEngine(AudioConfig(enabled=False, volume=1.5))


def test_cli_sound_and_volume_flags() -> None:
    args_default = build_parser().parse_args(["rl-play"])
    assert args_default.sound is True
    assert args_default.volume == pytest.approx(0.5)

    args_muted = build_parser().parse_args(["rl-play", "--no-sound", "--volume", "0.25"])
    assert args_muted.sound is False
    assert args_muted.volume == pytest.approx(0.25)

    play_args = build_parser().parse_args(["play", "--no-sound"])
    assert play_args.sound is False


def test_play_rl_wires_audio_and_handles_mute_toggle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_events: list[str] = []

    class FakeAudio:
        def __init__(self, config: AudioConfig | None = None) -> None:
            self.config = config or AudioConfig()
            self.muted = not self.config.enabled
            audio_events.append(f"init:{self.config.enabled}:{self.config.volume}")

        def reset(self) -> None:
            audio_events.append("reset")

        def update(self, snapshot: Any, *, area_type: int | None = None, finishing: bool = False) -> list[str]:
            del snapshot, area_type, finishing
            audio_events.append("update")
            return []

        def toggle_mute(self) -> bool:
            self.muted = not self.muted
            audio_events.append(f"mute:{self.muted}")
            return self.muted

        def close(self) -> None:
            audio_events.append("close")

    class FakeRLPolicy:
        def __init__(self, model_path: Path, *, deterministic: bool = True, device: str = "cpu") -> None:
            del model_path, deterministic, device

        def choose(self, snapshot: Any, actions: Any) -> Decision:
            del snapshot, actions
            return Decision(action=Action.RIGHT, confidence=1.0, probabilities={"right": 1.0}, latency_ms=1.0)

    class FakeEnv:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.last_snapshot = _snapshot()
            self.last_frame = np.zeros((4, 4, 3), dtype=np.uint8)
            self._frames: tuple[Any, ...] = (self.last_frame,)

        @property
        def last_step_frames(self) -> tuple[Any, ...]:
            return self._frames

        def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
            del seed
            return np.zeros(1), {"clear": False, "dead": False, "best_x": 40}

        def step(self, action_idx: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            del action_idx
            self.last_snapshot = _snapshot(clear=True)
            return (
                np.zeros(1),
                1.0,
                True,
                False,
                {"clear": True, "dead": False, "best_x": 100, "frames_executed": 1},
            )

        def close(self) -> None:
            return None

    draw_count = 0

    class FakeDashboard:
        def __init__(self, config: DashboardConfig | None = None) -> None:
            del config

        def draw(self, frame: Any, snapshot: Any, decision: Any, **kwargs: Any) -> DashboardCommand:
            nonlocal draw_count
            del frame, snapshot, decision
            draw_count += 1
            if draw_count == 1:
                assert kwargs.get("sound_enabled") is True
                return DashboardCommand.TOGGLE_MUTE
            assert kwargs.get("sound_enabled") is False
            return DashboardCommand.QUIT

        def close(self) -> None:
            return None

    monkeypatch.setattr("typesafe_mario.rl.play.MarioAudioEngine", FakeAudio)
    monkeypatch.setattr("typesafe_mario.rl.play.RLPolicy", FakeRLPolicy)
    monkeypatch.setattr("typesafe_mario.rl.play.MarioRLEnv", FakeEnv)
    monkeypatch.setattr("typesafe_mario.rl.play.LiveDashboard", FakeDashboard)

    play_rl(tmp_path / "dummy.zip", display="dashboard", sound=True, volume=0.6)
    assert "init:True:0.6" in audio_events
    assert "reset" in audio_events
    assert "update" in audio_events
    assert "mute:True" in audio_events
    assert audio_events[-1] == "close"
