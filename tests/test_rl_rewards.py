from __future__ import annotations

from typesafe_mario.rl.rewards import MarioRewardCalculator, RewardConfig
from typesafe_mario.state import MarioSnapshot


def _snapshot(**overrides: object) -> MarioSnapshot:
    defaults: dict[str, object] = {
        "goal": "Reach the flag in World 1-1 without dying.",
        "world": 1,
        "stage": 1,
        "area": 1,
        "x": 100,
        "y": 79,
        "dx": 0,
        "dy": 0,
        "direction": "nearly_stationary",
        "vertical_motion": "level_or_grounded",
        "airborne": False,
        "status": "small",
        "player_state": 8,
        "lives": 2,
        "coins": 0,
        "score": 0,
        "time_left": 395,
        "progress": 100,
        "best_progress": 100,
        "stalled_steps": 0,
        "grounded": True,
        "crossing_gap": False,
        "dead": False,
        "clear": False,
    }
    defaults.update(overrides)
    return MarioSnapshot(**defaults)  # type: ignore[arg-type]


def test_forward_progress_and_new_best_bonus() -> None:
    cfg = RewardConfig()
    calc = MarioRewardCalculator(cfg)
    prev = _snapshot(x=100)
    calc.reset(prev)

    curr = _snapshot(x=110)
    breakdown = calc.compute(prev, curr)

    assert breakdown.progress == 10 * cfg.progress_scale
    assert breakdown.new_best == 10 * cfg.new_best_bonus_scale
    assert breakdown.time_penalty == cfg.step_penalty
    assert breakdown.total > 0.0


def test_backward_movement_is_penalized_and_does_not_earn_new_best() -> None:
    cfg = RewardConfig()
    calc = MarioRewardCalculator(cfg)
    start = _snapshot(x=100)
    calc.reset(start)
    ahead = _snapshot(x=120)
    calc.compute(start, ahead)

    back = _snapshot(x=110)
    breakdown = calc.compute(ahead, back)
    assert breakdown.progress < 0.0
    assert breakdown.new_best == 0.0
    assert breakdown.total < 0.0


def test_gap_crossing_bonus_triggers_on_safe_landing() -> None:
    cfg = RewardConfig()
    calc = MarioRewardCalculator(cfg)
    takeoff = _snapshot(x=200, grounded=False, airborne=True, crossing_gap=True)
    calc.reset(takeoff)

    landed = _snapshot(x=240, grounded=True, airborne=False, crossing_gap=False)
    breakdown = calc.compute(takeoff, landed)
    assert breakdown.gap_clear == cfg.gap_clear_bonus


def test_death_penalty_and_stage_clear_bonus() -> None:
    cfg = RewardConfig()
    calc = MarioRewardCalculator(cfg)
    prev = _snapshot(x=300)
    calc.reset(prev)

    dead_breakdown = calc.compute(prev, _snapshot(x=300, dead=True), died=True)
    assert dead_breakdown.death == cfg.death_penalty
    assert dead_breakdown.total < -10.0

    calc.reset(prev)
    clear_breakdown = calc.compute(prev, _snapshot(x=3161, clear=True), cleared=True)
    assert clear_breakdown.stage_clear == cfg.stage_clear_bonus
    assert clear_breakdown.total > 50.0


def test_stall_penalty_applies_at_threshold() -> None:
    cfg = RewardConfig(stall_threshold=5, stall_penalty=-0.1)
    calc = MarioRewardCalculator(cfg)
    prev = _snapshot(x=100, stalled_steps=4)
    calc.reset(prev)

    stalled = _snapshot(x=100, stalled_steps=5)
    breakdown = calc.compute(prev, stalled)
    assert breakdown.stall == -0.1


def test_area_transition_does_not_penalize_x_reset() -> None:
    cfg = RewardConfig()
    calc = MarioRewardCalculator(cfg)
    underground = _snapshot(world=1, stage=2, area=2, x=2600)
    calc.reset(underground)

    overworld_exit = _snapshot(world=1, stage=2, area=1, x=120)
    breakdown = calc.compute(underground, overworld_exit)
    assert breakdown.progress == 0.0
    assert breakdown.new_best == 0.0
