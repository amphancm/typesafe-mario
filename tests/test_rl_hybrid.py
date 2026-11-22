from __future__ import annotations

from collections.abc import Sequence

from typesafe_mario.actions import Action
from typesafe_mario.policy import Decision
from typesafe_mario.rl.hybrid import HybridConfig, HybridPolicy
from typesafe_mario.state import MarioSnapshot


class _StubPolicy:
    def __init__(self, action: Action, confidence: float) -> None:
        self.action = action
        self.confidence = confidence
        self.calls = 0

    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        del snapshot
        self.calls += 1
        return Decision(
            action=self.action,
            confidence=self.confidence,
            probabilities={a.value: (1.0 if a == self.action else 0.0) for a in actions},
            latency_ms=0.5,
        )


def _snapshot(*, stalled_steps: int = 0) -> MarioSnapshot:
    return MarioSnapshot(
        goal="Reach the flag in World 1-1 without dying.",
        world=1,
        stage=1,
        area=1,
        x=120,
        y=79,
        dx=4,
        dy=0,
        direction="moving_right",
        vertical_motion="level_or_grounded",
        airborne=False,
        status="small",
        player_state=8,
        lives=2,
        coins=0,
        score=0,
        time_left=390,
        progress=120,
        best_progress=120,
        stalled_steps=stalled_steps,
    )


def test_hybrid_uses_rl_when_confident_and_escalates_on_low_confidence() -> None:
    rl_high = _StubPolicy(Action.RIGHT_RUN, confidence=0.92)
    jev = _StubPolicy(Action.RIGHT_RUN_JUMP, confidence=0.88)
    hybrid = HybridPolicy(rl_high, jev, config=HybridConfig(confidence_threshold=0.6))

    actions = tuple(Action)
    decision = hybrid.choose(_snapshot(), actions)
    assert decision.action == Action.RIGHT_RUN
    assert jev.calls == 0
    assert hybrid.stats().api_reduction_rate == 1.0

    # Low confidence triggers Jev escalation
    rl_low = _StubPolicy(Action.RIGHT, confidence=0.35)
    hybrid_low = HybridPolicy(
        rl_low,
        jev,
        config=HybridConfig(confidence_threshold=0.6, min_steps_between_jev_calls=3),
    )
    d1 = hybrid_low.choose(_snapshot(), actions)
    d2 = hybrid_low.choose(_snapshot(), actions)  # cooldown blocks Jev
    assert d1.action == Action.RIGHT_RUN_JUMP
    assert d2.action == Action.RIGHT
    assert jev.calls == 1
    stats = hybrid_low.stats()
    assert stats.jev_decisions == 1
    assert stats.rl_decisions == 1
    assert stats.escalation_reasons["low_rl_confidence"] == 1


def test_hybrid_escalates_on_stall_and_respects_budget() -> None:
    rl = _StubPolicy(Action.RIGHT_RUN, confidence=0.95)
    jev = _StubPolicy(Action.RIGHT_RUN_JUMP, confidence=0.9)
    hybrid = HybridPolicy(
        rl,
        jev,
        config=HybridConfig(
            stall_threshold=5,
            min_steps_between_jev_calls=1,
            max_jev_calls_per_episode=1,
        ),
    )
    actions = tuple(Action)
    first = hybrid.choose(_snapshot(stalled_steps=6), actions)
    second = hybrid.choose(_snapshot(stalled_steps=7), actions)

    assert first.action == Action.RIGHT_RUN_JUMP
    assert second.action == Action.RIGHT_RUN
    assert jev.calls == 1
    assert hybrid.stats().escalation_reasons["stalled"] == 1
