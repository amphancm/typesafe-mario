from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ..actions import Action
from ..policy import Decision, Policy
from ..state import MarioSnapshot


@dataclass(frozen=True)
class HybridConfig:
    confidence_threshold: float = 0.55
    stall_threshold: int = 8
    escalate_on_missed_takeoff: bool = True
    missed_takeoff_confidence_threshold: float = 0.75
    min_steps_between_jev_calls: int = 4
    max_jev_calls_per_episode: int = 50


@dataclass(frozen=True)
class HybridStats:
    total_decisions: int
    rl_decisions: int
    jev_decisions: int
    api_reduction_rate: float
    escalation_reasons: dict[str, int]


class HybridPolicy:
    """Hybrid AI policy: fast local RL policy with Jev Strategic Agent escalation.

    The RL policy handles routine frame-to-frame control at <1ms with zero API calls.
    Jev is consulted only when RL confidence is low, Mario is stalled, or a complex
    high-risk hazard occurs (subject to cooldown and per-episode API budget).
    """

    def __init__(
        self,
        rl_policy: Policy,
        strategic_policy: Policy | None = None,
        *,
        config: HybridConfig | None = None,
    ) -> None:
        self.rl_policy = rl_policy
        self.strategic_policy = strategic_policy
        self.config = config or HybridConfig()
        self._step = 0
        self._last_jev_step = -10_000
        self._rl_decisions = 0
        self._jev_decisions = 0
        self._reasons: Counter[str] = Counter()

    def reset_episode(self) -> None:
        self._step = 0
        self._last_jev_step = -10_000

    def stats(self) -> HybridStats:
        total = self._rl_decisions + self._jev_decisions
        reduction = 1.0 - (self._jev_decisions / total) if total > 0 else 1.0
        return HybridStats(
            total_decisions=total,
            rl_decisions=self._rl_decisions,
            jev_decisions=self._jev_decisions,
            api_reduction_rate=reduction,
            escalation_reasons=dict(self._reasons),
        )

    def should_consult_jev(
        self,
        snapshot: MarioSnapshot,
        rl_decision: Decision,
    ) -> tuple[bool, str]:
        if self.strategic_policy is None:
            return False, "no_strategic_policy"

        cfg = self.config
        if self._jev_decisions >= cfg.max_jev_calls_per_episode:
            return False, "jev_budget_exhausted"

        if (self._step - self._last_jev_step) < cfg.min_steps_between_jev_calls:
            return False, "jev_cooldown"

        if snapshot.stalled_steps >= cfg.stall_threshold:
            return True, "stalled"

        if rl_decision.confidence < cfg.confidence_threshold:
            return True, "low_rl_confidence"

        if cfg.escalate_on_missed_takeoff:
            threat = snapshot.threat_features()
            if (
                threat.get("takeoff_window_already_missed", False)
                and rl_decision.confidence < cfg.missed_takeoff_confidence_threshold
            ):
                return True, "missed_takeoff_hazard"

        return False, "rl_confident"

    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        self._step += 1
        rl_decision = self.rl_policy.choose(snapshot, actions)
        consult, reason = self.should_consult_jev(snapshot, rl_decision)

        if consult and self.strategic_policy is not None:
            self._last_jev_step = self._step
            self._jev_decisions += 1
            self._reasons[reason] += 1
            return self.strategic_policy.choose(snapshot, actions)

        self._rl_decisions += 1
        return rl_decision

    def close(self) -> None:
        for policy in (self.rl_policy, self.strategic_policy):
            close_fn = getattr(policy, "close", None)
            if callable(close_fn):
                close_fn()
