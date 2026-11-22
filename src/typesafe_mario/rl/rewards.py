from __future__ import annotations

from dataclasses import dataclass

from ..state import MarioSnapshot


@dataclass(frozen=True)
class RewardConfig:
    progress_scale: float = 0.1
    backward_scale: float = 0.12
    new_best_bonus_scale: float = 0.05
    step_penalty: float = -0.02
    stall_threshold: int = 12
    stall_penalty: float = -0.05
    gap_clear_bonus: float = 2.0
    score_scale: float = 0.002
    score_bonus_cap: float = 1.0
    death_penalty: float = -15.0
    stage_clear_bonus: float = 50.0


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    progress: float
    new_best: float
    time_penalty: float
    stall: float
    gap_clear: float
    score: float
    death: float
    stage_clear: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "progress": self.progress,
            "new_best": self.new_best,
            "time_penalty": self.time_penalty,
            "stall": self.stall,
            "gap_clear": self.gap_clear,
            "score": self.score,
            "death": self.death,
            "stage_clear": self.stage_clear,
        }


class MarioRewardCalculator:
    """Shaped reward function for Super Mario Bros. World 1-1 RL training."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self._best_x: int = 0
        self._was_crossing_gap: bool = False

    def reset(self, initial_snapshot: MarioSnapshot) -> None:
        self._best_x = initial_snapshot.x
        self._was_crossing_gap = initial_snapshot.crossing_gap

    def compute(
        self,
        previous: MarioSnapshot,
        current: MarioSnapshot,
        *,
        died: bool = False,
        cleared: bool = False,
    ) -> RewardBreakdown:
        cfg = self.config
        is_dead = bool(died or current.dead)
        is_clear = bool(cleared or current.clear)

        # Horizontal progress (ignore warp/area transitions such as World 1-2 exit pipe)
        same_area = (
            current.world == previous.world
            and current.stage == previous.stage
            and current.area == previous.area
        )
        if not same_area:
            delta_x = 0
            self._best_x = current.x
        else:
            delta_x = current.x - previous.x

        if delta_x >= 0:
            progress_reward = delta_x * cfg.progress_scale
        else:
            progress_reward = delta_x * cfg.backward_scale

        # New furthest x reached
        new_best_delta = max(0, current.x - self._best_x)
        new_best_reward = new_best_delta * cfg.new_best_bonus_scale
        self._best_x = max(self._best_x, current.x)

        # Time & stall penalties
        time_penalty = cfg.step_penalty
        stall_reward = cfg.stall_penalty if current.stalled_steps >= cfg.stall_threshold else 0.0

        # Safe gap crossing bonus: committed to crossing a gap, then landed safely ahead
        gap_clear_reward = 0.0
        if self._was_crossing_gap and current.grounded and not is_dead and current.x > previous.x:
            gap_clear_reward = cfg.gap_clear_bonus
        self._was_crossing_gap = current.crossing_gap

        # Score delta bonus (enemies stomped, coins, bricks, power-ups)
        delta_score = max(0, current.score - previous.score)
        score_reward = min(delta_score * cfg.score_scale, cfg.score_bonus_cap)

        # Terminal rewards
        death_reward = cfg.death_penalty if is_dead else 0.0
        stage_clear_reward = cfg.stage_clear_bonus if is_clear else 0.0

        total = (
            progress_reward
            + new_best_reward
            + time_penalty
            + stall_reward
            + gap_clear_reward
            + score_reward
            + death_reward
            + stage_clear_reward
        )
        return RewardBreakdown(
            total=float(total),
            progress=float(progress_reward),
            new_best=float(new_best_reward),
            time_penalty=float(time_penalty),
            stall=float(stall_reward),
            gap_clear=float(gap_clear_reward),
            score=float(score_reward),
            death=float(death_reward),
            stage_clear=float(stage_clear_reward),
        )
