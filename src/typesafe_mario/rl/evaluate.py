from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .env import MarioRLEnv, world_stage_env_ids


@dataclass(frozen=True)
class EvaluationMetrics:
    model_path: str
    env_id: str
    episodes: int
    clears: int
    deaths: int
    truncations: int
    completion_rate: float
    death_rate: float
    truncation_rate: float
    mean_reward: float
    std_reward: float
    mean_best_x: float
    max_best_x: int
    min_best_x: int
    mean_steps: float
    deterministic: bool
    random_noop_max: int

    def format_report(self) -> str:
        return (
            f"=== RL Evaluation Report ({self.episodes} episodes) ===\n"
            f"Model:           {self.model_path}\n"
            f"Environment:     {self.env_id}\n"
            f"Mode:            {'deterministic' if self.deterministic else 'stochastic'} "
            f"(random_noop_max={self.random_noop_max})\n"
            f"Completion Rate: {self.completion_rate * 100:.1f}% ({self.clears}/{self.episodes})\n"
            f"Death Rate:      {self.death_rate * 100:.1f}% ({self.deaths}/{self.episodes})\n"
            f"Truncation Rate: {self.truncation_rate * 100:.1f}% "
            f"({self.truncations}/{self.episodes})\n"
            f"Mean Reward:     {self.mean_reward:.2f} +/- {self.std_reward:.2f}\n"
            f"Best X:          mean={self.mean_best_x:.1f}, min={self.min_best_x}, "
            f"max={self.max_best_x}\n"
            f"Mean Steps:      {self.mean_steps:.1f}"
        )


def evaluate_policy(
    model_path: Path,
    *,
    episodes: int = 100,
    env_id: str = "SuperMarioBros-1-1-v0",
    frames_per_decision: int = 4,
    max_steps: int = 1500,
    max_stall_steps: int = 60,
    deterministic: bool = True,
    random_noop_max: int = 4,
    seed: int = 1000,
    device: str = "cpu",
) -> EvaluationMetrics:
    """Evaluate a trained PPO policy over N episodes on a single stage."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    model = PPO.load(str(model_path), device=device)
    env = MarioRLEnv(
        env_id=env_id,
        frames_per_decision=frames_per_decision,
        max_steps=max_steps,
        max_stall_steps=max_stall_steps,
        random_noop_max=random_noop_max,
        render_mode="rgb_array",
    )

    rewards: list[float] = []
    best_xs: list[int] = []
    steps_list: list[int] = []
    clears = 0
    deaths = 0
    truncations = 0

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            terminated = False
            truncated = False
            ep_reward = 0.0
            last_info: dict[str, object] = {}

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, last_info = env.step(int(action))
                ep_reward += float(reward)

            cleared = bool(last_info.get("clear", False))
            died = bool(last_info.get("dead", False))
            if cleared:
                clears += 1
            elif died:
                deaths += 1
            else:
                truncations += 1

            rewards.append(ep_reward)
            best_xs.append(int(last_info.get("best_x", 0)))
            steps_list.append(int(last_info.get("steps", 0)))
    finally:
        env.close()

    return EvaluationMetrics(
        model_path=str(model_path),
        env_id=env_id,
        episodes=episodes,
        clears=clears,
        deaths=deaths,
        truncations=truncations,
        completion_rate=clears / episodes,
        death_rate=deaths / episodes,
        truncation_rate=truncations / episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_best_x=float(np.mean(best_xs)),
        max_best_x=max(best_xs),
        min_best_x=min(best_xs),
        mean_steps=float(np.mean(steps_list)),
        deterministic=deterministic,
        random_noop_max=random_noop_max,
    )


def evaluate_multi_stage(
    model_path: Path,
    env_ids: Sequence[str],
    *,
    episodes_per_stage: int = 25,
    frames_per_decision: int = 4,
    max_steps: int = 1500,
    max_stall_steps: int = 60,
    deterministic: bool = True,
    random_noop_max: int = 4,
    seed: int = 1000,
    device: str = "cpu",
) -> list[EvaluationMetrics]:
    """Evaluate a trained policy across multiple stages and return per-stage metrics."""
    results: list[EvaluationMetrics] = []
    for idx, stage_env_id in enumerate(env_ids):
        metrics = evaluate_policy(
            model_path,
            episodes=episodes_per_stage,
            env_id=stage_env_id,
            frames_per_decision=frames_per_decision,
            max_steps=max_steps,
            max_stall_steps=max_stall_steps,
            deterministic=deterministic,
            random_noop_max=random_noop_max,
            seed=seed + idx * 1000,
            device=device,
        )
        results.append(metrics)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Mario PPO policy.")
    parser.add_argument("--model", type=Path, default=Path("models/mario_ppo.zip"))
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--envs", nargs="+", default=None, help="Evaluate across multiple stages")
    parser.add_argument(
        "--world",
        type=int,
        default=None,
        help="Evaluate across all 4 stages of a world (e.g. --world 1)",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--frames-per-decision", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--max-stall-steps", type=int, default=60)
    parser.add_argument("--random-noop-max", type=int, default=4)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.envs:
        stage_ids = tuple(args.envs)
    elif args.world is not None:
        stage_ids = world_stage_env_ids(args.world)
    else:
        stage_ids = (args.env,)

    if len(stage_ids) == 1:
        metrics = evaluate_policy(
            args.model,
            episodes=args.episodes,
            env_id=stage_ids[0],
            frames_per_decision=args.frames_per_decision,
            max_steps=args.max_steps,
            max_stall_steps=args.max_stall_steps,
            deterministic=not args.stochastic,
            random_noop_max=args.random_noop_max,
            seed=args.seed,
            device=args.device,
        )
        print(metrics.format_report())
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8")
            print(f"Saved evaluation report to {args.output.resolve()}")
        return 0

    results = evaluate_multi_stage(
        args.model,
        stage_ids,
        episodes_per_stage=args.episodes,
        frames_per_decision=args.frames_per_decision,
        max_steps=args.max_steps,
        max_stall_steps=args.max_stall_steps,
        deterministic=not args.stochastic,
        random_noop_max=args.random_noop_max,
        seed=args.seed,
        device=args.device,
    )
    for item in results:
        print(item.format_report())
        print()

    avg_clear = float(np.mean([r.completion_rate for r in results]))
    avg_death = float(np.mean([r.death_rate for r in results]))
    print(
        f"=== Multi-Stage Summary ({len(results)} stages x {args.episodes} eps) ===\n"
        f"Avg Completion Rate: {avg_clear * 100:.1f}%\n"
        f"Avg Death Rate:      {avg_death * 100:.1f}%"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_path": str(args.model),
            "avg_completion_rate": avg_clear,
            "avg_death_rate": avg_death,
            "stages": [asdict(r) for r in results],
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved multi-stage report to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
