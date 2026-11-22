from __future__ import annotations

import argparse
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from .env import MarioRLEnv, MultiStageMarioEnv, world_stage_env_ids


@dataclass(frozen=True)
class TrainConfig:
    env_id: str = "SuperMarioBros-1-1-v0"
    env_ids: tuple[str, ...] = ()
    total_timesteps: int = 1_500_000
    num_envs: int = 8
    frames_per_decision: int = 4
    max_steps: int = 1500
    max_stall_steps: int = 60
    random_noop_max: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.015
    net_arch: tuple[int, ...] = (256, 256)
    seed: int = 42
    device: str = "cpu"
    model_path: Path = Path("models/mario_ppo.zip")
    checkpoint_dir: Path = Path("models/checkpoints")
    checkpoint_freq: int = 100_000
    tensorboard_log: Path = Path("runs/mario_ppo")
    resume_from: Path | None = None
    target_clear_rate: float | None = None

    def resolved_env_ids(self) -> tuple[str, ...]:
        return self.env_ids if self.env_ids else (self.env_id,)


def _linear_schedule(initial_value: float, min_value: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)

    return schedule


def _make_env(
    env_ids: tuple[str, ...],
    *,
    frames_per_decision: int,
    max_steps: int,
    max_stall_steps: int,
    random_noop_max: int,
    seed: int,
    rank: int,
) -> Callable[[], Any]:
    def _init() -> Any:
        if len(env_ids) == 1:
            env: Any = MarioRLEnv(
                env_id=env_ids[0],
                frames_per_decision=frames_per_decision,
                max_steps=max_steps,
                max_stall_steps=max_stall_steps,
                random_noop_max=random_noop_max,
                render_mode="rgb_array",
            )
        else:
            # Offset starting stage per worker so all stages are trained in parallel.
            rotated = env_ids[rank % len(env_ids) :] + env_ids[: rank % len(env_ids)]
            env = MultiStageMarioEnv(
                env_ids=rotated,
                frames_per_decision=frames_per_decision,
                max_steps=max_steps,
                max_stall_steps=max_stall_steps,
                random_noop_max=random_noop_max,
                render_mode="rgb_array",
                sampling="round_robin",
            )
        env.reset(seed=seed + rank)
        return env

    return _init


class MarioProgressCallback(BaseCallback):
    """Track single- and multi-stage progress metrics and persist the best model."""

    def __init__(
        self,
        *,
        best_model_path: Path,
        window_size: int = 100,
        log_interval_steps: int = 20_000,
        target_clear_rate: float | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.best_model_path = best_model_path
        self.window_size = window_size
        self.log_interval_steps = log_interval_steps
        self.target_clear_rate = target_clear_rate
        self._recent_best_x: deque[int] = deque(maxlen=window_size)
        self._recent_clears: deque[int] = deque(maxlen=window_size)
        self._recent_deaths: deque[int] = deque(maxlen=window_size)
        self._recent_rewards: deque[float] = deque(maxlen=window_size)
        self._stage_clears: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=40))
        self._stage_best_x: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=40))
        self._best_clear_rate = -1.0
        self._best_mean_x = -1.0
        self._last_log_step = 0
        self._total_episodes = 0
        self._total_clears = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", ())
        infos = self.locals.get("infos", ())
        for done, info in zip(dones, infos, strict=False):
            if not done:
                continue
            self._total_episodes += 1
            best_x = int(info.get("best_x", info.get("x", 0)))
            cleared = int(bool(info.get("clear", False)))
            died = int(bool(info.get("dead", False)))
            ep_reward = float(info.get("episode_reward", 0.0))
            stage_id = str(info.get("env_id", "default"))

            self._recent_best_x.append(best_x)
            self._recent_clears.append(cleared)
            self._recent_deaths.append(died)
            self._recent_rewards.append(ep_reward)
            self._stage_clears[stage_id].append(cleared)
            self._stage_best_x[stage_id].append(best_x)
            self._total_clears += cleared

        if (
            self._recent_best_x
            and (self.num_timesteps - self._last_log_step) >= self.log_interval_steps
        ):
            self._last_log_step = self.num_timesteps
            mean_x = sum(self._recent_best_x) / len(self._recent_best_x)
            max_x = max(self._recent_best_x)
            clear_rate = sum(self._recent_clears) / len(self._recent_clears)
            death_rate = sum(self._recent_deaths) / len(self._recent_deaths)
            mean_rew = sum(self._recent_rewards) / len(self._recent_rewards)

            self.logger.record("mario/mean_best_x", mean_x)
            self.logger.record("mario/max_best_x", float(max_x))
            self.logger.record("mario/clear_rate", clear_rate)
            self.logger.record("mario/death_rate", death_rate)
            self.logger.record("mario/mean_episode_reward", mean_rew)
            self.logger.record("mario/total_clears", float(self._total_clears))

            stage_parts: list[str] = []
            if len(self._stage_clears) > 1:
                for stage_id in sorted(self._stage_clears):
                    s_clears = self._stage_clears[stage_id]
                    s_xs = self._stage_best_x[stage_id]
                    s_cr = sum(s_clears) / len(s_clears)
                    s_mx = sum(s_xs) / len(s_xs)
                    short = stage_id.replace("SuperMarioBros-", "").replace("-v0", "")
                    self.logger.record(f"mario_stage/{short}_clear_rate", s_cr)
                    self.logger.record(f"mario_stage/{short}_mean_x", s_mx)
                    stage_parts.append(f"{short}:{s_cr * 100:.0f}%(x={s_mx:.0f})")

            improved = (clear_rate > self._best_clear_rate) or (
                clear_rate == self._best_clear_rate and mean_x > self._best_mean_x
            )
            if improved and len(self._recent_best_x) >= 10:
                self._best_clear_rate = clear_rate
                self._best_mean_x = mean_x
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_path))

            if self.verbose:
                extra = f" | {' '.join(stage_parts)}" if stage_parts else ""
                print(
                    f"[step={self.num_timesteps:>8d}] eps={self._total_episodes:>5d} "
                    f"mean_x={mean_x:>6.1f} max_x={max_x:>4d} "
                    f"clear={clear_rate * 100:>5.1f}% death={death_rate * 100:>5.1f}% "
                    f"rew={mean_rew:>7.1f} total_clears={self._total_clears}{extra}",
                    flush=True,
                )

            if (
                self.target_clear_rate is not None
                and len(self._recent_clears) >= min(50, self.window_size)
                and clear_rate >= self.target_clear_rate
            ):
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_path))
                if self.verbose:
                    print(
                        f"Early stopping: reached target clear rate "
                        f"{clear_rate * 100:.1f}% >= {self.target_clear_rate * 100:.1f}%",
                        flush=True,
                    )
                return False
        return True


def train_ppo(config: TrainConfig) -> Path:
    """Train a PPO agent headless on one or more Super Mario Bros. stages."""
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.tensorboard_log.mkdir(parents=True, exist_ok=True)

    env_ids = config.resolved_env_ids()
    env_fns = [
        _make_env(
            env_ids,
            frames_per_decision=config.frames_per_decision,
            max_steps=config.max_steps,
            max_stall_steps=config.max_stall_steps,
            random_noop_max=config.random_noop_max,
            seed=config.seed,
            rank=rank,
        )
        for rank in range(config.num_envs)
    ]
    vec_env: Any = SubprocVecEnv(env_fns) if config.num_envs > 1 else DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    try:
        if config.resume_from is not None:
            model = PPO.load(
                str(config.resume_from),
                env=vec_env,
                device=config.device,
                tensorboard_log=str(config.tensorboard_log),
                custom_objects={
                    "learning_rate": _linear_schedule(
                        config.learning_rate, config.min_learning_rate
                    ),
                    "ent_coef": config.ent_coef,
                },
            )
        else:
            model = PPO(
                policy="MlpPolicy",
                env=vec_env,
                learning_rate=_linear_schedule(config.learning_rate, config.min_learning_rate),
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                clip_range=config.clip_range,
                ent_coef=config.ent_coef,
                policy_kwargs={"net_arch": list(config.net_arch)},
                tensorboard_log=str(config.tensorboard_log),
                seed=config.seed,
                device=config.device,
                verbose=0,
            )

        best_model_path = config.model_path.with_name(f"{config.model_path.stem}_best.zip")
        callbacks: list[BaseCallback] = [
            MarioProgressCallback(
                best_model_path=best_model_path,
                target_clear_rate=config.target_clear_rate,
                verbose=1,
            ),
        ]
        if config.checkpoint_freq > 0:
            save_freq = max(1, config.checkpoint_freq // max(1, config.num_envs))
            callbacks.append(
                CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(config.checkpoint_dir),
                    name_prefix=config.model_path.stem,
                )
            )

        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=False,
        )
        model.save(str(config.model_path))
        print(f"Saved final PPO model to {config.model_path.resolve()}", flush=True)
        if best_model_path.exists():
            print(f"Best checkpoint saved at {best_model_path.resolve()}", flush=True)
        return config.model_path
    finally:
        vec_env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PPO agent on Super Mario Bros.")
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Optional list of stage env IDs for multi-stage training",
    )
    parser.add_argument(
        "--world",
        type=int,
        default=None,
        help="Shortcut to train on all 4 stages of a world (e.g. --world 1)",
    )
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--frames-per-decision", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--max-stall-steps", type=int, default=60)
    parser.add_argument("--random-noop-max", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", type=Path, default=Path("models/mario_ppo.zip"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--tensorboard-log", type=Path, default=Path("runs/mario_ppo"))
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--target-clear-rate",
        type=float,
        default=None,
        help="Optional early-stop rolling clear rate in [0, 1] (e.g. 0.90)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.envs:
        env_ids = tuple(args.envs)
    elif args.world is not None:
        env_ids = world_stage_env_ids(args.world)
    else:
        env_ids = ()

    config = TrainConfig(
        env_id=args.env,
        env_ids=env_ids,
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        frames_per_decision=args.frames_per_decision,
        max_steps=args.max_steps,
        max_stall_steps=args.max_stall_steps,
        random_noop_max=args.random_noop_max,
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device=args.device,
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        tensorboard_log=args.tensorboard_log,
        resume_from=args.resume_from,
        target_clear_rate=args.target_clear_rate,
    )
    train_ppo(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
