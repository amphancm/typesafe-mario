from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .policy import HeuristicPolicy, TypeSafePolicy
from .runner import run_episode
from .state import MarioStateParser


def _demo_ram() -> bytearray:
    ram = bytearray(0x0800)
    # Mario position and one goomba ahead.
    ram[0x006D] = 0
    ram[0x0086] = 172
    ram[0x000F] = 1
    ram[0x0016] = 0x06
    ram[0x006E] = 0
    ram[0x0087] = 214
    ram[0x00CF] = 79
    # Fill the tile-map row immediately below Mario with solid ground.
    ground_row = (79 + 16 - 32) // 16
    for column in range(16):
        ram[0x0500 + ground_row * 16 + column] = 1
    return ram


def state_demo() -> int:
    info = {
        "world": 1,
        "stage": 1,
        "area": 1,
        "x_pos": 172,
        "y_pos": 79,
        "y_pixel": 79,
        "left_x_pos": 60,
        "progress": 172,
        "progress_max": 172,
        "status": "small",
        "player_state": 8,
        "life": 2,
        "coins": 0,
        "score": 200,
        "time": 387,
        "death": False,
        "clear": False,
    }
    snapshot = MarioStateParser().parse(info, _demo_ram(), previous_action="right")
    print(json.dumps(snapshot.to_state(), indent=2))
    print("\n--- TEXT VIEW ---\n")
    print(snapshot.to_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="typesafe-mario")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("state-demo", help="Show structured and text state without API calls")

    play = subparsers.add_parser("play", help="Run a Mario episode")
    play.add_argument("--env", default="SuperMarioBros-1-1-v0")
    play.add_argument(
        "--frames-per-decision",
        type=int,
        default=8,
        help="Minimum macro duration in emulator frames",
    )
    play.add_argument("--max-decisions", type=int, default=2000)
    play.add_argument("--seed", type=int, default=123)
    play.add_argument("--policy", choices=("typesafe", "heuristic"), default="typesafe")
    play.add_argument(
        "--display",
        choices=("dashboard", "game", "none"),
        default="dashboard",
        help="Combined telemetry dashboard, plain game window, or headless mode",
    )
    play.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    play.add_argument(
        "--screenshot",
        type=Path,
        help="Save the first populated dashboard frame as a PNG",
    )
    play.add_argument(
        "--sound",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable synthesized 8-bit Mario music and sound effects (use --no-sound to mute)",
    )
    play.add_argument(
        "--volume",
        type=float,
        default=0.5,
        help="Master audio volume from 0.0 to 1.0 (default: 0.5)",
    )

    rl_bc = subparsers.add_parser(
        "rl-bc",
        help="Pretrain a PPO policy via Behavior Cloning from Jev JSONL logs",
    )
    rl_bc.add_argument("--inputs", type=Path, nargs="+", default=[Path("artifacts")])
    rl_bc.add_argument("--model-path", type=Path, default=Path("models/mario_bc.zip"))
    rl_bc.add_argument("--min-confidence", type=float, default=0.0)
    rl_bc.add_argument("--epochs", type=int, default=30)
    rl_bc.add_argument("--device", default="cpu")

    rl_train = subparsers.add_parser("rl-train", help="Train a PPO agent on Super Mario Bros.")
    rl_train.add_argument("--env", default="SuperMarioBros-1-1-v0")
    rl_train.add_argument("--envs", nargs="+", default=None)
    rl_train.add_argument("--world", type=int, default=None)
    rl_train.add_argument("--timesteps", type=int, default=1_500_000)
    rl_train.add_argument("--num-envs", type=int, default=8)
    rl_train.add_argument("--frames-per-decision", type=int, default=4)
    rl_train.add_argument("--model-path", type=Path, default=Path("models/mario_ppo.zip"))
    rl_train.add_argument("--resume-from", type=Path, default=None)
    rl_train.add_argument("--device", default="cpu")

    rl_eval = subparsers.add_parser("rl-eval", help="Evaluate a trained PPO policy")
    rl_eval.add_argument("--model", type=Path, default=Path("models/mario_ppo.zip"))
    rl_eval.add_argument("--env", default="SuperMarioBros-1-1-v0")
    rl_eval.add_argument("--envs", nargs="+", default=None)
    rl_eval.add_argument("--world", type=int, default=None)
    rl_eval.add_argument("--episodes", type=int, default=100)
    rl_eval.add_argument("--frames-per-decision", type=int, default=4)
    rl_eval.add_argument("--random-noop-max", type=int, default=4)
    rl_eval.add_argument("--stochastic", action="store_true")
    rl_eval.add_argument("--output", type=Path, default=None)

    rl_play = subparsers.add_parser("rl-play", help="Play Mario using a trained PPO model")
    rl_play.add_argument("--model", type=Path, default=Path("models/mario_ppo.zip"))
    rl_play.add_argument("--env", default="SuperMarioBros-1-1-v0")
    rl_play.add_argument("--frames-per-decision", type=int, default=4)
    rl_play.add_argument("--display", choices=("dashboard", "game", "none"), default="dashboard")
    rl_play.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier relative to 60 FPS NES speed (1.0 = normal)",
    )
    rl_play.add_argument(
        "--sound",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable synthesized 8-bit Mario music and sound effects (use --no-sound to mute)",
    )
    rl_play.add_argument(
        "--volume",
        type=float,
        default=0.5,
        help="Master audio volume from 0.0 to 1.0 (default: 0.5)",
    )
    rl_play.add_argument("--stochastic", action="store_true")
    rl_play.add_argument("--hybrid", action="store_true")
    rl_play.add_argument("--confidence-threshold", type=float, default=0.55)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "state-demo":
        return state_demo()
    if args.command == "play":
        policy = TypeSafePolicy() if args.policy == "typesafe" else HeuristicPolicy()
        log_path = run_episode(
            env_id=args.env,
            policy=policy,
            frames_per_decision=args.frames_per_decision,
            max_decisions=args.max_decisions,
            seed=args.seed,
            artifacts_dir=args.artifacts_dir,
            display=args.display,
            screenshot_path=args.screenshot,
            sound=args.sound,
            volume=args.volume,
        )
        print(f"Run log: {log_path.resolve()}")
        return 0
    if args.command == "rl-bc":
        from .rl.bc import load_demonstrations, train_bc

        dataset = load_demonstrations(args.inputs, min_confidence=args.min_confidence)
        result = train_bc(
            dataset,
            model_path=args.model_path,
            epochs=args.epochs,
            device=args.device,
        )
        val_str = f"{result.val_accuracy * 100:.1f}%" if result.val_accuracy is not None else "n/a"
        print(
            f"BC complete: samples={result.samples} train_acc={result.final_train_accuracy * 100:.1f}% "
            f"val_acc={val_str} saved={result.model_path.resolve()}"
        )
        return 0
    if args.command == "rl-train":
        from .rl.env import world_stage_env_ids
        from .rl.train import TrainConfig, train_ppo

        if args.envs:
            env_ids = tuple(args.envs)
        elif args.world is not None:
            env_ids = world_stage_env_ids(args.world)
        else:
            env_ids = ()

        train_ppo(
            TrainConfig(
                env_id=args.env,
                env_ids=env_ids,
                total_timesteps=args.timesteps,
                num_envs=args.num_envs,
                frames_per_decision=args.frames_per_decision,
                model_path=args.model_path,
                resume_from=args.resume_from,
                device=args.device,
            )
        )
        return 0
    if args.command == "rl-eval":
        from .rl.evaluate import main as eval_main

        eval_argv = [
            "--model",
            str(args.model),
            "--env",
            args.env,
            "--episodes",
            str(args.episodes),
            "--frames-per-decision",
            str(args.frames_per_decision),
            "--random-noop-max",
            str(args.random_noop_max),
        ]
        if args.envs:
            eval_argv.extend(["--envs", *args.envs])
        if args.world is not None:
            eval_argv.extend(["--world", str(args.world)])
        if args.stochastic:
            eval_argv.append("--stochastic")
        if args.output is not None:
            eval_argv.extend(["--output", str(args.output)])
        return eval_main(eval_argv)
    if args.command == "rl-play":
        from .rl.play import play_rl

        summary = play_rl(
            args.model,
            env_id=args.env,
            frames_per_decision=args.frames_per_decision,
            deterministic=not args.stochastic,
            hybrid=args.hybrid,
            confidence_threshold=args.confidence_threshold,
            display=args.display,
            speed=args.speed,
            sound=args.sound,
            volume=args.volume,
        )
        print(
            f"RL play finished: clear={summary['clear']} dead={summary['dead']} "
            f"best_x={summary['best_x']} steps={summary['steps']} "
            f"reward={summary['episode_reward']:.2f}"
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
