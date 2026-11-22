from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO

from ..state import EnemyObservation, MarioSnapshot
from .actions import NUM_ACTIONS, RL_ACTIONS, index_from_action
from .observations import OBSERVATION_DIM, encode_observation


@dataclass(frozen=True)
class BCDataset:
    observations: np.ndarray
    actions: np.ndarray
    action_probs: np.ndarray
    confidences: np.ndarray
    source_files: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True)
class BCTrainResult:
    model_path: Path
    samples: int
    epochs: int
    final_train_loss: float
    final_train_accuracy: float
    val_accuracy: float | None


def _int_or(value: Any, default: int = 0) -> int:
    return int(value) if value is not None else default


def _opt_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def record_to_snapshot(record: Mapping[str, Any]) -> MarioSnapshot:
    """Reconstruct a MarioSnapshot from a Jev JSONL run record."""
    state = record.get("state", {})
    debug_state = record.get("debug_state", {})
    level = state.get("level", {})
    player = state.get("player", {})
    trajectory = state.get("trajectory", {})
    terrain = state.get("terrain", {})
    last_grounded = terrain.get("last_grounded_preview", {})
    reaction = state.get("reaction_timing", {})
    recent = state.get("recent_control", {})
    episode = state.get("episode", {})
    raw = debug_state.get("raw", {})

    enemies: list[EnemyObservation] = []
    for item in debug_state.get("visible_enemies", ()):
        enemies.append(
            EnemyObservation(
                slot=_int_or(item.get("slot"), 0),
                kind_id=_int_or(item.get("kind_id"), 0),
                kind=str(item.get("kind", "unknown")),
                dx_pixels=_int_or(item.get("relative_x_pixels"), 0),
                dy_pixels=_int_or(item.get("relative_y_pixels"), 0),
                relative_velocity_x=_int_or(item.get("relative_velocity_x"), 0),
            )
        )

    local_grid = tuple(str(row) for row in debug_state.get("local_grid", {}).get("rows", ()))
    grounded = bool(player.get("grounded", True))

    return MarioSnapshot(
        goal=str(state.get("objective", "Reach the flag in World 1-1 without dying.")),
        world=_int_or(level.get("world"), 1),
        stage=_int_or(level.get("stage"), 1),
        area=_int_or(level.get("area"), 1),
        x=_int_or(player.get("x"), 0),
        y=_int_or(player.get("y"), 0),
        dx=_int_or(player.get("horizontal_speed_px_per_frame"), 0),
        dy=_int_or(player.get("vertical_speed_px_per_frame"), 0),
        direction=str(raw.get("horizontal_motion", "nearly_stationary")),
        vertical_motion=str(raw.get("vertical_motion", "level_or_grounded")),
        airborne=not grounded,
        status=str(player.get("powerup_status", "small")),
        player_state=_int_or(raw.get("player_state"), 8),
        lives=_int_or(episode.get("lives"), 2),
        coins=_int_or(raw.get("coins"), 0),
        score=_int_or(raw.get("score"), 0),
        time_left=_int_or(episode.get("time_left"), 400),
        progress=_int_or(episode.get("progress"), _int_or(player.get("x"), 0)),
        best_progress=_int_or(episode.get("best_progress"), _int_or(player.get("x"), 0)),
        stalled_steps=_int_or(episode.get("stalled_frames"), 0),
        enemies=tuple(enemies),
        local_grid=local_grid,
        previous_action=recent.get("action"),
        previous_reward=float(raw.get("previous_reward", 0.0)),
        dead=bool(episode.get("dead", False)),
        clear=bool(episode.get("stage_clear", False)),
        grounded=grounded,
        jump_phase=str(player.get("jump_phase", "grounded")),
        action_frames=_int_or(recent.get("frames_observed"), 0),
        action_progress=_int_or(recent.get("progress_gained_pixels"), 0),
        airborne_frames=_int_or(trajectory.get("airborne_frames"), 0),
        jump_distance_pixels=_int_or(trajectory.get("horizontal_distance_since_takeoff_pixels"), 0),
        crossing_gap=bool(trajectory.get("crossing_known_gap", False)),
        gap_width_at_commit=_int_or(trajectory.get("gap_width_at_commit_tiles"), 0),
        decision_horizon_frames=_int_or(reaction.get("action_horizon_frames"), 8),
        last_response_delay_frames=_int_or(reaction.get("last_inference_delay_frames"), 0),
        last_grounded_gap_distance_tiles=_opt_int(last_grounded.get("gap_distance_tiles")),
        last_grounded_gap_width_tiles=_int_or(last_grounded.get("gap_width_tiles_visible"), 0),
        last_grounded_obstacle_distance_tiles=_opt_int(
            last_grounded.get("obstacle_distance_tiles")
        ),
        last_grounded_obstacle_height_tiles=_int_or(last_grounded.get("obstacle_height_tiles"), 0),
    )


def _expand_jsonl_paths(paths: Sequence[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            resolved.append(path)
    return resolved


def load_demonstrations(
    paths: Sequence[Path],
    *,
    min_confidence: float = 0.0,
) -> BCDataset:
    """Load Jev JSONL decision logs into numeric RL observation/action tensors."""
    files = _expand_jsonl_paths(paths)
    if not files:
        raise FileNotFoundError(f"No JSONL demonstration files found in: {paths}")

    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    prob_list: list[np.ndarray] = []
    conf_list: list[float] = []

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                confidence = float(record.get("confidence", 1.0))
                if confidence < min_confidence:
                    continue
                action_raw = record.get("action")
                if not isinstance(action_raw, str):
                    continue
                try:
                    action_idx = index_from_action(action_raw)
                except ValueError:
                    continue

                snapshot = record_to_snapshot(record)
                obs = encode_observation(snapshot)

                raw_probs = record.get("probabilities", {})
                probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
                if isinstance(raw_probs, Mapping) and raw_probs:
                    for idx, candidate in enumerate(RL_ACTIONS):
                        probs[idx] = max(0.0, float(raw_probs.get(candidate.value, 0.0)))
                prob_sum = float(probs.sum())
                if prob_sum > 0.0:
                    probs /= prob_sum
                else:
                    probs[action_idx] = 1.0

                obs_list.append(obs)
                action_list.append(action_idx)
                prob_list.append(probs)
                conf_list.append(confidence)

    if not obs_list:
        raise ValueError("No valid demonstration samples found after filtering.")

    return BCDataset(
        observations=np.stack(obs_list, axis=0).astype(np.float32),
        actions=np.asarray(action_list, dtype=np.int64),
        action_probs=np.stack(prob_list, axis=0).astype(np.float32),
        confidences=np.asarray(conf_list, dtype=np.float32),
        source_files=tuple(str(p) for p in files),
    )


class _DummyMarioSpaceEnv(gym.Env[np.ndarray, int]):
    """Lightweight space-only Gym env for initializing an SB3 PPO policy for BC."""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        return np.zeros(OBSERVATION_DIM, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        return np.zeros(OBSERVATION_DIM, dtype=np.float32), 0.0, True, False, {}


def train_bc(
    dataset: BCDataset,
    *,
    model_path: Path = Path("models/mario_bc.zip"),
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    soft_target_weight: float = 0.5,
    val_fraction: float = 0.15,
    net_arch: tuple[int, ...] = (256, 256),
    seed: int = 42,
    device: str = "cpu",
) -> BCTrainResult:
    """Pretrain a PPO-compatible MlpPolicy using Behavior Cloning on Jev logs."""
    if len(dataset) == 0:
        raise ValueError("dataset must not be empty")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    dummy_env = _DummyMarioSpaceEnv()
    model = PPO(
        policy="MlpPolicy",
        env=dummy_env,
        policy_kwargs={"net_arch": list(net_arch)},
        seed=seed,
        device=device,
        verbose=0,
    )
    policy = model.policy
    policy.set_training_mode(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    indices = rng.permutation(len(dataset))
    val_size = int(len(dataset) * val_fraction) if len(dataset) >= 10 else 0
    val_idx = indices[:val_size]
    train_idx = indices[val_size:] if val_size < len(dataset) else indices

    train_obs = torch.as_tensor(dataset.observations[train_idx], device=policy.device)
    train_act = torch.as_tensor(dataset.actions[train_idx], device=policy.device)
    train_probs = torch.as_tensor(dataset.action_probs[train_idx], device=policy.device)

    final_loss = 0.0
    final_acc = 0.0
    for _ in range(epochs):
        perm = torch.randperm(train_obs.shape[0], device=policy.device)
        epoch_loss = 0.0
        correct = 0
        total = 0

        for start in range(0, train_obs.shape[0], batch_size):
            batch_idx = perm[start : start + batch_size]
            obs_b = train_obs[batch_idx]
            act_b = train_act[batch_idx]
            probs_b = train_probs[batch_idx]

            dist = policy.get_distribution(obs_b)
            logits = dist.distribution.logits
            ce_loss = F.cross_entropy(logits, act_b)
            log_probs = F.log_softmax(logits, dim=-1)
            kl_loss = F.kl_div(log_probs, probs_b, reduction="batchmean")
            loss = (1.0 - soft_target_weight) * ce_loss + soft_target_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_n = int(obs_b.shape[0])
            epoch_loss += float(loss.item()) * batch_n
            preds = logits.argmax(dim=-1)
            correct += int((preds == act_b).sum().item())
            total += batch_n

        final_loss = epoch_loss / max(1, total)
        final_acc = correct / max(1, total)

    val_acc: float | None = None
    if len(val_idx) > 0:
        policy.set_training_mode(False)
        with torch.no_grad():
            val_obs = torch.as_tensor(dataset.observations[val_idx], device=policy.device)
            val_act = torch.as_tensor(dataset.actions[val_idx], device=policy.device)
            val_logits = policy.get_distribution(val_obs).distribution.logits
            val_acc = float((val_logits.argmax(dim=-1) == val_act).float().mean().item())

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    dummy_env.close()

    return BCTrainResult(
        model_path=model_path,
        samples=len(dataset),
        epochs=epochs,
        final_train_loss=final_loss,
        final_train_accuracy=final_acc,
        val_accuracy=val_acc,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain a Mario PPO policy via Behavior Cloning from Jev JSONL logs."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[Path("artifacts")],
        help="JSONL files or directories containing Jev run logs",
    )
    parser.add_argument("--model-path", type=Path, default=Path("models/mario_bc.zip"))
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--soft-target-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_demonstrations(args.inputs, min_confidence=args.min_confidence)
    result = train_bc(
        dataset,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        soft_target_weight=args.soft_target_weight,
        seed=args.seed,
        device=args.device,
    )
    val_str = f"{result.val_accuracy * 100:.1f}%" if result.val_accuracy is not None else "n/a"
    print(
        f"BC complete: samples={result.samples} epochs={result.epochs} "
        f"train_loss={result.final_train_loss:.4f} "
        f"train_acc={result.final_train_accuracy * 100:.1f}% val_acc={val_str} "
        f"saved={result.model_path.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
