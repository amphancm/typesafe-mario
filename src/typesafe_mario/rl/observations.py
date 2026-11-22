from __future__ import annotations

import numpy as np

from ..state import MarioSnapshot
from .actions import NUM_ACTIONS, index_from_action

GRID_ROWS = 9
GRID_COLS = 11
GRID_CELLS = GRID_ROWS * GRID_COLS

JUMP_PHASES: tuple[str, ...] = ("grounded", "rising", "apex", "falling", "airborne")
POWERUP_STATUSES: tuple[str, ...] = ("small", "tall", "fireball")
MAX_UPCOMING_ENEMIES = 3

# 20 player/motion + 7 prev_action + 12 terrain + 6 threat summary + 12 enemies + 99 solid grid
SCALAR_FEATURE_DIM = 20 + NUM_ACTIONS + 12 + 6 + (MAX_UPCOMING_ENEMIES * 4)
OBSERVATION_DIM = SCALAR_FEATURE_DIM + GRID_CELLS


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return float(value)


def _norm_optional(value: float | None, scale: float, default: float = 1.0) -> float:
    if value is None:
        return default
    return _clip(float(value) / scale, 0.0, 1.0)


def encode_observation(snapshot: MarioSnapshot) -> np.ndarray:
    """Convert a MarioSnapshot into a bounded float32 vector in [-1.0, 1.0]."""
    features: list[float] = []

    # --- Player & motion (20) ---
    features.append(_clip(snapshot.x / 3200.0, 0.0, 1.0))
    features.append(_clip(snapshot.y / 255.0, 0.0, 1.0))
    features.append(_clip(snapshot.dx / 16.0, -1.0, 1.0))
    features.append(_clip(snapshot.dy / 16.0, -1.0, 1.0))
    features.append(1.0 if snapshot.grounded else 0.0)
    features.append(1.0 if snapshot.airborne else 0.0)

    for phase in JUMP_PHASES:
        features.append(1.0 if snapshot.jump_phase == phase else 0.0)

    for status in POWERUP_STATUSES:
        features.append(1.0 if snapshot.status == status else 0.0)

    features.append(_clip(snapshot.airborne_frames / 45.0, 0.0, 1.0))
    features.append(_clip(snapshot.jump_distance_pixels / 128.0, 0.0, 1.0))
    features.append(1.0 if snapshot.crossing_gap else 0.0)
    features.append(_clip(snapshot.gap_width_at_commit / 6.0, 0.0, 1.0))
    features.append(_clip(snapshot.stalled_steps / 30.0, 0.0, 1.0))
    features.append(_clip(snapshot.time_left / 400.0, 0.0, 1.0))

    # --- Previous action one-hot (7) ---
    prev_idx = -1
    if snapshot.previous_action is not None:
        try:
            prev_idx = index_from_action(snapshot.previous_action)
        except ValueError:
            prev_idx = -1
    for idx in range(NUM_ACTIONS):
        features.append(1.0 if idx == prev_idx else 0.0)

    # --- Terrain / navigation features (12) ---
    nav = snapshot.navigation_features()
    geom_ok = bool(nav.get("geometry_available", False))
    features.append(1.0 if geom_ok else 0.0)
    features.append(1.0 if nav.get("obstacle_ahead", False) else 0.0)
    features.append(_norm_optional(nav.get("obstacle_distance_tiles"), 8.0, default=1.0))
    features.append(_clip(float(nav.get("obstacle_height_tiles", 0)) / 6.0, 0.0, 1.0))
    features.append(1.0 if nav.get("gap_ahead", False) else 0.0)
    features.append(_norm_optional(nav.get("gap_distance_tiles"), 8.0, default=1.0))
    features.append(_clip(float(nav.get("gap_width_tiles_visible", 0)) / 6.0, 0.0, 1.0))
    features.append(_clip(float(nav.get("clear_forward_tiles", 0)) / 8.0, 0.0, 1.0))
    features.append(_norm_optional(snapshot.last_grounded_gap_distance_tiles, 8.0, default=1.0))
    features.append(_clip(snapshot.last_grounded_gap_width_tiles / 6.0, 0.0, 1.0))
    features.append(
        _norm_optional(snapshot.last_grounded_obstacle_distance_tiles, 8.0, default=1.0)
    )
    features.append(_clip(snapshot.last_grounded_obstacle_height_tiles / 6.0, 0.0, 1.0))

    # --- Threat summary (6) ---
    threat = snapshot.threat_features()
    features.append(1.0 if threat.get("enemy_ahead", False) else 0.0)
    features.append(1.0 if threat.get("jump_must_start_this_decision", False) else 0.0)
    features.append(1.0 if threat.get("contact_within_reaction_horizon", False) else 0.0)
    features.append(1.0 if threat.get("takeoff_window_already_missed", False) else 0.0)
    features.append(_clip(float(threat.get("closing_speed_pixels_per_frame", 0)) / 12.0, 0.0, 1.0))
    features.append(_norm_optional(threat.get("estimated_contact_frames"), 60.0, default=1.0))

    # --- Up to 3 upcoming enemies ahead (12) ---
    upcoming = threat.get("upcoming_enemies", ())
    for slot in range(MAX_UPCOMING_ENEMIES):
        if slot < len(upcoming):
            enemy = upcoming[slot]
            features.append(1.0)
            features.append(_clip(float(enemy["distance_pixels"]) / 256.0, -1.0, 1.0))
            features.append(_clip(float(enemy["vertical_offset_pixels"]) / 128.0, -1.0, 1.0))
            features.append(_clip(float(enemy["relative_velocity_x"]) / 12.0, -1.0, 1.0))
        else:
            features.extend((0.0, 1.0, 0.0, 0.0))

    # --- Local solid tile grid 9x11 (99) ---
    grid = snapshot.local_grid
    for row_idx in range(GRID_ROWS):
        row = grid[row_idx] if row_idx < len(grid) else ""
        for col_idx in range(GRID_COLS):
            ch = row[col_idx] if col_idx < len(row) else "."
            features.append(1.0 if ch == "#" else 0.0)

    obs = np.asarray(features, dtype=np.float32)
    if obs.shape != (OBSERVATION_DIM,):
        raise RuntimeError(f"Expected observation dim {OBSERVATION_DIM}, got {obs.shape}")
    return np.clip(np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
