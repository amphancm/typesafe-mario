from __future__ import annotations

from collections.abc import Sequence

from ..actions import ACTION_TO_INDEX, JUMP_ACTIONS, JUMP_RELEASE_ACTION, Action

# Canonical ordered 7-action space matching gym_super_mario_bros SIMPLE_MOVEMENT.
RL_ACTIONS: tuple[Action, ...] = tuple(
    action for action, _ in sorted(ACTION_TO_INDEX.items(), key=lambda item: item[1])
)
NUM_ACTIONS: int = len(RL_ACTIONS)


def action_from_index(index: int) -> Action:
    """Map an integer RL action index (0..6) to a Mario Action enum."""
    idx = int(index)
    if idx < 0 or idx >= NUM_ACTIONS:
        raise ValueError(f"Action index {idx} out of range [0, {NUM_ACTIONS - 1}]")
    return RL_ACTIONS[idx]


def index_from_action(action: Action | str) -> int:
    """Map an Action enum or its string value to an integer RL action index."""
    resolved = action if isinstance(action, Action) else Action(action)
    return ACTION_TO_INDEX[resolved]


def expand_macro_frames(
    action: Action,
    *,
    frames: int,
    previous_action: Action | None = None,
    grounded: bool = False,
) -> list[int]:
    """Expand a macro action into per-frame JoypadSpace indices.

    On the NES, holding the A button continuously across landing prevents Mario
    from starting a new jump. When a jump macro is selected while grounded after
    a previous jump macro, emit a single button-up release frame first so a fresh
    jump edge is triggered.
    """
    if frames < 1:
        raise ValueError("frames must be at least 1")

    action_index = ACTION_TO_INDEX[action]
    needs_release_edge = (
        grounded and action in JUMP_ACTIONS and previous_action in JUMP_ACTIONS and frames > 1
    )
    if not needs_release_edge:
        return [action_index] * frames

    release_index = ACTION_TO_INDEX[JUMP_RELEASE_ACTION[action]]
    return [release_index] + [action_index] * (frames - 1)


def action_names() -> Sequence[str]:
    """Return ordered human-readable names for the RL action space."""
    return tuple(action.value for action in RL_ACTIONS)
