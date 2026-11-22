from __future__ import annotations

import pytest

from typesafe_mario.actions import ACTION_TO_INDEX, Action
from typesafe_mario.rl.actions import (
    NUM_ACTIONS,
    RL_ACTIONS,
    action_from_index,
    action_names,
    expand_macro_frames,
    index_from_action,
)


def test_action_space_has_seven_canonical_actions() -> None:
    assert NUM_ACTIONS == 7
    assert len(RL_ACTIONS) == 7
    assert action_names() == (
        "noop",
        "right",
        "right_jump",
        "right_run",
        "right_run_jump",
        "jump",
        "left",
    )


def test_index_action_round_trip() -> None:
    for idx, action in enumerate(RL_ACTIONS):
        assert action_from_index(idx) == action
        assert index_from_action(action) == idx
        assert index_from_action(action.value) == idx


def test_action_from_index_rejects_out_of_bounds() -> None:
    with pytest.raises(ValueError):
        action_from_index(-1)
    with pytest.raises(ValueError):
        action_from_index(NUM_ACTIONS)


def test_expand_macro_frames_emits_release_edge_when_grounded_after_jump() -> None:
    frames = expand_macro_frames(
        Action.RIGHT_RUN_JUMP,
        frames=4,
        previous_action=Action.RIGHT_RUN_JUMP,
        grounded=True,
    )
    assert frames == [
        ACTION_TO_INDEX[Action.RIGHT_RUN],
        ACTION_TO_INDEX[Action.RIGHT_RUN_JUMP],
        ACTION_TO_INDEX[Action.RIGHT_RUN_JUMP],
        ACTION_TO_INDEX[Action.RIGHT_RUN_JUMP],
    ]


def test_expand_macro_frames_holds_jump_while_airborne() -> None:
    frames = expand_macro_frames(
        Action.RIGHT_RUN_JUMP,
        frames=4,
        previous_action=Action.RIGHT_RUN_JUMP,
        grounded=False,
    )
    assert frames == [ACTION_TO_INDEX[Action.RIGHT_RUN_JUMP]] * 4


def test_expand_macro_frames_no_release_for_non_jump() -> None:
    frames = expand_macro_frames(
        Action.RIGHT_RUN,
        frames=4,
        previous_action=Action.RIGHT_RUN_JUMP,
        grounded=True,
    )
    assert frames == [ACTION_TO_INDEX[Action.RIGHT_RUN]] * 4


def test_expand_macro_frames_rejects_zero_frames() -> None:
    with pytest.raises(ValueError):
        expand_macro_frames(Action.RIGHT, frames=0)
