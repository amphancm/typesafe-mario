from __future__ import annotations

from types import SimpleNamespace
import unittest

from typesafe_mario.actions import JUMP_RELEASE_ACTION, Action
from typesafe_mario.dashboard import (
    DashboardCommand,
    DashboardTheme,
    _dashboard_status,
    _load_font,
    clamp01,
)


class DashboardTests(unittest.TestCase):
    def test_clamp01_handles_missing_and_out_of_range_values(self) -> None:
        self.assertEqual(clamp01(None), 0.0)
        self.assertEqual(clamp01(-0.2), 0.0)
        self.assertEqual(clamp01(0.42), 0.42)
        self.assertEqual(clamp01(1.8), 1.0)

    def test_jump_macros_have_non_jump_release_frames(self) -> None:
        self.assertEqual(JUMP_RELEASE_ACTION[Action.RIGHT_JUMP], Action.RIGHT)
        self.assertEqual(JUMP_RELEASE_ACTION[Action.RIGHT_RUN_JUMP], Action.RIGHT_RUN)
        self.assertEqual(JUMP_RELEASE_ACTION[Action.JUMP], Action.NOOP)

    def test_dashboard_commands_are_stable_strings(self) -> None:
        self.assertEqual(DashboardCommand.RESTART, "restart")
        self.assertEqual(DashboardCommand.QUIT, "quit")

    def test_load_font_uses_first_matched_candidate_or_default(self) -> None:
        calls: list[tuple[str, bool]] = []

        def match_font(name: str, *, bold: bool = False) -> str | None:
            calls.append((name, bold))
            return "/fonts/NotoSans-Bold.ttf" if name == "Noto Sans" else None

        fake_pg = SimpleNamespace(
            font=SimpleNamespace(
                match_font=match_font,
                Font=lambda path, size: (path, size),
            )
        )

        font = _load_font(fake_pg, ("Segoe UI Semibold", "Noto Sans", "Arial"), 25, bold=True)
        self.assertEqual(font, ("/fonts/NotoSans-Bold.ttf", 25))
        self.assertEqual(calls, [("Segoe UI Semibold", True), ("Noto Sans", True)])

        fallback = _load_font(fake_pg, ("Missing Font",), 14)
        self.assertEqual(fallback, (None, 14))

    def test_dashboard_status_distinguishes_clear_finish_and_death(self) -> None:
        theme = DashboardTheme()
        clear_snap = SimpleNamespace(clear=True, dead=False)
        dead_snap = SimpleNamespace(clear=False, dead=True)
        normal_snap = SimpleNamespace(clear=False, dead=False)

        self.assertEqual(
            _dashboard_status(theme, snapshot=clear_snap, waiting=False, run_ended=True, finishing=True),
            ("Stage clear — finishing", theme.success),
        )
        self.assertEqual(
            _dashboard_status(theme, snapshot=clear_snap, waiting=False, run_ended=True, finishing=False),
            ("Stage clear!", theme.success),
        )
        self.assertEqual(
            _dashboard_status(theme, snapshot=dead_snap, waiting=False, run_ended=True),
            ("Mario died", theme.danger),
        )
        self.assertEqual(
            _dashboard_status(theme, snapshot=normal_snap, waiting=False, run_ended=True),
            ("Run ended", theme.danger),
        )
        self.assertEqual(
            _dashboard_status(theme, snapshot=normal_snap, waiting=False, run_ended=False),
            ("Live decision loop", theme.accent),
        )


if __name__ == "__main__":
    unittest.main()
