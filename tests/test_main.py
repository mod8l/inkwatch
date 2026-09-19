"""Tests for `__main__._log_tick` (L1): which event lines a tick produces.
`_log_tick` is pure apart from the logger/`_save_frame` calls, so it's
driven here with a fake logger and `rectified=None` (no frame writes) --
the live loop around it still needs a real camera and stays untested.

The load-bearing case: the move that ends the game changes the board AND
enters GAME_OVER on the same tick, and must log BOTH a "commit" and a
"result" -- a "result"-only line means metrics.py's commit-based accuracy
never counts the winning move, capping every game at (N-1)/N.
"""

from __future__ import annotations

from inkwatch.__main__ import _log_tick
from inkwatch.metrics import GameLabel, TrueMove, score_game
from inkwatch.session import Phase, SessionResult

EMPTY: tuple = (None,) * 9


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def log(self, event_type: str, **fields) -> None:
        self.events.append({"type": event_type, **fields})


def result(phase: Phase, board: tuple, message: str | None = None) -> SessionResult:
    return SessionResult(
        phase=phase,
        board=board,
        turn="X",
        message=message,
        target_cell=None,
        confidence="accepted",
        cell_marks=None,
        escalation_cells=frozenset(),
    )


def tick(logger: FakeLogger, prev: SessionResult | None, cur: SessionResult, frame_ts: float = 1.0, tmp=None) -> SessionResult:
    _log_tick(logger, tmp, None, prev, cur, frame_ts)
    return cur


def types(logger: FakeLogger) -> list[str]:
    return [e["type"] for e in logger.events]


def test_first_tick_logs_start(tmp_path):
    logger = FakeLogger()
    tick(logger, None, result(Phase.WAIT_HUMAN, EMPTY), tmp=tmp_path)
    assert types(logger) == ["start"]


def test_quiet_tick_logs_nothing(tmp_path):
    logger = FakeLogger()
    prev = tick(logger, None, result(Phase.WAIT_HUMAN, EMPTY), tmp=tmp_path)
    tick(logger, prev, result(Phase.WAIT_HUMAN, EMPTY), frame_ts=2.0, tmp=tmp_path)
    assert types(logger) == ["start"]


def test_board_change_logs_commit(tmp_path):
    logger = FakeLogger()
    prev = tick(logger, None, result(Phase.WAIT_HUMAN, EMPTY), tmp=tmp_path)
    board = (None,) * 4 + ("X",) + (None,) * 4
    tick(logger, prev, result(Phase.WAIT_AGENT_INK, board), frame_ts=2.0, tmp=tmp_path)
    assert types(logger) == ["start", "commit"]


def test_entering_ask_human_logs_question(tmp_path):
    logger = FakeLogger()
    prev = tick(logger, None, result(Phase.WAIT_HUMAN, EMPTY), tmp=tmp_path)
    tick(logger, prev, result(Phase.ASK_HUMAN, EMPTY, "which one?"), frame_ts=2.0, tmp=tmp_path)
    assert types(logger) == ["start", "question"]


def test_the_terminal_move_logs_both_commit_and_result(tmp_path):
    logger = FakeLogger()
    prev = tick(logger, None, result(Phase.WAIT_HUMAN, EMPTY), tmp=tmp_path)
    final_board = ("O", "O", None, "X", "X", "X", None, None, None)
    tick(logger, prev, result(Phase.GAME_OVER, final_board, "I win."), frame_ts=2.0, tmp=tmp_path)
    assert types(logger) == ["start", "commit", "result"]


def test_a_full_games_events_score_every_move_including_the_winner(tmp_path):
    """Drives _log_tick through a 5-move X win and scores the events
    against ground truth: all five moves must be counted, or the README's
    accuracy metric is capped at 4/5 for a flawless game."""
    logger = FakeLogger()
    boards = [
        (None, None, None, None, "X", None, None, None, None),
        ("O", None, None, None, "X", None, None, None, None),
        ("O", None, None, None, "X", "X", None, None, None),
        ("O", "O", None, None, "X", "X", None, None, None),
        ("O", "O", None, "X", "X", "X", None, None, None),
    ]
    phases = [Phase.WAIT_AGENT_INK, Phase.WAIT_HUMAN, Phase.WAIT_AGENT_INK, Phase.WAIT_HUMAN, Phase.GAME_OVER]

    prev = tick(logger, None, result(Phase.WAIT_HUMAN, EMPTY), frame_ts=0.0, tmp=tmp_path)
    for i, (board, phase) in enumerate(zip(boards, phases)):
        prev = tick(logger, prev, result(phase, board), frame_ts=float(i + 1), tmp=tmp_path)

    label = GameLabel(
        session_id="test",
        true_moves=tuple(TrueMove(cell=c, hand_left_frame_ts=float(i + 1)) for i, c in enumerate([4, 0, 5, 1, 3])),
        false_trigger_commits=0,
        final_board=boards[-1],
    )
    score = score_game(logger.events, label)

    assert score.turns == 5
    assert score.matched_moves == 5
    assert score.total_true_moves == 5
    assert score.desynced is False
