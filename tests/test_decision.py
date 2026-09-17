"""Tests for minimax move selection (G3).

`test_agent_as_*_never_loses` is the exhaustive proof CLAUDE.md's testing
section asks for: it plays out *every* possible opponent strategy (the
opponent tries every legal move at every one of its turns) against the
agent's `best_move`, from the empty board, and asserts the agent is never
on the losing side of a terminal board. Since every reachable board in
tic-tac-toe is reachable by some sequence of moves from the empty board,
this covers all of them, not just a sample.
"""

from __future__ import annotations

from inkwatch.decision import TIE_BREAK_ORDER, best_move
from inkwatch.rules import EMPTY_BOARD, apply_move, is_terminal, legal_moves, other, winner


def _play_out(board, agent, to_move):
    if is_terminal(board):
        result = winner(board)
        if result is not None:
            won_by, _ = result
            assert won_by == agent, f"agent ({agent}) lost from a reachable position: {board}"
        return

    if to_move == agent:
        chosen = best_move(board, agent)
        assert chosen in legal_moves(board)
        _play_out(apply_move(board, chosen, to_move), agent, other(to_move))
    else:
        for move in legal_moves(board):
            _play_out(apply_move(board, move, to_move), agent, other(to_move))


def test_agent_as_x_never_loses_against_any_opponent_play():
    _play_out(EMPTY_BOARD, agent="X", to_move="X")


def test_agent_as_o_never_loses_against_any_opponent_play():
    _play_out(EMPTY_BOARD, agent="O", to_move="X")


def test_opening_move_on_an_empty_board_is_the_center():
    assert best_move(EMPTY_BOARD, "X") == 4


def test_agent_takes_the_center_when_opponent_took_a_corner():
    board = apply_move(EMPTY_BOARD, 0, "X")

    assert best_move(board, "O") == 4


def test_agent_blocks_an_immediate_opponent_win():
    # X has top row minus the last cell; O must block cell 2 or lose next.
    board = ("X", "X", None, None, None, None, None, None, None)

    assert best_move(board, "O") == 2


def test_agent_takes_an_immediate_win_over_blocking():
    # O could block at 2, but O already has two in the middle column (1, 4)
    # and can win outright at 7.
    board = ("X", "O", "X", None, "O", None, None, None, None)

    assert best_move(board, "O") == 7


def test_tie_break_prefers_center_then_corners_then_edges():
    assert TIE_BREAK_ORDER[0] == 4
    assert set(TIE_BREAK_ORDER[1:5]) == {0, 2, 6, 8}
    assert set(TIE_BREAK_ORDER[5:]) == {1, 3, 5, 7}


def test_best_move_raises_on_a_terminal_board():
    board = ("X", "X", "X", "O", "O", None, None, None, None)
    try:
        best_move(board, "O")
    except ValueError:
        return
    raise AssertionError("expected best_move to raise on a terminal board")
