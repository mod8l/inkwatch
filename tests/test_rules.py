"""Tests for the pure rules engine (G2)."""

from __future__ import annotations

import pytest

from inkwatch.rules import (
    EMPTY_BOARD,
    LINES,
    apply_move,
    is_draw,
    is_terminal,
    legal_moves,
    other,
    winner,
)


def test_empty_board_has_all_nine_cells_legal():
    assert legal_moves(EMPTY_BOARD) == list(range(9))


def test_apply_move_places_symbol_and_leaves_original_board_untouched():
    board = apply_move(EMPTY_BOARD, 4, "X")

    assert board[4] == "X"
    assert EMPTY_BOARD[4] is None
    assert legal_moves(board) == [0, 1, 2, 3, 5, 6, 7, 8]


def test_apply_move_rejects_an_occupied_cell():
    board = apply_move(EMPTY_BOARD, 0, "X")

    with pytest.raises(ValueError):
        apply_move(board, 0, "O")


def test_other_swaps_symbol():
    assert other("X") == "O"
    assert other("O") == "X"


@pytest.mark.parametrize("line", LINES)
def test_three_in_a_row_on_every_line_is_a_win_for_that_symbol(line):
    board = list(EMPTY_BOARD)
    for cell in line:
        board[cell] = "X"
    board = tuple(board)

    result = winner(board)

    assert result is not None
    symbol, won_line = result
    assert symbol == "X"
    assert won_line == line


def test_no_three_in_a_row_is_not_a_win():
    # X top row minus one cell, O elsewhere: no line completed.
    board = ("X", "X", None, "O", "O", None, None, None, None)

    assert winner(board) is None
    assert not is_terminal(board)


def test_full_board_with_no_winner_is_a_draw():
    board = ("X", "O", "X", "X", "O", "O", "O", "X", "X")

    assert winner(board) is None
    assert is_draw(board)
    assert is_terminal(board)


def test_full_board_with_a_winner_is_not_a_draw():
    board = ("X", "X", "X", "O", "O", "X", "X", "O", "O")

    assert winner(board) is not None
    assert not is_draw(board)
    assert is_terminal(board)


def test_partial_board_is_not_a_draw():
    board = ("X", None, None, None, None, None, None, None, None)

    assert not is_draw(board)
    assert not is_terminal(board)
