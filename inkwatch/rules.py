"""Pure tic-tac-toe rules: board representation, legal moves, terminal
detection (G2). No OpenCV, no I/O, no imports from other `inkwatch`
modules — the one shared source of truth for what's legal, won, or drawn,
used by both `session.py` (applying committed moves) and `decision.py`
(walking the game tree).
"""

from __future__ import annotations

from typing import Literal

Symbol = Literal["X", "O"]
Board = tuple[Symbol | None, ...]  # 9 cells, row-major, None = empty

EMPTY_BOARD: Board = (None,) * 9

LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)


def other(symbol: Symbol) -> Symbol:
    return "O" if symbol == "X" else "X"


def legal_moves(board: Board) -> list[int]:
    return [i for i, cell in enumerate(board) if cell is None]


def apply_move(board: Board, cell: int, symbol: Symbol) -> Board:
    if board[cell] is not None:
        raise ValueError(f"cell {cell} is already occupied")
    return board[:cell] + (symbol,) + board[cell + 1 :]


def winner(board: Board) -> tuple[Symbol, tuple[int, int, int]] | None:
    """The winning symbol and line, or None if nobody has won (yet)."""
    for line in LINES:
        a, b, c = line
        if board[a] is not None and board[a] == board[b] == board[c]:
            return board[a], line
    return None


def is_draw(board: Board) -> bool:
    return winner(board) is None and all(cell is not None for cell in board)


def is_terminal(board: Board) -> bool:
    return winner(board) is not None or is_draw(board)
