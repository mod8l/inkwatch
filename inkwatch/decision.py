"""Move selection via minimax with alpha-beta pruning (G3).

No OpenCV, no I/O. Imports `rules.py` (one-way; `rules.py` imports
nothing back) for legal moves, applying a move, and terminal/winner
checks at every node of the game tree — `rules.py` is this module's only
allowed `inkwatch` import (Gad's call on CLAUDE.md's hard rule:
reimplementing rules here too would risk the two disagreeing about what's
legal).

Perfect play; ties are broken by a fixed preference order (center,
corners, edges) so the agent's choice is deterministic and testable.
"""

from __future__ import annotations

from inkwatch.rules import Board, Symbol, apply_move, is_terminal, legal_moves, other, winner

TIE_BREAK_ORDER: tuple[int, ...] = (4, 0, 2, 6, 8, 1, 3, 5, 7)  # center, corners, edges


def _score(board: Board, agent: Symbol, depth: int) -> int:
    """Terminal score from `agent`'s point of view. Prefers a faster win
    and a slower loss (via `depth`) so minimax doesn't stall or throw a
    winning game away when several lines lead to the same outcome."""
    result = winner(board)
    if result is None:
        return 0
    won_by, _ = result
    return (10 - depth) if won_by == agent else depth - 10


def _minimax(board: Board, to_move: Symbol, agent: Symbol, depth: int, alpha: int, beta: int) -> int:
    if is_terminal(board):
        return _score(board, agent, depth)

    maximizing = to_move == agent
    best = -100 if maximizing else 100
    for cell in legal_moves(board):
        value = _minimax(apply_move(board, cell, to_move), other(to_move), agent, depth + 1, alpha, beta)
        if maximizing:
            best = max(best, value)
            alpha = max(alpha, best)
        else:
            best = min(best, value)
            beta = min(beta, best)
        if beta <= alpha:
            break
    return best


def best_move(board: Board, agent: Symbol) -> int:
    """The best cell for `agent` to play now, breaking ties by
    `TIE_BREAK_ORDER`. Raises if `board` is already terminal."""
    if is_terminal(board):
        raise ValueError("no move to make: board is already terminal")
    moves = legal_moves(board)

    scores = {
        cell: _minimax(apply_move(board, cell, agent), other(agent), agent, 1, -100, 100) for cell in moves
    }
    best_score = max(scores.values())
    best_cells = {cell for cell, score in scores.items() if score == best_score}
    for cell in TIE_BREAK_ORDER:
        if cell in best_cells:
            return cell
    return next(iter(best_cells))  # unreachable for a 3x3 board, kept for safety
