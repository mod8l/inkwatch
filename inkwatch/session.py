"""The turn state machine: the single owner of board, turn, and baseline
(G1, PRODUCT.md §8). Perception only ever hands this an `Observation`; only
this module commits a move or moves the baseline forward.

Scope note (M3, "playable loop... happy path only"): this implements the
happy-path spine of §8 — CALIBRATING, WAIT_HUMAN, THINK, WAIT_AGENT_INK,
COMMIT_*, GAME_OVER. `EVALUATE` and `THINK` have no external input of
their own (they're pure computation on the observation/board already in
hand), so they're folded into the same `update()` call rather than
persisted as their own resting phase — session never actually pauses
*in* them between frames. `BOARD_LOST`, `RESYNC`, `ESCALATE`, and
`ASK_HUMAN` (§9's edge cases, low-confidence handling) are M4/M5: for now,
anything that isn't a clean high-confidence read (D2) simply doesn't
commit and waits for a cleaner one, per the hard rule that the agent never
changes committed state on a guess. That's a real gap, not a silent one —
see NOTES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from inkwatch.events import CellMark, Observation
from inkwatch.decision import best_move
from inkwatch.output import cell_name, describe_line
from inkwatch.rules import EMPTY_BOARD, Board, Symbol, apply_move, is_draw, winner

DEFAULT_REMINDER_S: tuple[float, float] = (10.0, 20.0)


class Phase(str, Enum):
    CALIBRATING = "CALIBRATING"
    WAIT_HUMAN = "WAIT_HUMAN"
    EVALUATE = "EVALUATE"
    THINK = "THINK"
    WAIT_AGENT_INK = "WAIT_AGENT_INK"
    GAME_OVER = "GAME_OVER"
    # Not implemented before M4/M5; kept here so Phase matches PRODUCT.md §8
    # in full, even though update() never sets these yet.
    BOARD_LOST = "BOARD_LOST"
    RESYNC = "RESYNC"
    ESCALATE = "ESCALATE"
    ASK_HUMAN = "ASK_HUMAN"


@dataclass(frozen=True)
class SessionResult:
    """What `update()` hands back for `__main__.py` to speak and draw."""

    phase: Phase
    board: Board
    turn: Symbol
    message: str | None
    target_cell: int | None
    confidence: str  # "accepted" | "escalating" | "asking" (§6.5)
    cell_marks: tuple[CellMark, ...] | None


def _new_marks(cell_marks: tuple[CellMark, ...], board: Board) -> tuple[list[int], list[int], list[int]]:
    """Splits this frame's classifications into (newly-marked empty
    cells, ambiguous cells, occupied cells whose ink changed)."""
    marked, ambiguous, occupied_changed = [], [], []
    for i, mark in enumerate(cell_marks):
        if mark == "none":
            continue
        if board[i] is not None:
            occupied_changed.append(i)
        elif mark == "marked":
            marked.append(i)
        else:
            ambiguous.append(i)
    return marked, ambiguous, occupied_changed


class Session:
    """One game. `agent_first=True` has the agent play X and open (A2)."""

    def __init__(self, agent_first: bool = False, reminder_s: tuple[float, float] = DEFAULT_REMINDER_S) -> None:
        self.human_symbol: Symbol = "O" if agent_first else "X"
        self.agent_symbol: Symbol = "X" if agent_first else "O"
        self.turn: Symbol = "X"  # X always moves first, by convention (A2)

        self.phase = Phase.CALIBRATING
        self.board: Board = EMPTY_BOARD
        self.baseline: list[float] | None = None

        self.target_cell: int | None = None
        self._pending_cell: int | None = None

        self.reminder_s = reminder_s
        self._armed_at: float | None = None
        self._next_reminder_at: float | None = None
        self._agent_message: str | None = None  # repeated on reminder (O3)

    def update(self, observation: Observation, now: float) -> SessionResult:
        message = None

        if self.phase == Phase.WAIT_AGENT_INK:
            message = self._maybe_remind(now)

        if self.phase == Phase.CALIBRATING:
            message = self._handle_calibrating(observation, now) or message
        elif self.phase == Phase.WAIT_HUMAN and observation.stable and observation.cell_marks is not None:
            message = self._handle_human_turn(observation, now) or message
        elif self.phase == Phase.WAIT_AGENT_INK and observation.stable and observation.cell_marks is not None:
            message = self._handle_agent_ink(observation, now) or message

        return SessionResult(
            phase=self.phase,
            board=self.board,
            turn=self.turn,
            message=message,
            target_cell=self.target_cell,
            confidence="accepted",
            cell_marks=observation.cell_marks,
        )

    # -- CALIBRATING ---------------------------------------------------

    def _handle_calibrating(self, observation: Observation, now: float) -> str | None:
        if not (observation.found and observation.stable and observation.ratios is not None):
            return None
        self.baseline = list(observation.ratios)

        if self.turn == self.human_symbol:
            self.phase = Phase.WAIT_HUMAN
            return "I can see the board. You're X, you go first."

        self.phase = Phase.WAIT_AGENT_INK
        return f"I can see the board. I'm X, I'll go first. {self._announce_agent_move(now)}"

    # -- WAIT_HUMAN / EVALUATE ------------------------------------------

    def _handle_human_turn(self, observation: Observation, now: float) -> str | None:
        assert observation.cell_marks is not None and self.baseline is not None
        marked, ambiguous, occupied_changed = _new_marks(observation.cell_marks, self.board)

        if occupied_changed or ambiguous or len(marked) != 1:
            # D3: low confidence. M4 will escalate/ask; for now, wait for
            # a cleaner read rather than guess.
            self._pending_cell = None
            return None

        candidate = marked[0]
        if self._pending_cell != candidate:
            # D6: needs a second consecutive stable read agreeing.
            self._pending_cell = candidate
            return None

        self._pending_cell = None
        return self._commit(candidate, self.human_symbol, observation.ratios, now)

    # -- WAIT_AGENT_INK ---------------------------------------------------

    def _handle_agent_ink(self, observation: Observation, now: float) -> str | None:
        assert observation.cell_marks is not None and self.baseline is not None and self.target_cell is not None
        marked, ambiguous, occupied_changed = _new_marks(observation.cell_marks, self.board)

        if occupied_changed or ambiguous or marked != [self.target_cell]:
            # Anything but exactly the armed cell (nothing elsewhere) is
            # the M4 "wrong cell" edge case; for now, keep waiting.
            self._pending_cell = None
            return None

        if self._pending_cell != self.target_cell:
            self._pending_cell = self.target_cell
            return None

        self._pending_cell = None
        self._armed_at = None
        self._next_reminder_at = None
        return self._commit(self.target_cell, self.agent_symbol, observation.ratios, now)

    def _maybe_remind(self, now: float) -> str | None:
        if self._next_reminder_at is None or now < self._next_reminder_at:
            return None
        self._next_reminder_at = now + self.reminder_s[1]
        return self._agent_message

    # -- Commit + agent reply --------------------------------------------

    def _commit(self, cell: int, symbol: Symbol, ratios: tuple[float, ...] | None, now: float) -> str:
        """Apply a committed move (G1), move the baseline forward (D7),
        and either end the game or hand off the turn (G4)."""
        self.board = apply_move(self.board, cell, symbol)
        assert ratios is not None
        self.baseline = list(ratios)
        self.target_cell = None

        result = winner(self.board)
        if result is not None or is_draw(self.board):
            self.phase = Phase.GAME_OVER
            self._armed_at = None
            self._next_reminder_at = None
            return self._terminal_message(result)

        if symbol == self.human_symbol:
            return self._think_and_arm(cell, now)

        self.turn = self.human_symbol
        self.phase = Phase.WAIT_HUMAN
        return "Got it."

    def _think_and_arm(self, human_cell: int, now: float) -> str:
        """THINK, folded into the same beat as the human's commit (§6.2
        step 2-3: "immediately"), then arm the target cell and start the
        O3 reminder clock."""
        return f"You played {cell_name(human_cell)}. {self._announce_agent_move(now)}"

    def _announce_agent_move(self, now: float) -> str:
        """Picks the agent's move (THINK, G3), arms the target cell (G4),
        and starts the O3 reminder clock. Returns the phrase alone, so
        callers can prefix it with whatever led up to this turn."""
        move = best_move(self.board, self.agent_symbol)
        self.target_cell = move
        self.turn = self.agent_symbol
        self.phase = Phase.WAIT_AGENT_INK
        self._armed_at = now
        self._next_reminder_at = now + self.reminder_s[0]
        self._agent_message = f"I'll take {cell_name(move)}. Please draw an {self.agent_symbol} there."
        return self._agent_message

    def _terminal_message(self, result: tuple[Symbol, tuple[int, int, int]] | None) -> str:
        if result is None:
            return "It's a draw."
        symbol, line = result
        who = "You win" if symbol == self.human_symbol else "I win"
        return f"{who}, {describe_line(line)}."
