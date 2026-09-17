"""The turn state machine: the single owner of board, turn, and baseline
(G1, PRODUCT.md §8). Perception only ever hands this an `Observation`; only
this module commits a move or moves the baseline forward.

M4 scope ("recovery... all edge cases in §9 behave as specified"): this
implements every §8 phase, including `BOARD_LOST`, `RESYNC`, `ESCALATE`,
and `ASK_HUMAN`. `EVALUATE` and `THINK` still have no external input of
their own, so (per M3) they stay folded into the same `update()` call
rather than persisted as their own resting phase.

`ESCALATE` has no vision model wired in yet (that's M5's `escalation.py`)
— it always falls straight through to `ASK_HUMAN` on the next `update()`
call, which is exactly the §9 behavior specified for "vision model timeout
or no API key". When M5 adds a real model call, it plugs into
`_resolve_escalate` and only takes the fallback path when the model
disagrees or times out.

Two deliberate divergences from a literal reading of §8, flagged here per
CLAUDE.md ("stop and say so") for Gad's milestone review rather than
silently changed in PRODUCT.md:
- §8 only draws `RESYNC --> WAIT_HUMAN`. If the board was lost mid
  `WAIT_AGENT_INK` (the agent's move already armed), resuming into
  `WAIT_HUMAN` would silently drop the armed turn and wait on the wrong
  side. `_resume_after_resync` instead resumes whichever phase matches
  whose turn it already was, re-arming the same target cell rather than
  picking a new one.
- §8 doesn't draw `CALIBRATING --> BOARD_LOST` as ever firing before the
  first board is found — there's no committed state to lose yet, and
  `__main__.py`'s "board not found" banner already covers that wait. This
  build treats CALIBRATING as immune to BOARD_LOST; it just keeps waiting,
  same as M3.

Also cut from M4, tracked in NOTES.md rather than built: keyboard `y`/`n`
as the "last resort" answer channel `ARCHITECTURE.md` describes. Every
`ASK_HUMAN` case here resolves through the page instead (D8's primary
channel) — the human fixes what's on paper and the next stable read
either confirms or clears it. `y`/`n` doesn't map cleanly onto "which of
two cells is your move," so building it well is its own scoped piece of
work, not a few extra lines here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from inkwatch.events import CellMark, Observation
from inkwatch.decision import best_move
from inkwatch.output import cell_name, describe_line
from inkwatch.rules import EMPTY_BOARD, Board, Symbol, apply_move, is_draw, winner

DEFAULT_REMINDER_S: tuple[float, float] = (10.0, 20.0)
DEFAULT_OCCLUSION_REMINDER_S = 15.0  # §9: "Hand stays over the page... after 15 s"
DEFAULT_INK_LOW = 0.02   # duplicates perception.py's default; kept a literal here so
DEFAULT_INK_HIGH = 0.05  # this module doesn't need a module-level import of perception.py

# How many consecutive stable reads of the same lone ambiguous cell before
# it counts as "persistent" (shadow/glare/faint pen, §9) rather than a
# pen mid-stroke (D6, which two reads already covers). Not in PRODUCT.md
# §12 — no camera-verified value to tune yet; see NOTES.md.
AMBIGUOUS_ESCALATE_READS = 3


class Phase(str, Enum):
    CALIBRATING = "CALIBRATING"
    WAIT_HUMAN = "WAIT_HUMAN"
    EVALUATE = "EVALUATE"
    THINK = "THINK"
    WAIT_AGENT_INK = "WAIT_AGENT_INK"
    GAME_OVER = "GAME_OVER"
    BOARD_LOST = "BOARD_LOST"
    RESYNC = "RESYNC"
    ESCALATE = "ESCALATE"
    ASK_HUMAN = "ASK_HUMAN"


_CONFIDENCE_BY_PHASE = {
    Phase.ESCALATE: "escalating",
    Phase.ASK_HUMAN: "asking",
}


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
    cells, ambiguous cells, occupied cells with more ink than baseline)."""
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

    def __init__(
        self,
        agent_first: bool = False,
        reminder_s: tuple[float, float] = DEFAULT_REMINDER_S,
        occlusion_reminder_s: float = DEFAULT_OCCLUSION_REMINDER_S,
        ink_low: float = DEFAULT_INK_LOW,
        ink_high: float = DEFAULT_INK_HIGH,
    ) -> None:
        self.human_symbol: Symbol = "O" if agent_first else "X"
        self.agent_symbol: Symbol = "X" if agent_first else "O"
        self.turn: Symbol = "X"  # X always moves first, by convention (A2)

        self.phase = Phase.CALIBRATING
        self.board: Board = EMPTY_BOARD
        self.baseline: list[float] | None = None
        # Captured once, when CALIBRATING first finds the (blank) board;
        # never moved forward like `baseline` is. RESYNC uses this instead
        # of `baseline`, which §8 calls "no longer trustworthy" once the
        # page has been lost or bumped.
        self._blank_baseline: list[float] | None = None

        self.target_cell: int | None = None
        self._pending_cell: int | None = None

        self.reminder_s = reminder_s
        self._armed_at: float | None = None
        self._next_reminder_at: float | None = None
        self._agent_message: str | None = None  # repeated on reminder (O3)

        self._occlusion_reminder_s = occlusion_reminder_s
        self._occlusion_since: float | None = None
        self._next_occlusion_reminder_at: float | None = None

        self._ink_low = ink_low
        self._ink_high = ink_high

        # What an ASK_HUMAN/ESCALATE phase is currently about, so a later
        # update() knows how to interpret the next observation and word
        # the question: "two_marks" | "ambiguous" | "wrong_cell" | "resync".
        self._ask_context: str | None = None
        self._ask_cells: frozenset[int] = frozenset()

        # Debounce state (D6-style: two consecutive matching stable reads)
        # for each recovery check, kept separate so an ambiguous flicker on
        # one doesn't reset the count on another.
        self._two_marks_pending: frozenset[int] | None = None
        self._ambiguous_pending: int | None = None
        self._ambiguous_streak = 0
        self._wrong_cell_pending: int | None = None
        self._occupied_pending: tuple[frozenset[int], frozenset[int]] | None = None
        self._occupied_warned: tuple[frozenset[int], frozenset[int]] | None = None

    def update(self, observation: Observation, now: float) -> SessionResult:
        if self.phase == Phase.CALIBRATING:
            return self._result(observation, self._handle_calibrating(observation, now))

        if self.phase == Phase.GAME_OVER:
            return self._result(observation, None)

        # P2's hold-over already covers a brief marker dropout at the
        # perception layer; once it reports not-found, the board is really
        # gone regardless of what this phase was doing (§8: BOARD_LOST).
        if not observation.found and self.phase != Phase.BOARD_LOST:
            return self._result(observation, self._enter_board_lost())

        if self.phase == Phase.BOARD_LOST:
            message = self._enter_resync() if observation.found else None
            return self._result(observation, message)

        if self.phase == Phase.RESYNC:
            message = None
            if observation.stable and observation.ratios is not None:
                message = self._handle_resync(observation, now)
            return self._result(observation, message)

        if self.phase == Phase.ESCALATE:
            return self._result(observation, self._resolve_escalate())

        message = None
        if self.phase == Phase.WAIT_AGENT_INK:
            message = self._maybe_remind(now)
        elif self.phase == Phase.WAIT_HUMAN:
            message = self._maybe_remind_occlusion(observation, now)

        if observation.stable and observation.cell_marks is not None:
            if self.phase == Phase.WAIT_HUMAN:
                message = self._handle_human_turn(observation, now) or message
            elif self.phase == Phase.WAIT_AGENT_INK:
                message = self._handle_agent_ink(observation, now) or message
            elif self.phase == Phase.ASK_HUMAN:
                message = self._handle_ask_human(observation, now) or message

        return self._result(observation, message)

    def force_resync(self) -> None:
        """README's `r` key: manually trigger the same full-board re-read
        BOARD_LOST normally leads to, without waiting for the markers to
        actually drop out first."""
        if self.phase not in (Phase.CALIBRATING, Phase.GAME_OVER):
            self._enter_resync()
            self._ask_context = None
            self._reset_recovery_debounce()

    def _result(self, observation: Observation, message: str | None) -> SessionResult:
        return SessionResult(
            phase=self.phase,
            board=self.board,
            turn=self.turn,
            message=message,
            target_cell=self.target_cell,
            confidence=_CONFIDENCE_BY_PHASE.get(self.phase, "accepted"),
            cell_marks=observation.cell_marks,
        )

    # -- CALIBRATING ---------------------------------------------------

    def _handle_calibrating(self, observation: Observation, now: float) -> str | None:
        if not (observation.found and observation.stable and observation.ratios is not None):
            return None
        self.baseline = list(observation.ratios)
        self._blank_baseline = list(observation.ratios)

        if self.turn == self.human_symbol:
            self.phase = Phase.WAIT_HUMAN
            return "I can see the board. You're X, you go first."

        self.phase = Phase.WAIT_AGENT_INK
        return f"I can see the board. I'm X, I'll go first. {self._announce_agent_move(now)}"

    # -- BOARD_LOST / RESYNC --------------------------------------------

    def _reset_recovery_debounce(self) -> None:
        """Clears every recovery check's debounce/streak state. Called
        whenever a commit or a board-lost transition makes it stale —
        without this, a leftover candidate from an abandoned check could
        (very unlikely, but possible) coincidentally match a later one in
        a different phase and skip its debounce."""
        self._pending_cell = None
        self._two_marks_pending = None
        self._ambiguous_pending = None
        self._ambiguous_streak = 0
        self._wrong_cell_pending = None
        self._occupied_pending = None
        self._occupied_warned = None

    def _enter_board_lost(self) -> None:
        """Silent (§9 gives no spoken line for a bumped/lost page —
        `__main__.py` speaks its own line for the camera-disconnected
        case, which routes through here the same way)."""
        self.phase = Phase.BOARD_LOST
        self._armed_at = None
        self._next_reminder_at = None
        self._occlusion_since = None
        self._next_occlusion_reminder_at = None
        self._ask_context = None
        self._reset_recovery_debounce()
        return None

    def _enter_resync(self) -> None:
        self.phase = Phase.RESYNC
        return None

    def _handle_resync(self, observation: Observation, now: float) -> str | None:
        """Re-reads all nine cells against the untouched blank baseline
        (`baseline` "is no longer trustworthy", §8) and compares ink
        presence to what `session.board` believes is there."""
        assert observation.ratios is not None
        mismatches = self._resync_mismatches(observation.ratios)
        if not mismatches:
            self.baseline = list(observation.ratios)
            return self._resume_after_resync(now)

        self.phase = Phase.ASK_HUMAN
        self._ask_context = "resync"
        names = ", ".join(cell_name(c) for c in mismatches)
        return f"The page doesn't match what I have. Please check {names}."

    def _resync_mismatches(self, ratios: tuple[float, ...]) -> list[int]:
        # Local import: this module already pulls in cv2 transitively via
        # output.py, so this isn't avoiding that — it's keeping perception's
        # larger surface out of session.py's own import list, borrowing
        # only the one pure (no cv2, no I/O) classification function.
        from inkwatch.perception import classify_cells

        assert self._blank_baseline is not None
        marks = classify_cells(list(ratios), self._blank_baseline, self._ink_low, self._ink_high)
        return [i for i, mark in enumerate(marks) if (mark != "none") != (self.board[i] is not None)]

    def _resume_after_resync(self, now: float) -> str:
        self._ask_context = None
        if self.turn == self.human_symbol:
            self.phase = Phase.WAIT_HUMAN
            return "Okay, I can see the board again."

        self.phase = Phase.WAIT_AGENT_INK
        self._armed_at = now
        self._next_reminder_at = now + self.reminder_s[0]
        assert self._agent_message is not None
        return f"Okay, I can see the board again. {self._agent_message}"

    # -- ESCALATE ---------------------------------------------------------

    def _enter_escalate(self, situation: str, cells: frozenset[int]) -> None:
        self.phase = Phase.ESCALATE
        self._ask_context = situation
        self._ask_cells = cells
        self._pending_cell = None
        return None

    def _resolve_escalate(self) -> str:
        """M5 will try a vision-model read here first. With no model
        wired in yet, every escalation skips straight to asking — §9's
        specified behavior for "vision model timeout or no API key"."""
        self.phase = Phase.ASK_HUMAN
        if self._ask_context == "two_marks":
            return "I see two new marks. Which one is your move?"
        cell = next(iter(self._ask_cells))
        return f"I can't tell if you've drawn in {cell_name(cell)}. Can you check the light or the page?"

    # -- WAIT_HUMAN / EVALUATE ------------------------------------------

    def _handle_human_turn(self, observation: Observation, now: float) -> str | None:
        assert observation.cell_marks is not None and self.baseline is not None and observation.ratios is not None
        marked, ambiguous, occupied_changed = _new_marks(observation.cell_marks, self.board)
        erased = self._erased_cells(observation.ratios)

        if erased or occupied_changed:
            return self._check_occupied_drift(erased, occupied_changed)
        self._occupied_pending = None
        self._occupied_warned = None

        if len(marked) >= 2:
            return self._check_two_marks(marked)
        self._two_marks_pending = None

        if len(marked) == 1 and not ambiguous:
            self._ambiguous_pending = None
            self._ambiguous_streak = 0
            candidate = marked[0]
            if self._pending_cell != candidate:
                # D6: needs a second consecutive stable read agreeing.
                self._pending_cell = candidate
                return None
            self._pending_cell = None
            return self._commit(candidate, self.human_symbol, observation.ratios, now)
        self._pending_cell = None

        if len(ambiguous) == 1 and not marked:
            return self._check_ambiguous(ambiguous[0])
        self._ambiguous_pending = None
        self._ambiguous_streak = 0
        return None

    # -- WAIT_AGENT_INK ---------------------------------------------------

    def _handle_agent_ink(self, observation: Observation, now: float) -> str | None:
        assert (
            observation.cell_marks is not None
            and self.baseline is not None
            and self.target_cell is not None
            and observation.ratios is not None
        )
        marked, ambiguous, occupied_changed = _new_marks(observation.cell_marks, self.board)
        erased = self._erased_cells(observation.ratios)

        if erased or occupied_changed:
            return self._check_occupied_drift(erased, occupied_changed)
        self._occupied_pending = None
        self._occupied_warned = None

        if marked == [self.target_cell] and not ambiguous:
            self._wrong_cell_pending = None
            if self._pending_cell != self.target_cell:
                self._pending_cell = self.target_cell
                return None
            self._pending_cell = None
            self._armed_at = None
            self._next_reminder_at = None
            return self._commit(self.target_cell, self.agent_symbol, observation.ratios, now)
        self._pending_cell = None

        if len(marked) == 1 and marked[0] != self.target_cell and not ambiguous:
            return self._check_wrong_cell(marked[0])
        self._wrong_cell_pending = None
        return None

    def _maybe_remind(self, now: float) -> str | None:
        if self._next_reminder_at is None or now < self._next_reminder_at:
            return None
        self._next_reminder_at = now + self.reminder_s[1]
        return self._agent_message

    def _maybe_remind_occlusion(self, observation: Observation, now: float) -> str | None:
        """§9: "Hand stays over the page... waits silently; after 15 s"
        reminds. Scoped to WAIT_HUMAN — WAIT_AGENT_INK already has its own
        O3 reminder clock covering the same "waiting silently" need."""
        if observation.stable:
            self._occlusion_since = None
            self._next_occlusion_reminder_at = None
            return None
        if self._occlusion_since is None:
            self._occlusion_since = now
            self._next_occlusion_reminder_at = now + self._occlusion_reminder_s
            return None
        if self._next_occlusion_reminder_at is not None and now >= self._next_occlusion_reminder_at:
            self._next_occlusion_reminder_at = now + self._occlusion_reminder_s
            return "Take your time. Move your hand away when you're done."
        return None

    # -- Recovery checks shared by WAIT_HUMAN / WAIT_AGENT_INK -----------

    def _erased_cells(self, ratios: tuple[float, ...]) -> list[int]:
        """Occupied cells whose ink dropped well below baseline — a mark
        that seems to have been erased. classify_cell only ever reports
        "none" for a negative delta, so this can't come from cell_marks;
        it needs the raw ratios session already has."""
        assert self.baseline is not None
        return [i for i in range(9) if self.board[i] is not None and self.baseline[i] - ratios[i] >= self._ink_low]

    def _check_occupied_drift(self, erased: list[int], occupied_changed: list[int]) -> str | None:
        candidate = (frozenset(erased), frozenset(occupied_changed))
        if candidate != self._occupied_pending:
            self._occupied_pending = candidate
            self._pending_cell = None
            return None
        self._occupied_pending = None
        if candidate == self._occupied_warned:
            return None  # already said it; don't repeat every frame while it persists
        self._occupied_warned = candidate

        if erased:
            return f"A mark seems to have disappeared from {cell_name(erased[0])}."
        if len(occupied_changed) == 1:
            return f"{cell_name(occupied_changed[0]).capitalize()} is already taken. Please draw in an empty cell."
        return "Those cells are already taken. Please draw in an empty cell."

    def _check_two_marks(self, marked: list[int]) -> str | None:
        candidate = frozenset(marked)
        if candidate != self._two_marks_pending:
            self._two_marks_pending = candidate
            return None
        self._two_marks_pending = None
        return self._enter_escalate("two_marks", candidate)

    def _check_ambiguous(self, cell: int) -> str | None:
        if cell != self._ambiguous_pending:
            self._ambiguous_pending = cell
            self._ambiguous_streak = 1
            return None
        self._ambiguous_streak += 1
        if self._ambiguous_streak < AMBIGUOUS_ESCALATE_READS:
            return None
        self._ambiguous_pending = None
        self._ambiguous_streak = 0
        return self._enter_escalate("ambiguous", frozenset({cell}))

    def _check_wrong_cell(self, cell: int) -> str | None:
        if cell != self._wrong_cell_pending:
            self._wrong_cell_pending = cell
            return None
        self._wrong_cell_pending = None
        self.phase = Phase.ASK_HUMAN
        self._ask_context = "wrong_cell"
        assert self.target_cell is not None
        return f"I asked for {cell_name(self.target_cell)}, but I see a mark in {cell_name(cell)}."

    # -- ASK_HUMAN ----------------------------------------------------------

    def _handle_ask_human(self, observation: Observation, now: float) -> str | None:
        if self._ask_context in ("two_marks", "ambiguous"):
            return self._resolve_human_ask(observation, now)
        if self._ask_context == "wrong_cell":
            return self._resolve_wrong_cell_ask(observation, now)
        if self._ask_context == "resync":
            return self._handle_resync(observation, now)
        return None

    def _resolve_human_ask(self, observation: Observation, now: float) -> str | None:
        assert observation.cell_marks is not None and self.baseline is not None
        marked, ambiguous, occupied_changed = _new_marks(observation.cell_marks, self.board)

        if not marked and not ambiguous and not occupied_changed:
            # §8: "human withdraws / redraws" — the page is clean again.
            self.phase = Phase.WAIT_HUMAN
            self._ask_context = None
            self._pending_cell = None
            return None

        if len(marked) == 1 and not ambiguous and not occupied_changed:
            candidate = marked[0]
            if self._pending_cell != candidate:
                self._pending_cell = candidate
                return None
            self._pending_cell = None
            self._ask_context = None
            return self._commit(candidate, self.human_symbol, observation.ratios, now)

        self._pending_cell = None
        return None  # still contested; keep waiting, no repeat spam

    def _resolve_wrong_cell_ask(self, observation: Observation, now: float) -> str | None:
        assert observation.cell_marks is not None and self.target_cell is not None and observation.ratios is not None
        marked, ambiguous, occupied_changed = _new_marks(observation.cell_marks, self.board)
        erased = self._erased_cells(observation.ratios)

        if erased or occupied_changed:
            return self._check_occupied_drift(erased, occupied_changed)

        if marked == [self.target_cell] and not ambiguous:
            if self._pending_cell != self.target_cell:
                self._pending_cell = self.target_cell
                return None
            self._pending_cell = None
            self._armed_at = None
            self._next_reminder_at = None
            self._ask_context = None
            return self._commit(self.target_cell, self.agent_symbol, observation.ratios, now)

        if not marked and not ambiguous:
            # The stray mark is gone; still armed, no guessing at a
            # half-fixed page (D5) — just resume waiting for the real one.
            self.phase = Phase.WAIT_AGENT_INK
            self._ask_context = None
            self._pending_cell = None
            return None

        self._pending_cell = None
        return None

    # -- Commit + agent reply --------------------------------------------

    def _commit(self, cell: int, symbol: Symbol, ratios: tuple[float, ...] | None, now: float) -> str:
        """Apply a committed move (G1), move the baseline forward (D7),
        and either end the game or hand off the turn (G4)."""
        self.board = apply_move(self.board, cell, symbol)
        assert ratios is not None
        self.baseline = list(ratios)
        self.target_cell = None
        self._ask_context = None
        self._occlusion_since = None
        self._next_occlusion_reminder_at = None
        self._reset_recovery_debounce()

        result = winner(self.board)
        if result is not None or is_draw(self.board):
            self.phase = Phase.GAME_OVER
            self._armed_at = None
            self._next_reminder_at = None
            return self._terminal_message(result, ratios)

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

    def _terminal_message(self, result: tuple[Symbol, tuple[int, int, int]] | None, ratios: tuple[float, ...]) -> str:
        if result is None:
            message = "It's a draw."
        else:
            symbol, line = result
            who = "You win" if symbol == self.human_symbol else "I win"
            message = f"{who}, {describe_line(line)}."
        return f"{message} {self._final_board_check(ratios)}".rstrip()

    def _final_board_check(self, ratios: tuple[float, ...]) -> str:
        """G5: a full-board re-read against the untouched blank baseline,
        the same absolute check RESYNC uses — not a trust in whatever
        `session.board` already believes."""
        mismatches = self._resync_mismatches(ratios)
        if not mismatches:
            return ""
        names = ", ".join(cell_name(c) for c in mismatches)
        return f"But the page doesn't quite match what I have — please check {names}."
