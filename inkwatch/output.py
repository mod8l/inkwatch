"""Speech, cell vocabulary, and the debug overlay (O1-O4, §6.3, §6.5).

`Speaker` speaks on its own thread so the frame loop never blocks on TTS
(a CLAUDE.md hard rule) and drops anything still queued-but-unsaid when a
newer utterance arrives (O2). It never owns game state; `session.py`
decides *what* to say, this module only says it and draws it.
"""

from __future__ import annotations

import queue
# Only ever invoked below (_speak_say_command) with a fixed argv list, never a shell string.
import subprocess  # nosec B404
import sys
import threading
from typing import Callable

import cv2
import numpy as np

from inkwatch.rules import Board

# §6.3: row (top/middle/bottom) x column (left/center/right); the
# middle-center cell is just "center".
CELL_NAMES: tuple[str, ...] = (
    "top left", "top center", "top right",
    "middle left", "center", "middle right",
    "bottom left", "bottom center", "bottom right",
)

# The 8 winning lines from rules.LINES, in the same order, described the
# way a person would say them out loud.
_LINE_NAMES: dict[tuple[int, int, int], str] = {
    (0, 1, 2): "the top row",
    (3, 4, 5): "the middle row",
    (6, 7, 8): "the bottom row",
    (0, 3, 6): "the left column",
    (1, 4, 7): "the middle column",
    (2, 5, 8): "the right column",
    (0, 4, 8): "the diagonal from top left",
    (2, 4, 6): "the diagonal from top right",
}


def cell_name(cell: int) -> str:
    return CELL_NAMES[cell]


def describe_line(line: tuple[int, int, int]) -> str:
    return _LINE_NAMES[line]


_ENGINE = None


def _speak_pyttsx3(text: str) -> None:
    import pyttsx3  # imported lazily: not needed at all with --no-voice

    global _ENGINE
    if _ENGINE is None:
        # One engine for the process: re-init per utterance costs ~0.5 s
        # each time and, on the espeak driver, fires a stray ctypes
        # callback into a garbage-collected engine (the "weakly-referenced
        # object no longer exists" spam). The only caller is Speaker's
        # single worker thread, so sharing one engine is safe here.
        _ENGINE = pyttsx3.init()
    _ENGINE.say(text)
    _ENGINE.runAndWait()


def _speak_say_command(text: str) -> None:
    # List-form argv (no shell=True) and a fixed macOS system command --
    # not resolved from any untrusted input.
    subprocess.run(["say", text], check=False, timeout=15)  # nosec


def default_speak_fn(text: str) -> None:
    """Local OS TTS, pyttsx3 first, macOS `say` as a fallback if it
    misbehaves (CLAUDE.md "Stack"). Never raises: a broken TTS backend
    must not take down the frame loop or the game (O4 still shows the
    text on the overlay regardless) -- but a failure is still printed,
    not swallowed silently, so it's visible when debugging."""
    try:
        _speak_pyttsx3(text)
    except Exception:
        if sys.platform == "darwin":
            try:
                _speak_say_command(text)
            except Exception as exc:
                print(f"TTS fallback also failed: {exc}", file=sys.stderr)


class Speaker:
    """Non-blocking speech queue with at-most-one-pending utterance (O1, O2).

    `speak_fn` is injectable so tests can verify queuing/cancellation
    behavior without a real TTS engine or audio hardware.
    """

    def __init__(self, enabled: bool = True, speak_fn: Callable[[str], None] | None = None) -> None:
        self.enabled = enabled
        self._speak_fn = speak_fn or default_speak_fn
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def say(self, text: str) -> None:
        """Queue `text`, dropping anything queued but not yet started (O2)."""
        if not self.enabled:
            return
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text)

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                return
            self._speak_fn(text)

    def close(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=2)
            self._thread = None


_CONFIDENCE_COLOR = {
    "accepted": (0, 200, 0),
    "escalating": (0, 165, 255),
    "asking": (0, 0, 255),
}


def draw_overlay(
    rectified: np.ndarray,
    *,
    board: Board,
    phase: str,
    message: str | None,
    target_cell: int | None = None,
    confidence: str = "accepted",
    cell_marks: tuple[str, ...] | None = None,
    debug: bool = False,
    grid_lines: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    highlight_cells: frozenset[int] = frozenset(),
    target_symbol: str | None = None,
) -> None:
    """Draws the §6.5 debug overlay onto `rectified` in place: per-cell
    X/O, a status banner, the spoken text, a confidence-colored border,
    and a filled, unmissable highlight on the cells the human needs to
    act on — the agent's armed target (with the symbol to draw, §6.3's
    spoken-cell/visual-highlight pairing) and every cell involved in an
    open question. With `grid_lines` (bare-grid boards), cell boxes
    follow the REAL detected lines rather than perfect thirds."""
    from inkwatch.perception import DEFAULT_CELL_INSET, cell_bounds, cell_bounds_grid  # local import: keeps rendering optional

    size = rectified.shape[0]
    color = _CONFIDENCE_COLOR.get(confidence, (200, 200, 200))
    bounds = (
        cell_bounds_grid(grid_lines[0], grid_lines[1], DEFAULT_CELL_INSET, size)
        if grid_lines is not None
        else cell_bounds(size)
    )

    for idx, (x0, y0, x1, y1) in enumerate(bounds):
        pending = cell_marks is not None and cell_marks[idx] != "none" and board[idx] is None
        if idx == target_cell:
            box_color, thickness = (255, 180, 0), 3
        elif pending or idx in highlight_cells:
            # §9 "two new marks at once": both candidate cells need to be
            # visible on the overlay, not just in --debug's per-cell text.
            box_color, thickness = (0, 255, 255), 2
        else:
            box_color, thickness = color, 1
        if idx in highlight_cells or idx == target_cell:
            fill = rectified.copy()
            cv2.rectangle(fill, (x0, y0), (x1, y1), box_color, -1)
            cv2.addWeighted(fill, 0.22, rectified, 0.78, 0, rectified)
        cv2.rectangle(rectified, (x0, y0), (x1, y1), box_color, thickness)
        if idx == target_cell and target_symbol is not None:
            # What to draw, where — big enough to read from the chair.
            text_size, _ = cv2.getTextSize(target_symbol, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5)
            tx = (x0 + x1 - text_size[0]) // 2
            ty = (y0 + y1 + text_size[1]) // 2
            cv2.putText(rectified, target_symbol, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 3.0, box_color, 5, cv2.LINE_AA)
        symbol = board[idx]
        if symbol is not None:
            cv2.putText(
                rectified, symbol, (x0 + 10, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA,
            )
        if debug and cell_marks is not None:
            cv2.putText(
                rectified, cell_marks[idx], (x0 + 4, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA,
            )

    cv2.putText(rectified, phase, (10, size - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if message:
        cv2.putText(rectified, message, (10, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
