"""Speech, cell vocabulary, and the debug overlay (O1-O4, §6.3, §6.5).

`Speaker` speaks on its own thread so the frame loop never blocks on TTS
(a CLAUDE.md hard rule) and drops anything still queued-but-unsaid when a
newer utterance arrives (O2). It never owns game state; `session.py`
decides *what* to say, this module only says it and draws it.
"""

from __future__ import annotations

import queue
import subprocess
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


def _speak_pyttsx3(text: str) -> None:
    import pyttsx3  # imported lazily: not needed at all with --no-voice

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def _speak_say_command(text: str) -> None:
    subprocess.run(["say", text], check=False, timeout=15)


def default_speak_fn(text: str) -> None:
    """Local OS TTS, pyttsx3 first, macOS `say` as a fallback if it
    misbehaves (CLAUDE.md "Stack"). Never raises: a broken TTS backend
    must not take down the frame loop or the game (O4 still shows the
    text on the overlay regardless)."""
    try:
        _speak_pyttsx3(text)
    except Exception:
        if sys.platform == "darwin":
            try:
                _speak_say_command(text)
            except Exception:
                pass


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
) -> None:
    """Draws the §6.5 debug overlay onto `rectified` in place: per-cell
    X/O, a status banner, the spoken text, a confidence-colored border,
    and a pulsing highlight on the agent's target cell."""
    from inkwatch.perception import cell_bounds  # local import: avoids a hard cv2 dependency for callers that don't render

    size = rectified.shape[0]
    color = _CONFIDENCE_COLOR.get(confidence, (200, 200, 200))

    for idx, (x0, y0, x1, y1) in enumerate(cell_bounds(size)):
        box_color = (255, 180, 0) if idx == target_cell else color
        thickness = 3 if idx == target_cell else 1
        cv2.rectangle(rectified, (x0, y0), (x1, y1), box_color, thickness)
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
