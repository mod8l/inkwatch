"""Event types crossing the perception -> session boundary.

`Observation` is what perception hands to session every frame: it never
includes raw pixels (ARCHITECTURE.md §2, "raw frames never cross to the
session") or a baseline of its own — perception measures against whatever
baseline session currently owns and hands back, it doesn't keep one
(CLAUDE.md hard rule: perception never mutates game state).

Kept free of cv2/numpy so a future `rules.py`/`decision.py` import of a
type from here (CLAUDE.md's hard rule carves out that exception) can never
drag OpenCV in transitively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CellMark = Literal["none", "ambiguous", "marked"]


@dataclass(frozen=True)
class Observation:
    """One frame's read of the board (P1-P6, D1).

    `cell_marks` and `ratios` are `None` when the board wasn't found, or
    was found but classification wasn't requested (no baseline yet, e.g.
    during CALIBRATING before the first one is set).
    """

    frame_ts: float
    found: bool
    stable: bool
    occluded: bool
    missing_corners: tuple[str, ...]
    ratios: tuple[float, ...] | None
    cell_marks: tuple[CellMark, ...] | None
