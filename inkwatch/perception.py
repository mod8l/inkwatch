"""Board location, rectification, stability, and per-cell ink measurement.

Finds the four ArUco corner markers, computes the homography from the
inner (board-facing) marker corners to a fixed-size top-down image, and
warps the frame (P1, P2). Divides the rectified board into 9 inset cells
and measures ink per cell against an accepted baseline (P3, P4, D1).
Gates evaluation on a stable, unoccluded scene (P5, P6).

Perception never mutates game state. `Perceiver.observe()` is the one
entry point session.py calls each frame: it takes the frame and session's
current baseline, and hands back an `Observation` (events.py) — never a
raw frame, never anything it owns itself. The confidence rules that turn
a classification into an accepted move (D2, D3) and everything downstream
of that live in the session state machine (M3+), not here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from inkwatch.events import CellMark, Observation

# Corner name -> (marker id, index of that marker's corner facing the
# board interior). ArUco corners are returned in the order the marker was
# encoded: top-left, top-right, bottom-right, bottom-left. For an
# upright, axis-aligned printed marker that order matches the image, so
# e.g. the top-left marker's board-facing corner is its own bottom-right
# corner (index 2).
CORNER_ROLES: dict[str, tuple[int, int]] = {
    "top_left": (0, 2),
    "top_right": (1, 3),
    "bottom_right": (2, 0),
    "bottom_left": (3, 1),
}

DEFAULT_DICTIONARY = cv2.aruco.DICT_4X4_50
DEFAULT_OUTPUT_SIZE = 600
DEFAULT_HOLD_SECONDS = 0.5

DEFAULT_CELL_INSET = 0.15
DEFAULT_INK_LOW = 0.02
DEFAULT_INK_HIGH = 0.05

DEFAULT_STABILITY_FRAMES = 10
DEFAULT_MOTION_THRESHOLD = 2.0  # mean abs pixel diff (0-255) between consecutive rectified frames


@dataclass
class RectifyResult:
    """Outcome of one frame through the board tracker."""

    found: bool
    rectified: np.ndarray | None
    missing_corners: list[str] = field(default_factory=list)
    homography: np.ndarray | None = None
    reused: bool = False  # True when found via a held-over homography (P2)


def detect_markers(
    frame: np.ndarray, detector: cv2.aruco.ArucoDetector
) -> dict[int, np.ndarray]:
    """Detect ArUco markers, returning {id: corners(4,2) float32}."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _ = detector.detectMarkers(gray)
    detected: dict[int, np.ndarray] = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            detected[int(i)] = c.reshape(4, 2).astype(np.float32)
    return detected


def compute_homography(
    detected: dict[int, np.ndarray], output_size: int
) -> tuple[np.ndarray | None, list[str]]:
    """Build the marker-corners -> top-down-image homography.

    Returns (homography, missing_corner_names). homography is None
    unless all four corner markers were detected.
    """
    missing = [name for name, (mid, _) in CORNER_ROLES.items() if mid not in detected]
    if missing:
        return None, missing

    src = np.array(
        [detected[mid][idx] for _, (mid, idx) in CORNER_ROLES.items()],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src, dst)
    return homography, []


def cell_bounds(
    size: int, inset: float = DEFAULT_CELL_INSET
) -> list[tuple[int, int, int, int]]:
    """Inner-cell (x0, y0, x1, y1) box for each of the 9 cells.

    Row-major, top-left to bottom-right. Each cell is shrunk by `inset`
    on every side so the grid lines drawn on the sheet don't count as
    ink (P3).
    """
    cell = size / 3
    margin = cell * inset
    bounds = []
    for row in range(3):
        for col in range(3):
            x0 = col * cell + margin
            y0 = row * cell + margin
            x1 = (col + 1) * cell - margin
            y1 = (row + 1) * cell - margin
            bounds.append((round(x0), round(y0), round(x1), round(y1)))
    return bounds


def ink_ratio(cell_image: np.ndarray) -> float:
    """Fraction of dark (ink) pixels in a cell crop.

    Adaptive thresholding rather than a single global cutoff, so uneven
    lighting across the page doesn't bias one cell against another (P4).
    """
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY) if cell_image.ndim == 3 else cell_image
    h, w = gray.shape[:2]
    block_size = max(3, (min(h, w) // 2) | 1)  # odd, roughly half the cell
    dark = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        5,
    )
    return float(np.count_nonzero(dark)) / dark.size


def measure_cells(rectified: np.ndarray, inset: float = DEFAULT_CELL_INSET) -> list[float]:
    """Ink ratio for each of the 9 cells, row-major (P3, P4)."""
    size = rectified.shape[0]
    return [ink_ratio(rectified[y0:y1, x0:x1]) for x0, y0, x1, y1 in cell_bounds(size, inset)]


def classify_cell(
    ratio: float,
    baseline: float,
    low: float = DEFAULT_INK_LOW,
    high: float = DEFAULT_INK_HIGH,
) -> CellMark:
    """Classify one cell's ink delta against its accepted baseline (D1)."""
    delta = ratio - baseline
    if delta >= high:
        return "marked"
    if delta >= low:
        return "ambiguous"
    return "none"


def classify_cells(
    ratios: list[float],
    baseline: list[float],
    low: float = DEFAULT_INK_LOW,
    high: float = DEFAULT_INK_HIGH,
) -> list[CellMark]:
    """Classify all 9 cells against their per-cell baselines (D1)."""
    return [classify_cell(r, b, low, high) for r, b in zip(ratios, baseline)]


class BoardTracker:
    """Per-frame board location, with a short hold-over when markers drop out.

    P1: computes a homography every frame and warps to a fixed-size
    top-down image.
    P2: if fewer than 4 markers are found, reuses the last homography for
    up to `hold_seconds`, then reports not-found (the caller drives
    BOARD_LOST from that).
    """

    def __init__(
        self,
        dictionary_id: int = DEFAULT_DICTIONARY,
        output_size: int = DEFAULT_OUTPUT_SIZE,
        hold_seconds: float = DEFAULT_HOLD_SECONDS,
    ) -> None:
        self.output_size = output_size
        self.hold_seconds = hold_seconds
        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._params)
        self._last_homography: np.ndarray | None = None
        self._last_seen: float | None = None

    def update(self, frame: np.ndarray, now: float | None = None) -> RectifyResult:
        now = time.monotonic() if now is None else now
        detected = detect_markers(frame, self._detector)
        homography, missing = compute_homography(detected, self.output_size)

        if homography is not None:
            self._last_homography = homography
            self._last_seen = now
            rectified = self._warp(frame, homography)
            return RectifyResult(found=True, rectified=rectified, homography=homography)

        if self._last_homography is not None and self._last_seen is not None:
            if now - self._last_seen <= self.hold_seconds:
                rectified = self._warp(frame, self._last_homography)
                return RectifyResult(
                    found=True,
                    rectified=rectified,
                    missing_corners=missing,
                    homography=self._last_homography,
                    reused=True,
                )

        return RectifyResult(found=False, rectified=None, missing_corners=missing)

    def _warp(self, frame: np.ndarray, homography: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, homography, (self.output_size, self.output_size))


class StabilityGate:
    """Tracks whether the rectified scene has gone quiet (P5) or is
    occluded (P6), across successive calls to `update()`.

    Stable requires N consecutive frames with all four markers currently
    visible (not a P2 hold-over) and low inter-frame motion. Anything
    else — markers missing/held-over, or motion above threshold — is
    occluded and resets the quiet-frame count.
    """

    def __init__(
        self,
        stability_frames: int = DEFAULT_STABILITY_FRAMES,
        motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
    ) -> None:
        self.stability_frames = stability_frames
        self.motion_threshold = motion_threshold
        self._prev: np.ndarray | None = None
        self._quiet_count = 0

    def update(self, rectified: np.ndarray | None, markers_visible: bool) -> tuple[bool, bool]:
        """Returns (stable, occluded) for this frame."""
        if rectified is None or not markers_visible:
            self._prev = None
            self._quiet_count = 0
            return False, True

        moved = False
        if self._prev is not None:
            diff = cv2.absdiff(rectified, self._prev)
            moved = float(diff.mean()) > self.motion_threshold
        self._prev = rectified

        if moved:
            self._quiet_count = 0
            return False, True

        self._quiet_count += 1
        return self._quiet_count >= self.stability_frames, False


class Perceiver:
    """The one call session.py (or `__main__.py` on its behalf) makes per
    frame: locate the board, gate on stability, measure ink, and classify
    against the baseline session currently owns — packaged as one
    `Observation` (events.py), never a raw frame.
    """

    def __init__(
        self,
        board: BoardTracker | None = None,
        stability: StabilityGate | None = None,
        cell_inset: float = DEFAULT_CELL_INSET,
    ) -> None:
        self.board = board if board is not None else BoardTracker()
        self.stability = stability if stability is not None else StabilityGate()
        self.cell_inset = cell_inset
        # The last rectified frame, for a caller's own display purposes
        # only (e.g. __main__.py's overlay) — never part of `Observation`,
        # which never carries pixels to session.py.
        self.last_rectified: np.ndarray | None = None

    def observe(
        self,
        frame: np.ndarray,
        baseline: list[float] | None,
        now: float | None = None,
        low: float = DEFAULT_INK_LOW,
        high: float = DEFAULT_INK_HIGH,
    ) -> Observation:
        now = time.monotonic() if now is None else now
        result = self.board.update(frame, now)
        self.last_rectified = result.rectified if result.found else None
        markers_visible = result.found and not result.reused
        stable, occluded = self.stability.update(
            result.rectified if result.found else None, markers_visible
        )

        ratios: list[float] | None = None
        marks: list[CellMark] | None = None
        if result.found:
            ratios = measure_cells(result.rectified, inset=self.cell_inset)
            if baseline is not None:
                marks = classify_cells(ratios, baseline, low, high)

        return Observation(
            frame_ts=now,
            found=result.found,
            stable=stable,
            occluded=occluded,
            missing_corners=tuple(result.missing_corners),
            ratios=tuple(ratios) if ratios is not None else None,
            cell_marks=tuple(marks) if marks is not None else None,
        )
