"""Board location, rectification, stability, and per-cell ink measurement.

Finds the four ArUco corner markers, computes the homography from the
inner (board-facing) marker corners to a fixed-size top-down image, and
warps the frame (P1, P2). When no markers decode — a hand-drawn board
made without a printer — falls back down a ladder of hand-drawable
references: four solid black corner squares, then a bare grid's outer
lines (both shadow-tolerant, both feeding the same homography path).
Divides the rectified board into 9 inset cells and measures ink per cell
against an accepted baseline (P3, P4, D1). Gates evaluation on a stable,
unoccluded scene (P5, P6).

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

# Corner smoothing for the hand-drawn detection paths (their corner
# estimates jitter; ArUco is subpixel-stable and never smoothed).
SMOOTH_ALPHA = 0.35  # weight of the newest measurement
SMOOTH_SNAP_FRAC = 0.10  # of the frame diagonal: a bigger jump is a real page move


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


def _homography_from_points(src: np.ndarray, output_size: int) -> np.ndarray:
    """Perspective transform from 4 role-ordered corner points
    (top_left, top_right, bottom_right, bottom_left) to the rectified
    top-down square."""
    dst = np.array(
        [
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(np.asarray(src, dtype=np.float32), dst)


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
    return _homography_from_points(src, output_size), []


# Blob fallback (D1's hand-drawn path, no printer and no ruler needed):
# four solid black squares at the grid corners. Thresholds are fractions
# of the frame so any camera resolution works.
BLOB_MIN_AREA_FRAC = 0.001  # a corner square is at least this much of the frame
BLOB_MAX_AREA_FRAC = 0.05  # ... and at most this (a shadow is bigger)
BLOB_MIN_EXTENT = 0.75  # a filled square fills its bounding box
BLOB_DARKER_THAN = 0.7  # blob mean must be this much darker than the frame median
MIN_CORNER_QUAD_FRAC = 0.15  # the four corner centers must span this much of the frame


def _order_quad(points: np.ndarray) -> np.ndarray:
    """4 (x, y) points -> TL, TR, BR, BL order, the ArUco corner
    convention compute_homography's CORNER_ROLES indices rely on."""
    s = points.sum(axis=1)
    d = points[:, 0] - points[:, 1]
    return np.array(
        [points[np.argmin(s)], points[np.argmax(d)], points[np.argmax(s)], points[np.argmin(d)]],
        dtype=np.float32,
    )


def detect_corner_squares(frame: np.ndarray) -> dict[int, np.ndarray]:
    """Hand-drawn fallback when no ArUco markers decode: finds four solid
    black squares and returns them in detect_markers()'s exact shape —
    {id: corners(4,2)} with ids 0/1/2/3 = top-left/top-right/bottom-right/
    bottom-left and corners ordered TL,TR,BR,BL — so compute_homography()
    runs unchanged. Empty dict when four plausible squares aren't there:
    a random dark object fails the squareness, darkness, or quad-span
    checks rather than hallucinating a board."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Opening: drawn grid lines touch the corner squares and would merge
    # them into one giant contour — eroding a few px cuts the thin lines,
    # dilating restores the fat squares.
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = gray.shape[0] * gray.shape[1]
    frame_median = float(np.median(blurred))
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not BLOB_MIN_AREA_FRAC * frame_area <= area <= BLOB_MAX_AREA_FRAC * frame_area:
            continue
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        _, (w, h), _ = cv2.minAreaRect(contour)
        if min(w, h) == 0 or not 0.6 <= w / h <= 1.6:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        if area / (bw * bh) < BLOB_MIN_EXTENT:
            continue
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [approx], -1, 255, -1)
        if cv2.mean(blurred, mask=mask)[0] > BLOB_DARKER_THAN * frame_median:
            continue
        candidates.append((area, approx.reshape(4, 2).astype(np.float32)))

    if len(candidates) < 4:
        return {}
    candidates.sort(key=lambda c: -c[0])
    quads = [quad for _, quad in candidates[:4]]
    centers = np.array([quad.mean(axis=0) for quad in quads])
    hull_area = cv2.contourArea(cv2.convexHull(centers.astype(np.int32)))
    if hull_area < MIN_CORNER_QUAD_FRAC * frame_area:
        return {}

    s = centers.sum(axis=1)
    d = centers[:, 0] - centers[:, 1]
    picked = {
        0: quads[int(np.argmin(s))],  # top_left
        1: quads[int(np.argmax(d))],  # top_right
        2: quads[int(np.argmax(s))],  # bottom_right
        3: quads[int(np.argmin(d))],  # bottom_left
    }
    return {mid: _order_quad(quad) for mid, quad in picked.items()}


# Bare-grid fallback (a hand-drawn grid with no corner marks at all):
# the two dominant perpendicular line families' outermost supported
# lines. Shadow tolerance comes from adaptive thresholding (ink is
# judged against its local neighborhood, so a smooth shadow gradient
# neither fakes nor hides lines) plus the support requirement below —
# a shadow's hard edge is shorter than a full grid line, so shadow
# fragments can't outvote a real border.
GRID_MIN_LINE_LEN_FRAC = 0.25  # a grid border spans at least this of the frame's small side
GRID_ANGLE_BIN_DEG = 3.0
GRID_ANGLE_TOL_DEG = 8.0
GRID_CLUSTER_GAP_FRAC = 0.04  # rho clustering gap, fraction of the frame's small side
GRID_BORDER_MIN_SUPPORT = 0.10  # an outer line needs at least this share of its family's total length...
GRID_BORDER_REL_SUPPORT = 0.4  # ... and at least this share of the family's strongest cluster —
# a real border is a substantial line; shadow fragments are neither. (A fixed absolute share
# alone is fragile: Hough may return one segment or two for the same drawn line, halving the
# measured support for an implementation reason, not a scene reason.)
MIN_GRID_QUAD_FRAC = 0.15  # the four intersections must span this much of the frame


def _angle_distance(a: float, b: float) -> float:
    """180-periodic distance between two line angles, in degrees."""
    return abs((a - b + 90.0) % 180.0 - 90.0)


def _line_intersection(line_a: tuple[float, float], line_b: tuple[float, float]) -> np.ndarray | None:
    """Intersection of two lines in normal form (theta, rho):
    x*cos(theta) + y*sin(theta) = rho."""
    theta_a, rho_a = line_a
    theta_b, rho_b = line_b
    det = np.cos(theta_a) * np.sin(theta_b) - np.cos(theta_b) * np.sin(theta_a)
    if abs(det) < 1e-6:
        return None
    x = (rho_a * np.sin(theta_b) - rho_b * np.sin(theta_a)) / det
    y = (np.cos(theta_a) * rho_b - np.cos(theta_b) * rho_a) / det
    return np.array([x, y], dtype=np.float32)


def _border_lines(segments: list[tuple], theta_deg: float, side: int) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """The (inner-side, outer-side) pair of outermost supported lines of
    one grid family, in normal form. None when the family doesn't have
    two well-supported extremes."""
    normal = np.radians((theta_deg + 90.0) % 180.0)
    normal_vec = np.array([np.cos(normal), np.sin(normal)])
    members = []
    for x1, y1, x2, y2, length, _ in segments:
        mid = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        members.append((float(mid @ normal_vec), length))
    if len(members) < 2:
        return None
    members.sort()
    total = sum(length for _, length in members)

    clusters = [[members[0]]]
    for member in members[1:]:
        if member[0] - clusters[-1][-1][0] > GRID_CLUSTER_GAP_FRAC * side:
            clusters.append([])
        clusters[-1].append(member)
    if len(clusters) < 2:
        return None

    def fit(cluster: list[tuple[float, float]]) -> tuple[float, float, float]:
        rho = sum(r * length for r, length in cluster) / sum(length for _, length in cluster)
        support = sum(length for _, length in cluster) / total
        return normal, rho, support

    lo, hi = fit(clusters[0]), fit(clusters[-1])
    strongest = max(fit(cluster)[2] for cluster in clusters)
    for border in (lo, hi):
        if border[2] < GRID_BORDER_MIN_SUPPORT or border[2] < GRID_BORDER_REL_SUPPORT * strongest:
            return None
    return (lo[0], lo[1]), (hi[0], hi[1])


def detect_grid_lines(frame: np.ndarray) -> np.ndarray | None:
    """Last-resort board finding for a bare hand-drawn grid: finds the
    two dominant ~90°-apart line directions, takes the outermost
    supported line in each as a border, and returns the four border
    intersections ordered TL,TR,BR,BL (compute_homography's role order).
    None — never a guessed quad — when the scene has no convincing grid:
    too few lines, no perpendicular family, weak borders, or a tiny/
    degenerate intersection quad."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    scale = 1.0
    small = gray
    if min(gray.shape[:2]) > 480:
        scale = 480.0 / min(gray.shape[:2])
        small = cv2.resize(gray, None, fx=scale, fy=scale)
    side = min(small.shape[:2])

    block = max(15, (side // 8) | 1)
    binary = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 5)
    min_len = GRID_MIN_LINE_LEN_FRAC * side
    lines = cv2.HoughLinesP(
        binary, 1, np.pi / 180,
        threshold=int(min_len * 0.6),
        minLineLength=int(min_len),
        maxLineGap=int(0.03 * side),
    )
    if lines is None or len(lines) < 4:
        return None

    segments = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
        segments.append((float(x1), float(y1), float(x2), float(y2), length, angle))

    # Dominant direction from a length-weighted angle histogram (smoothed
    # circularly so a family straddling the 0/180 wrap doesn't split).
    bins = int(180 / GRID_ANGLE_BIN_DEG)
    hist = np.zeros(bins)
    for *_, length, angle in segments:
        hist[int(angle / GRID_ANGLE_BIN_DEG) % bins] += length
    smooth = hist + np.roll(hist, 1) + np.roll(hist, -1)
    theta1 = (float(np.argmax(smooth)) + 0.5) * GRID_ANGLE_BIN_DEG
    near = [(a, l) for *_, l, a in segments if _angle_distance(a, theta1) <= GRID_ANGLE_TOL_DEG]
    if near:
        # Double-angle mean: line angles are 180-periodic.
        z = sum(l * np.exp(2j * np.radians(a)) for a, l in near)
        theta1 = float(np.degrees(np.angle(z) / 2) % 180.0)
    theta2 = (theta1 + 90.0) % 180.0

    fam1 = [s for s in segments if _angle_distance(s[5], theta1) <= GRID_ANGLE_TOL_DEG]
    fam2 = [s for s in segments if _angle_distance(s[5], theta2) <= GRID_ANGLE_TOL_DEG]
    borders1 = _border_lines(fam1, theta1, side) if fam1 else None
    borders2 = _border_lines(fam2, theta2, side) if fam2 else None
    if borders1 is None or borders2 is None:
        return None

    points = []
    for border_a in borders1:
        for border_b in borders2:
            point = _line_intersection(border_a, border_b)
            if point is None:
                return None
            points.append(point / scale)
    quad = _order_quad(np.array(points))

    int_quad = quad.astype(np.int32)
    area = cv2.contourArea(int_quad)
    hull_area = cv2.contourArea(cv2.convexHull(int_quad))
    frame_area = gray.shape[0] * gray.shape[1]
    if hull_area == 0 or area < MIN_GRID_QUAD_FRAC * frame_area or area / hull_area < 0.9:
        return None
    return quad


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
        self._smooth_src: np.ndarray | None = None

    def _smooth_corners(self, src: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Hand-drawn detections (corner squares, grid lines) jitter a
        few px frame to frame, and jitter in the rectified image looks
        like motion to the stability gate (P5) — so EMA-smooth the corner
        points. A jump beyond the snap distance (the page really moved)
        is passed through unsmoothed instead of lagging behind it."""
        if self._smooth_src is not None:
            frame_diag = float(np.hypot(frame.shape[0], frame.shape[1]))
            jumped = np.linalg.norm(src - self._smooth_src, axis=1).max() > SMOOTH_SNAP_FRAC * frame_diag
            if not jumped:
                src = SMOOTH_ALPHA * src + (1.0 - SMOOTH_ALPHA) * self._smooth_src
        self._smooth_src = src
        return src

    def update(self, frame: np.ndarray, now: float | None = None) -> RectifyResult:
        now = time.monotonic() if now is None else now
        detected = detect_markers(frame, self._detector)
        homography, missing = compute_homography(detected, self.output_size)

        if homography is None:
            # Hand-drawn paths (D1's fallback ladder): four solid black
            # corner squares first, then a bare grid's outer lines. ArUco
            # stays the primary, more precise path whenever it decodes.
            squares = detect_corner_squares(frame)
            src: np.ndarray | None = None
            if squares:
                src = np.array(
                    [squares[mid][idx] for _, (mid, idx) in CORNER_ROLES.items()],
                    dtype=np.float32,
                )
            else:
                src = detect_grid_lines(frame)
            if src is not None:
                homography = _homography_from_points(self._smooth_corners(src, frame), self.output_size)

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

        self._smooth_src = None
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
