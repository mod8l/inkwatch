"""Realistic synthetic camera frames for driving Inkwatch without a camera.

Everything the scenario director (`tools/simulate.py`) shows the running
app comes from here. Realism is the point of the exercise — the app must
be unable to tell the feed from Gad's actual desk — so the world is built
from the real recordings wherever possible:

- `RecordingBackground`: the median of quiet empty-board frames from a
  real `--record`ed session (real paper, real pencil grid, real lighting,
  real sensor noise floor). Extract once with
  `python tools/simrender.py --extract sessions/<id>` into
  `recordings/sim_assets/`.
- `SyntheticBackground`: a fully procedural paper + wobbly hand-drawn
  grid for machines without a recording (CI, a stranger's checkout). The
  tests in `tests/test_simulate.py` run against this one.
- Pencil: the actual pencil RGBA sprite segmented from the recording
  (turquoise body, wooden cone, graphite point), translated around the
  frame tip-first, entering and leaving from the left edge exactly like
  the real hand does. Fixed orientation — Gad is right-handed and the
  pencil points the same way in every recorded frame.
- Strokes: X/O paths with human wobble (subdivided polylines with a
  low-frequency sinusoid plus jitter), graphite gray sampled from real
  strokes (~median 143 on paper ~190), 2 px anti-aliased — thin pencil
  marks, exactly the case the dilated `ink_ratio` is tuned for.
- Effects: soft shadows (multiplicative blobs, adaptive-threshold-visible
  edges), erasing (paper texture blended back over the stroke), page
  bumps (pose affine with motion-smear plus a hand shadow sweep, so
  detection drops out mid-bump like a real grab), and a lingering-hand
  hover (pencil tremor plus shadow over the grid border, so the board
  read flickers instead of going stable — a static pencil over a cell
  would otherwise be legitimate ink).
- Per-frame Gaussian sensor noise, sigma matched to the measured
  inter-frame absdiff of the real camera (~1.0 against motion_threshold
  2.0).

Deterministic: everything is seeded per Scene. `Scene.tick(dt)` advances
the animation clock and renders one frame, so a 30 fps driver gets
frame-accurate control of timing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from inkwatch.perception import (
    BoardTracker,
    cell_bounds,
    cell_bounds_grid,
    DEFAULT_CELL_INSET,
    DEFAULT_OUTPUT_SIZE,
)

FRAME_W, FRAME_H = 640, 480
# Measured ground truth (the real recording, quiet windows): rectified
# inter-frame absdiff p50 ~1.6, with >2.0 spikes in bursts — the real
# stability gate lives with ~20-40% "moved" frames. sigma 0.3 reproduces
# the calm end of that (mean ~1.4, ~15-25% moved): realistic jitter for
# the detection path without making scenario timing chaotic.
NOISE_SIGMA = 0.15

GRAPHITE_GRAY = (141, 143, 145)  # BGR, sampled from real X strokes
STROKE_WIDTH = 1  # 1 px at VGA: a full X dilates to ~4.5-6.5% of a cell —
# Gad's thin pencil measured at ~4-5% (NOTES.md); a 55% X reads ambiguous

PENCIL_DWELL_S = 0.15
PENCIL_ENTER_S = 0.35
PENCIL_EXIT_S = 0.30


# ---------------------------------------------------------------- backgrounds


class RecordingBackground:
    """Median empty-board frame from a real recording, extracted into
    recordings/sim_assets/ by `--extract`."""

    def __init__(self, assets_dir: Path | str = "recordings/sim_assets") -> None:
        assets_dir = Path(assets_dir)
        self.frame = cv2.imread(str(assets_dir / "paper_bg.png"))
        if self.frame is None:
            raise FileNotFoundError(
                f"{assets_dir}/paper_bg.png missing — run: python tools/simrender.py --extract sessions/<id>"
            )
        pencil = cv2.imread(str(assets_dir / "pencil.png"), cv2.IMREAD_UNCHANGED)
        tip = (assets_dir / "pencil_tip.txt").read_text().strip().split(",")
        self.pencil = pencil
        self.pencil_tip = (int(tip[0]), int(tip[1]))


class SyntheticBackground:
    """Procedural paper + hand-style wobbly grid (CI-friendly: no
    recording needed). Geometry mirrors the real desk frame: grid
    slightly rotated, filling the left half of a 640x480 frame."""

    GRID = (45, 30, 400, 430)  # x, y, w, h of the grid's outer square

    def __init__(self, seed: int = 11) -> None:
        rng = np.random.default_rng(seed)
        base = np.full((FRAME_H, FRAME_W, 3), (196, 202, 206), np.float32)
        # paper brightness drift + fiber noise
        yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
        base += (6 * np.sin(xx / 140.0) + 5 * np.cos(yy / 170.0))[..., None]
        base += rng.normal(0, 1.6, base.shape).astype(np.float32)
        frame = np.clip(base, 0, 255).astype(np.uint8)

        x, y, w, h = self.GRID
        # wobbly hand grid: 4 outer + 2+2 inner lines, thirds with jitter
        xs = [x, x + w / 3 + 6, x + 2 * w / 3 - 4, x + w]
        ys = [y, y + h / 3 - 5, y + 2 * h / 3 + 7, y + h]
        line = (108, 112, 115)
        for gx in xs:
            pts = _wobble_line((gx, y - 6), (gx, y + h + 6), rng, amp=3.0)
            cv2.polylines(frame, [pts], False, line, 2, cv2.LINE_AA)
        for gy in ys:
            pts = _wobble_line((x - 8, gy), (x + w + 8, gy), rng, amp=3.0)
            cv2.polylines(frame, [pts], False, line, 2, cv2.LINE_AA)
        self.frame = frame
        # No recording -> no real pencil; a plausible stand-in sprite the
        # same size/shade (tests never assert on pencil pixels).
        self.pencil = _synthetic_pencil()
        self.pencil_tip = (254, 8)


def _synthetic_pencil() -> np.ndarray:
    w, h = 259, 247
    rgba = np.zeros((h, w, 4), np.uint8)
    cv2.line(rgba, (254, 8), (60, 200), (150, 170, 190, 255), 26, cv2.LINE_AA)  # wood cone
    cv2.line(rgba, (120, 140), (20, 236), (140, 190, 175, 255), 44, cv2.LINE_AA)  # body
    cv2.circle(rgba, (254, 8), 7, (60, 60, 62, 255), -1, cv2.LINE_AA)  # graphite
    return rgba


# ------------------------------------------------------------------- strokes


def _wobble_line(p0: tuple[float, float], p1: tuple[float, float], rng, amp: float = 2.5, step: float = 3.0) -> np.ndarray:
    """Subdivided p0->p1 polyline with human wobble: a couple of
    low-frequency sine cycles plus jitter, perpendicular to the line."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    vec = p1 - p0
    length = float(np.hypot(*vec)) or 1.0
    direction = vec / length
    normal = np.array([-direction[1], direction[0]])
    n = max(2, int(length / step))
    phase, cycles = rng.uniform(0, math.pi), rng.uniform(1.5, 3.0)
    pts = []
    for i in range(n + 1):
        t = i / n
        off = amp * math.sin(phase + cycles * 2 * math.pi * t) + rng.normal(0, 0.6)
        pts.append(p0 + vec * t + normal * off)
    return np.array(pts, np.int32)


def _arc_length(points: np.ndarray) -> float:
    d = np.diff(points.astype(float), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def _partial_path(points: np.ndarray, frac: float) -> np.ndarray:
    """First `frac` of the polyline by arc length (the drawn part)."""
    if frac >= 1.0:
        return points
    d = np.diff(points.astype(float), axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = frac * cum[-1]
    idx = int(np.searchsorted(cum, target, side="right")) - 1
    idx = min(idx, len(seg) - 1)
    t = 0.0 if seg[idx] == 0 else (target - cum[idx]) / seg[idx]
    head = points[idx] + (points[idx + 1] - points[idx]) * t
    return np.vstack([points[: idx + 1], head.astype(points.dtype)])


class Mark:
    """One drawn symbol: 1-2 wobbly stroke paths in raw-frame coords,
    revealed progressively by `draw(frame, frac)`; `head(frac)` is where
    the pencil tip sits at that progress."""

    def __init__(self, paths: list[np.ndarray]) -> None:
        self.paths = paths
        self.lengths = [_arc_length(p) for p in paths]
        self.total = sum(self.lengths)

    def draw(self, frame: np.ndarray, frac: float, color=GRAPHITE_GRAY) -> None:
        remaining = frac * self.total
        for path, length in zip(self.paths, self.lengths):
            if remaining <= 0:
                break
            part = _partial_path(path, min(1.0, remaining / length))
            if len(part) >= 2:
                cv2.polylines(frame, [part], False, color, STROKE_WIDTH, cv2.LINE_AA)
            remaining -= length

    def head(self, frac: float) -> tuple[float, float]:
        remaining = frac * self.total
        for path, length in zip(self.paths, self.lengths):
            if remaining <= length:
                part = _partial_path(path, max(0.02, remaining / max(length, 1e-6)))
                return float(part[-1][0]), float(part[-1][1])
            remaining -= length
        last = self.paths[-1]
        return float(last[-1][0]), float(last[-1][1])


def make_x(box: tuple[int, int, int, int], rng, h_inv: np.ndarray) -> Mark:
    """Two diagonal strokes inside the rectified-space inner box, mapped
    to raw-frame coords through the inverse homography."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    mx, my = w * 0.30, h * 0.30
    diag1 = _wobble_line((x0 + mx, y0 + my), (x1 - mx, y1 - my), rng)
    diag2 = _wobble_line((x1 - mx, y0 + my * 1.3), (x0 + mx * 1.2, y1 - my), rng)
    return Mark([_to_raw(diag1, h_inv), _to_raw(diag2, h_inv)])


def make_o(box: tuple[int, int, int, int], rng, h_inv: np.ndarray) -> Mark:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) * 0.22, (y1 - y0) * 0.22
    turns = 1.12  # hand Os overshoot the close
    n = 60
    phase = rng.uniform(0, math.pi)
    angles = np.linspace(0, turns * 2 * math.pi, n) + phase
    wob = 1.0 + 0.10 * np.sin(3 * angles + rng.uniform(0, math.pi)) + rng.normal(0, 0.03, n)
    pts = np.stack([cx + rx * wob * np.cos(angles), cy + ry * wob * np.sin(angles)], axis=1)
    return Mark([_to_raw(pts.astype(np.int32), h_inv)])


def _to_raw(points: np.ndarray, h_inv: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32).reshape(1, -1, 2)
    raw = cv2.perspectiveTransform(pts, h_inv).reshape(-1, 2)
    return np.round(raw).astype(np.int32)


# ------------------------------------------------------------------- effects


def _alpha_sprite(dst: np.ndarray, sprite: np.ndarray, tip: tuple[int, int], at: tuple[float, float]) -> None:
    """Paste RGBA `sprite` so its graphite tip lands on `at`."""
    h, w = sprite.shape[:2]
    x = int(round(at[0] - tip[0]))
    y = int(round(at[1] - tip[1]))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - x, y0 - y
    region = sprite[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
    alpha = (region[..., 3:4].astype(np.float32) / 255.0) * 0.96
    dst[y0:y1, x0:x1] = (region[..., :3] * alpha + dst[y0:y1, x0:x1] * (1 - alpha)).astype(np.uint8)


def _soft_blob_mask(shape: tuple[int, int], center: tuple[float, float], radii: tuple[float, float], blur: int = 31, jitter: float = 0.15, rng=None) -> np.ndarray:
    """A hand-ish soft ellipse mask (0..1 float), slightly deformed."""
    rng = rng or np.random.default_rng(0)
    mask = np.zeros(shape, np.float32)
    n = 9
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = []
    for a in angles:
        r = 1.0 + rng.uniform(-jitter, jitter)
        pts.append((center[0] + radii[0] * r * math.cos(a), center[1] + radii[1] * r * math.sin(a)))
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 1.0)
    return cv2.GaussianBlur(mask, (blur | 1, blur | 1), 0)


HAND_COLOR = (72, 80, 95)  # skin-ish dark, BGR


def _apply_hand(frame: np.ndarray, mask: np.ndarray, opacity: float = 0.85) -> np.ndarray:
    """An opaque hand: ADDITIVE pull toward the skin color. Multiplicative
    'shadow' darkening preserves local contrast (paper and line dim
    together, the grid survives — adaptive threshold still fires); a real
    hand replaces the light, so lines under it lose their contrast and
    detection actually drops out."""
    a = (mask * opacity)[..., None]
    color = np.array(HAND_COLOR, np.float32)
    return (frame.astype(np.float32) * (1.0 - a) + color * a).astype(np.uint8)


# --------------------------------------------------------------------- scene


class _Action:
    """One scheduled animation (draw / linger / bump / erase / hover)."""

    def __init__(self, kind: str, duration: float, data: dict) -> None:
        self.kind = kind
        self.duration = duration
        self.data = data
        self.t = 0.0

    @property
    def done(self) -> bool:
        return self.t >= self.duration


class Scene:
    """The simulated desk: paper world + pencil + effects, tick-driven.

    Coordinates: the "world" (background + committed marks) lives in raw
    frame space and moves under the bump pose affine; pencil and shadows
    are camera-space (the hand is not glued to the page).
    """

    def __init__(self, background, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)
        self.bg = background.frame.copy()
        self.pencil_sprite = background.pencil
        self.pencil_tip = background.pencil_tip
        self.noise_rng = np.random.default_rng(seed + 1)

        tracker = BoardTracker(output_size=DEFAULT_OUTPUT_SIZE)
        result = None
        for _ in range(5):
            result = tracker.update(self.bg)
            if result.found and result.homography is not None:
                break
        if result is None or not result.found or result.homography is None:
            raise RuntimeError("background board not detected — cannot map cells")
        self.homography = result.homography
        self.h_inv = np.linalg.inv(self.homography)
        if result.grid_lines is not None:
            boxes = cell_bounds_grid(result.grid_lines[0], result.grid_lines[1], DEFAULT_CELL_INSET, DEFAULT_OUTPUT_SIZE)
        else:
            boxes = cell_bounds(DEFAULT_OUTPUT_SIZE, DEFAULT_CELL_INSET)
        self.cell_boxes = boxes  # rectified-space inner boxes

        self.marks: dict[int, Mark] = {}
        self.erased: set[int] = set()
        self.pose = {"dx": 0.0, "dy": 0.0, "deg": 0.0}
        self._action: _Action | None = None
        self._pencil_at: tuple[float, float] | None = None  # None = off-screen
        self._shadow: dict | None = None
        self._hand: dict | None = None

    # -- cell helpers -----------------------------------------------------

    def cell_center_raw(self, cell: int) -> tuple[float, float]:
        x0, y0, x1, y1 = self.cell_boxes[cell]
        pt = cv2.perspectiveTransform(np.array([[[(x0 + x1) / 2, (y0 + y1) / 2]]], np.float32), self.h_inv)
        return float(pt[0, 0, 0]), float(pt[0, 0, 1])

    # -- public actions (queue one at a time; `busy` while running) -------

    @property
    def busy(self) -> bool:
        return self._action is not None

    def _pose_inverse(self) -> np.ndarray:
        """Inverse of the current page-bump affine. New ink is placed
        through it: marks are rendered into the world BEFORE the pose
        warp (so existing ink moves with the page), and a mark meant for
        a cell must land on that cell's CURRENT position after the warp."""
        M = cv2.getRotationMatrix2D((FRAME_W / 2, FRAME_H / 2), self.pose["deg"], 1.0)
        M[0, 2] += self.pose["dx"]
        M[1, 2] += self.pose["dy"]
        return cv2.invertAffineTransform(M)

    def _place_mark(self, cell: int, symbol: str) -> Mark:
        box = self.cell_boxes[cell]
        mark = make_x(box, self.rng, self.h_inv) if symbol == "X" else make_o(box, self.rng, self.h_inv)
        if any(abs(self.pose[k]) > 1e-3 for k in ("dx", "dy", "deg")):
            inv = self._pose_inverse()
            for path in mark.paths:
                pts = np.hstack([path.astype(np.float32), np.ones((len(path), 1), np.float32)])
                path[:] = np.round((inv @ pts.T).T).astype(np.int32)
        return mark

    def draw(self, cell: int, symbol: str, draw_s: float | None = None) -> None:
        """Pencil enters from the left edge, draws the mark progressively,
        dwells a beat, exits. Symbol 'X' or 'O'."""
        mark = self._place_mark(cell, symbol)
        if draw_s is None:
            draw_s = max(0.5, mark.total / 260.0)  # ~260 px/s hand speed
        self._action = _Action("draw", PENCIL_ENTER_S + draw_s + PENCIL_DWELL_S + PENCIL_EXIT_S, {
            "cell": cell, "mark": mark, "draw_s": draw_s,
        })

    def scribble(self, cell: int, duration: float = 1.4) -> None:
        """Dense zigzag over an already-marked cell — §9's 'human draws in
        an occupied cell'. APPENDS ink (the old mark stays under it), so
        the cell's delta vs baseline is unmistakably past T_high and,
        being paper, stays there — the recovery has to live with it."""
        rng = self.rng
        x0, y0, x1, y1 = self.cell_boxes[cell]
        paths = []
        n_zig = 5
        for k in range(n_zig):
            yk = y0 + (k + 1) * (y1 - y0) / (n_zig + 1)
            paths.append(_wobble_line((x0 + 8, yk), (x1 - 8, y0 + (n_zig - k) * (y1 - y0) / (n_zig + 1)), rng))
        mark = Mark([_to_raw(p, self.h_inv) for p in paths])
        self._action = _Action("scribble", PENCIL_ENTER_S + duration + PENCIL_DWELL_S + PENCIL_EXIT_S, {
            "cell": cell, "mark": mark, "draw_s": duration,
        })

    def half_draw_then_finish(self, cell: int, symbol: str, pause_s: float = 1.4) -> None:
        """§9 'half-drawn mark, pen lifted briefly': draws ~55%, pencil
        fully leaves for `pause_s`, returns and finishes the same mark."""
        mark = self._place_mark(cell, symbol)
        self._action = _Action("half", PENCIL_ENTER_S + 0.6 + pause_s + PENCIL_ENTER_S + 0.7 + PENCIL_DWELL_S + PENCIL_EXIT_S, {
            "cell": cell, "mark": mark, "pause_s": pause_s,
        })

    def linger(self, seconds: float, over_cell: int = 4) -> None:
        """§9 'hand stays over the page': pencil hovers with tremor near a
        grid-line crossing (never inside an inner-cell box — a static
        pencil over a cell is legitimate ink), while a soft hand shadow
        breathes over the grid border so detection flickers."""
        border = self._grid_border_point(over_cell)
        self._action = _Action("linger", seconds, {"at": border})

    def bump(self, dx: float, dy: float, deg: float, duration: float = 2.2) -> None:
        """§9 'page bumped or rotated': a grabbing hand (strong, near-
        opaque shadow) shoves the page — motion smear plus covered grid
        lines kill detection for longer than P2's 1.5 s hold-over, so the
        session really does fall into BOARD_LOST; the page stays at the
        new pose afterwards (RESYNC's job to re-read it)."""
        self._action = _Action("bump", duration, {
            "from": dict(self.pose), "to": {"dx": self.pose["dx"] + dx, "dy": self.pose["dy"] + dy, "deg": self.pose["deg"] + deg},
        })

    def erase(self, cell: int, duration: float = 1.0) -> None:
        """Rub the mark in `cell` away: pencil works over the cell while
        the stroke fades back to paper (soft mask, slight residue)."""
        self._action = _Action("erase", duration, {"cell": cell, "at": self.cell_center_raw(cell)})

    def shadow(self, cell: int, band: str = "ambiguous", perceiver=None, strength: float = 0.12) -> None:
        """A soft shadow settles over an empty cell and stays (persistent
        ambiguous read, §9 'shadow or glare'). Cleared by clear_shadow().

        Corner cells only (0/2/6/8): the falloff's dark side points off
        the board, so no neighbor cell leaves baseline — a mid-board
        shadow big enough to register always spans two cells, which is a
        different (messier) scenario.

        The ambiguous band is narrow (2.5-3.5% of the inner cell), so the
        shadow edge's coverage is CONSTRUCTED, not guessed: the mask is
        built in rectified space as a soft diagonal falloff clipping the
        cell's inner corner, at a few candidate chord lengths, and each is
        measured IN CONTEXT — a Perceiver over the live scene, noise and
        converged homography included — first placement landing mid-band
        wins. Pass `perceiver` to calibrate against a specific (already
        warmed) instance; without one a fresh instance is used and the
        caller should verify the app's actual reaction (the scenario
        director strengthens the shadow in a feedback loop if needed).
        Single-frame estimates transfer badly across a band this narrow;
        in-context measurement is the same spirit as the session's own
        start-up calibration. `band='marked'` picks a clearly-dark shadow."""
        from inkwatch.perception import Perceiver

        if cell not in (0, 2, 6, 8):
            raise ValueError("shadow() supports corner cells only (0/2/6/8) — see docstring")
        x0, y0, x1, y1 = self.cell_boxes[cell]
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        corner = min(corners, key=lambda c: abs(c[0] - 300) + abs(c[1] - 300))
        target = (0.027, 0.034) if band == "ambiguous" else (0.05, 0.10)

        # outward unit diagonal at that corner (dark side faces away
        # from the board center)
        out = np.sign(np.array(corner, float) - 300.0)
        out[out == 0] = 1.0
        out /= np.hypot(*out)
        w = 2.5  # penumbra width, px — soft enough to look like a shadow,
        # hard enough for the cell's adaptive threshold to fire along it
        yy, xx = np.mgrid[0:DEFAULT_OUTPUT_SIZE, 0:DEFAULT_OUTPUT_SIZE].astype(np.float32)

        # the shadow is LOCAL: the penumbra edge provides the ink, and a
        # coarse soft falloff (far too gentle to threshold-fire) keeps
        # the dark side from spilling into neighbors or the grid borders
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        coarse = np.zeros((DEFAULT_OUTPUT_SIZE, DEFAULT_OUTPUT_SIZE), np.float32)
        cv2.ellipse(coarse, (int(cx), int(cy)), (int((x1 - x0) * 1.5), int((y1 - y0) * 1.5)), 0, 0, 360, 1.0, -1)
        coarse = cv2.GaussianBlur(coarse, (61, 61), 0)
        coarse /= coarse.max()

        def in_context_deltas(mask: np.ndarray) -> np.ndarray:
            """Median per-cell ink delta of this shadow, measured through
            a Perceiver on the LIVE scene (noise, converged homography) —
            single-frame fresh-tracker estimates transfer badly across
            the narrow ambiguous band."""
            local = perceiver if perceiver is not None else Perceiver()
            self._shadow = None
            for _ in range(8):
                obs = local.observe(self.tick(), None)
            base = np.array(obs.ratios)
            try:
                self._shadow = {"mask": mask, "strength": 0.12}
                samples = []
                for _ in range(14):
                    obs = local.observe(self.tick(), None)
                    samples.append(np.array(obs.ratios))
            finally:
                self._shadow = None
            return np.median(np.stack(samples), axis=0) - base

        for cut in (0.6, 0.8, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2):
            chord = cut * (x1 - x0)
            # sigmoid edge along the outward diagonal through the corner
            dist = (xx - corner[0]) * out[0] + (yy - corner[1]) * out[1]
            mask_rect = 1.0 / (1.0 + np.exp(-(dist - chord * 0.5) / w))
            mask_rect = mask_rect * coarse
            mask = cv2.warpPerspective(mask_rect, self.h_inv, (FRAME_W, FRAME_H))
            deltas = in_context_deltas(mask)
            # clean read: the target cell is in band AND no other cell
            # leaves its baseline (a shadow spanning two cells is a
            # different, messier scenario)
            if target[0] <= deltas[cell] <= target[1] and np.max(np.delete(deltas, cell)) < 0.024:
                self._shadow = {"mask": mask, "strength": strength}
                return
        raise RuntimeError(f"no in-band shadow placement found for cell {cell}")

    def clear_shadow(self) -> None:
        self._shadow = None

    def set_shadow_strength(self, strength: float) -> None:
        """Deepen the EXISTING shadow in place (same geometry, stronger
        darkening). See shadow_ramp for why this exists."""
        if self._shadow is not None:
            self._shadow["strength"] = strength

    def shadow_ramp(self, cell: int, s0: float = 0.12, s1: float = 0.18, duration: float = 20.0) -> None:
        """Calibrate the shadow geometry once, then deepen it CONTINUOUSLY
        over `duration` seconds. The ambiguous band is ~1% wide with a
        ±1-2% sim↔app transfer gap, so discrete strength steps jump over
        it (measured live: nothing at 0.13, committed as a full mark at
        0.14); a slow ramp sweeps through it, giving the app's ambiguous
        streak seconds to fire before the shadow reads as ink."""
        self.shadow(cell, strength=s0)
        self._action = _Action("ramp", duration, {"s0": s0, "s1": s1})

    def stop_ramp(self) -> None:
        """Freeze the ramp where it is (the shadow stays at its current
        strength — the app is already asking about it)."""
        if self._action is not None and self._action.kind == "ramp":
            self._action = None

    # -- clock ------------------------------------------------------------

    def tick(self, dt: float = 1 / 30) -> np.ndarray:
        if self._action is not None:
            self._action.t += dt
            self._step(self._action)
            if self._action.done:
                self._finish(self._action)
                self._action = None
        else:
            self._pencil_at = None
        return self._render()

    def run_for(self, seconds: float) -> None:
        """Advance without caring about frames (the caller still ticks
        the server; this helper only exists for tests)."""
        for _ in range(int(seconds * 30)):
            self.tick()

    # -- action stepping ---------------------------------------------------

    def _step(self, action: _Action) -> None:
        k, d, t = action.kind, action.data, action.t
        if k in ("draw", "scribble"):
            enter, draw_s, dwell, exit_ = PENCIL_ENTER_S, d["draw_s"], PENCIL_DWELL_S, PENCIL_EXIT_S
            if t < enter:
                self._pencil_at = self._enter_from_left(d["mark"].head(0.0), t / enter)
            elif t < enter + draw_s:
                frac = (t - enter) / draw_s
                d["frac"] = frac
                self._pencil_at = d["mark"].head(frac)
            elif t < enter + draw_s + dwell:
                d["frac"] = 1.0
                self._pencil_at = d["mark"].head(1.0)
            else:
                self._pencil_at = self._exit_to_left(d["mark"].head(1.0), (t - enter - draw_s - dwell) / exit_)
        elif k == "half":
            pause = d["pause_s"]
            enter, seg1, seg2, dwell, exit_ = PENCIL_ENTER_S, 0.6, 0.7, PENCIL_DWELL_S, PENCIL_EXIT_S
            t2 = t - enter
            if t < enter:
                self._pencil_at = self._enter_from_left(d["mark"].head(0.0), t / enter)
            elif t2 < seg1:
                d["frac"] = 0.55 * (t2 / seg1)
                self._pencil_at = d["mark"].head(d["frac"])
            elif t2 < seg1 + pause:
                d["frac"] = 0.55
                gone = min(1.0, (t2 - seg1) / 0.25, (seg1 + pause - t2) / 0.25)
                self._pencil_at = self._exit_to_left(d["mark"].head(0.55), gone)
            elif t2 < seg1 + pause + enter:
                self._pencil_at = self._enter_from_left(d["mark"].head(0.55), (t2 - seg1 - pause) / enter)
            elif t2 < seg1 + pause + enter + seg2:
                d["frac"] = 0.55 + 0.45 * ((t2 - seg1 - pause - enter) / seg2)
                self._pencil_at = d["mark"].head(d["frac"])
            elif t2 < seg1 + pause + enter + seg2 + dwell:
                d["frac"] = 1.0
                self._pencil_at = d["mark"].head(1.0)
            else:
                self._pencil_at = self._exit_to_left(d["mark"].head(1.0), (t2 - seg1 - pause - enter - seg2 - dwell) / exit_)
        elif k == "linger":
            cx, cy = d["at"]
            a = 2 * math.pi * t / 1.7
            self._pencil_at = (cx + 6.0 * math.sin(a), cy + 4.0 * math.sin(2 * a + 1.3))
            # the resting hand itself: an opaque blob parked over the
            # grid's inner lines near the pencil — line contrast under it
            # dies, so detection drops (P6 occluded) for the whole stay
            center = (cx + 20 * math.sin(a / 2.3), cy + 14 * math.cos(a / 3.1))
            mask = _soft_blob_mask((FRAME_H, FRAME_W), center, (175, 150), rng=self.rng)
            self._hand = {"mask": mask, "opacity": 0.85}
        elif k == "bump":
            frac = min(1.0, t / action.duration)
            ease = frac * frac * (3 - 2 * frac)
            f, to = d["from"], d["to"]
            for key in ("dx", "dy", "deg"):
                self.pose[key] = f[key] + (to[key] - f[key]) * ease
            # the grabbing hand: an opaque palm over the page for the
            # whole shove — grid lines under it lose all contrast
            mask = _soft_blob_mask((FRAME_H, FRAME_W), (FRAME_W * 0.45, FRAME_H * 0.5), (330, 310), rng=self.rng)
            self._hand = {"mask": mask, "opacity": 0.88}
        elif k == "erase":
            at = d["at"]
            wob = 14 * math.sin(2 * math.pi * t / 0.28)
            self._pencil_at = (at[0] + wob, at[1] + 6 * math.cos(2 * math.pi * t / 0.21))
            d["erase_frac"] = min(1.0, t / (action.duration * 0.8))
        elif k == "ramp":
            frac = min(1.0, t / action.duration)
            self.set_shadow_strength(d["s0"] + (d["s1"] - d["s0"]) * frac)

    def _finish(self, action: _Action) -> None:
        k, d = action.kind, action.data
        if k in ("draw", "half"):
            self.marks[d["cell"]] = d["mark"]
            self.erased.discard(d["cell"])
        elif k == "scribble":
            cell = d["cell"]
            if cell in self.marks:
                self.marks[cell].paths.extend(d["mark"].paths)
                self.marks[cell].lengths.extend(d["mark"].lengths)
                self.marks[cell].total += d["mark"].total
            else:
                self.marks[cell] = d["mark"]
            self.erased.discard(cell)
        elif k == "erase":
            cell = d["cell"]
            self.marks.pop(cell, None)
            self.erased.add(cell)
        if k in ("linger", "bump"):
            self._shadow = None
            self._hand = None
        self._pencil_at = None

    # -- pencil motion helpers --------------------------------------------

    def _enter_from_left(self, target: tuple[float, float], frac: float) -> tuple[float, float]:
        frac = min(1.0, max(0.05, frac))
        ease = 1 - (1 - frac) ** 2
        start_x = -self.pencil_sprite.shape[1] * 0.6
        return (start_x + (target[0] - start_x) * ease, target[1])

    def _exit_to_left(self, start: tuple[float, float], frac: float) -> tuple[float, float]:
        frac = min(1.0, max(0.0, frac))
        ease = frac * frac
        end_x = -self.pencil_sprite.shape[1] * 0.6
        return (start[0] + (end_x - start[0]) * ease, start[1])

    def _grid_border_point(self, near_cell: int) -> tuple[float, float]:
        """A point ON a grid line near the cell — hover target for
        'lingering hand', outside every inset inner-cell box."""
        x0, y0, x1, y1 = self.cell_boxes[near_cell]
        # the inner box's top-left corner is inset ~15% from the real
        # crossing; stepping back lands on the crossing itself
        pt_rect = (x0 - (x1 - x0) * 0.5, y0 - (y1 - y0) * 0.5)
        pt = cv2.perspectiveTransform(np.array([[pt_rect]], np.float32), self.h_inv)
        return float(pt[0, 0, 0]), float(pt[0, 0, 1])

    # -- rendering ----------------------------------------------------------

    def _render(self) -> np.ndarray:
        world = self.bg.copy()
        for cell, mark in self.marks.items():
            mark.draw(world, 1.0)
        action = self._action
        if action is not None and action.kind in ("draw", "half", "scribble") and action.data.get("frac"):
            action.data["mark"].draw(world, action.data["frac"])
        if action is not None and action.kind == "erase" and action.data.get("erase_frac"):
            cell = action.data["cell"]
            mark = self.marks.get(cell)
            if mark is not None:
                mark.draw(world, 1.0 - action.data["erase_frac"])

        # pose (page bump) — world warp, replicate edges (desk beyond page)
        if any(abs(self.pose[k]) > 1e-3 for k in ("dx", "dy", "deg")):
            M = cv2.getRotationMatrix2D((FRAME_W / 2, FRAME_H / 2), self.pose["deg"], 1.0)
            M[0, 2] += self.pose["dx"]
            M[1, 2] += self.pose["dy"]
            world = cv2.warpAffine(world, M, (FRAME_W, FRAME_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if self._action is not None and self._action.kind == "bump" and not self._action.done:
                # motion smear of a real grab-and-shove: destroys the
                # line profiles detection relies on (measured: 0/28
                # frames detected at k=25 through the real pipeline)
                world = cv2.GaussianBlur(world, (31, 31), 0)

        frame = world
        if self._shadow is not None:
            s = self._shadow
            frame = (frame.astype(np.float32) * (1.0 - s["strength"] * s["mask"][..., None])).astype(np.uint8)
        if self._hand is not None:
            frame = _apply_hand(frame, self._hand["mask"], self._hand["opacity"])
        if self._pencil_at is not None:
            _alpha_sprite(frame, self.pencil_sprite, self.pencil_tip, self._pencil_at)

        noise = self.noise_rng.normal(0, NOISE_SIGMA, frame.shape).astype(np.float32)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ------------------------------------------------------------- asset extract


def extract_assets(session_dir: Path, out_dir: Path) -> None:
    """Pull the sim's real-world assets out of a `--record`ed session:
    median empty-board background + the segmented pencil sprite with its
    graphite-tip anchor. The empty stretch is located by the event log
    (frames well before the first commit), verified low-ink by measure."""
    from inkwatch.perception import measure_cells

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(line) for line in (session_dir / "manifest.jsonl").open() if line.strip()]
    events = [json.loads(line) for line in (session_dir / "events.jsonl").open() if line.strip()]
    first_commit = next((e["frame_ts"] for e in events if e["type"] == "commit"), None)
    if first_commit is None:
        raise RuntimeError("session has no commit — cannot find the empty-board stretch")

    candidates = [m for m in manifest if m["frame_ts"] < first_commit - 4.0][-90:]
    imgs = []
    for entry in candidates:
        img = cv2.imread(str(session_dir / entry["path"]))
        if img is not None:
            imgs.append(img)
    if len(imgs) < 10:
        raise RuntimeError("not enough empty-board frames before the first commit")
    bg = np.median(np.stack(imgs), axis=0).astype(np.uint8)
    ratios = measure_cells(bg)
    if max(ratios) > 0.06:
        raise RuntimeError(f"extracted background is not blank (max cell ink {max(ratios):.3f})")
    cv2.imwrite(str(out_dir / "paper_bg.png"), bg)

    # pencil: the most saturated connected component in a drawing frame
    # (frames right after the first commit show the pencil at work)
    pencil_img = None
    for entry in reversed([m for m in manifest if first_commit + 2.0 < m["frame_ts"] < first_commit + 30.0]):
        img = cv2.imread(str(session_dir / entry["path"]))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if (hsv[..., 1] > 55).sum() > 3000:
            pencil_img = img
            break
    if pencil_img is None:
        raise RuntimeError("no frame with a saturated pencil found — extract the sprite by hand")

    hsv = cv2.cvtColor(pencil_img, cv2.COLOR_BGR2HSV)
    body = (hsv[..., 1] > 55).astype(np.uint8) * 255
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(body)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pencil = (labels == big).astype(np.uint8)
    dark = (hsv[..., 2] < 110).astype(np.uint8)
    near = cv2.dilate(pencil, np.ones((15, 15), np.uint8))
    mask = ((pencil | (dark & near)) > 0).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ys, xs = np.where(mask > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    rgba = np.dstack([pencil_img, mask])[y0 : y1 + 1, x0 : x1 + 1]
    cv2.imwrite(str(out_dir / "pencil.png"), rgba)
    gray = cv2.cvtColor(pencil_img, cv2.COLOR_BGR2GRAY)
    masked = np.where(mask > 0, gray, 255)
    ty, tx = np.unravel_index(np.argmin(masked), masked.shape)
    (out_dir / "pencil_tip.txt").write_text(f"{tx - x0},{ty - y0}")
    print(f"extracted: paper_bg.png {bg.shape}, pencil.png {rgba.shape}, tip {(tx - x0, ty - y0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", metavar="SESSION_DIR", help="pull sim assets from a recorded session into recordings/sim_assets/")
    parser.add_argument("--out", default="recordings/sim_assets")
    args = parser.parse_args()
    if args.extract:
        extract_assets(Path(args.extract), Path(args.out))


if __name__ == "__main__":
    main()
