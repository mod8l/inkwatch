"""Tests for board rectification (PRODUCT.md P1, P2), per-cell ink
measurement (P3, P4, D1), and stability/observation (P5, P6).

Builds synthetic scenes and rectified boards with known geometry and
known ink coverage, so the expected result is known without a camera.
This is the "replay test" the milestones ask for.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from inkwatch.perception import (
    BoardTracker,
    CORNER_ROLES,
    Perceiver,
    StabilityGate,
    cell_bounds,
    classify_cell,
    classify_cells,
    compute_homography,
    ink_ratio,
    measure_cells,
)

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

FRAME_SIZE = 900
BOARD_SIDE = 500
MARKER_SIDE = 60
BOARD_ORIGIN = (150, 150)  # (x, y) top-left of the board square in the frame


def _marker_positions() -> dict[str, tuple[int, int]]:
    bx, by = BOARD_ORIGIN
    return {
        "top_left": (bx, by),
        "top_right": (bx + BOARD_SIDE - MARKER_SIDE, by),
        "bottom_right": (bx + BOARD_SIDE - MARKER_SIDE, by + BOARD_SIDE - MARKER_SIDE),
        "bottom_left": (bx, by + BOARD_SIDE - MARKER_SIDE),
    }


def make_synthetic_frame(*, skew: bool = False, omit: set[str] | None = None) -> np.ndarray:
    """A white frame with the four corner markers painted at known spots.

    `skew` applies a perspective warp to the whole frame, so a correct
    rectification still has to undo it. `omit` drops named corners to
    simulate occlusion.
    """
    omit = omit or set()
    frame = np.full((FRAME_SIZE, FRAME_SIZE, 3), 255, dtype=np.uint8)

    marker_ids = {"top_left": 0, "top_right": 1, "bottom_right": 2, "bottom_left": 3}
    for name, (x, y) in _marker_positions().items():
        if name in omit:
            continue
        marker_img = cv2.aruco.generateImageMarker(DICTIONARY, marker_ids[name], MARKER_SIDE)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        frame[y : y + MARKER_SIDE, x : x + MARKER_SIDE] = marker_bgr

    if skew:
        src = np.array(
            [[0, 0], [FRAME_SIZE, 0], [FRAME_SIZE, FRAME_SIZE], [0, FRAME_SIZE]],
            dtype=np.float32,
        )
        dst = np.array(
            [[60, 40], [FRAME_SIZE - 20, 10], [FRAME_SIZE - 60, FRAME_SIZE - 30], [30, FRAME_SIZE - 50]],
            dtype=np.float32,
        )
        warp = cv2.getPerspectiveTransform(src, dst)
        frame = cv2.warpPerspective(frame, warp, (FRAME_SIZE, FRAME_SIZE), borderValue=(255, 255, 255))

    return frame


def test_all_four_markers_found_rectifies_to_requested_size():
    tracker = BoardTracker(output_size=600)
    frame = make_synthetic_frame()

    result = tracker.update(frame, now=0.0)

    assert result.found
    assert not result.reused
    assert result.missing_corners == []
    assert result.rectified.shape[:2] == (600, 600)


def test_rectification_undoes_perspective_skew():
    tracker = BoardTracker(output_size=600)
    frame = make_synthetic_frame(skew=True)

    result = tracker.update(frame, now=0.0)

    assert result.found
    # The rectified crop should be (close to) blank inside the grid: no
    # skew artifacts smeared across it. Sample the center, away from grid
    # lines and markers.
    center = result.rectified[280:320, 280:320]
    assert center.mean() > 200  # mostly white


def test_missing_corner_marker_with_no_prior_homography_is_not_found():
    tracker = BoardTracker(output_size=600)
    frame = make_synthetic_frame(omit={"bottom_right"})

    result = tracker.update(frame, now=0.0)

    assert not result.found
    assert result.rectified is None
    assert "bottom_right" in result.missing_corners


def test_homography_is_held_over_within_hold_window_then_lost():
    tracker = BoardTracker(output_size=600, hold_seconds=0.5)
    good_frame = make_synthetic_frame()
    occluded_frame = make_synthetic_frame(omit={"top_left"})

    first = tracker.update(good_frame, now=0.0)
    assert first.found and not first.reused

    within_hold = tracker.update(occluded_frame, now=0.3)
    assert within_hold.found
    assert within_hold.reused

    past_hold = tracker.update(occluded_frame, now=0.6)
    assert not past_hold.found


def test_compute_homography_uses_the_board_facing_marker_corner():
    frame = make_synthetic_frame()
    corners, ids, _ = cv2.aruco.ArucoDetector(DICTIONARY, cv2.aruco.DetectorParameters()).detectMarkers(frame)
    detected = {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.flatten())}

    homography, missing = compute_homography(detected, output_size=600)

    assert missing == []
    # The top-left marker's inward (bottom-right) corner should map close to (0, 0).
    tl_marker_id, tl_corner_idx = CORNER_ROLES["top_left"]
    inward_point = detected[tl_marker_id][tl_corner_idx]
    mapped = cv2.perspectiveTransform(inward_point.reshape(1, 1, 2), homography)[0, 0]
    assert mapped == pytest.approx([0, 0], abs=2.0)


# --- Per-cell ink measurement (P3, P4, D1) -------------------------------

BOARD_SIZE = 600


def make_synthetic_board(
    size: int = BOARD_SIZE, marks: dict[int, float] | None = None
) -> np.ndarray:
    """A blank rectified board (grid lines only) with optional ink.

    `marks` maps a cell index (0-8, row-major) to the fraction of that
    cell's *inner* (inset) area to fill with a solid black square,
    centered in the cell, standing in for a hand-drawn mark of known
    coverage.
    """
    board = np.full((size, size, 3), 255, dtype=np.uint8)
    cell = size // 3
    for i in range(1, 3):
        cv2.line(board, (i * cell, 0), (i * cell, size), (0, 0, 0), 2)
        cv2.line(board, (0, i * cell), (size, i * cell), (0, 0, 0), 2)

    bounds = cell_bounds(size)
    for idx, coverage in (marks or {}).items():
        x0, y0, x1, y1 = bounds[idx]
        side_frac = coverage**0.5
        fill_w = int(round((x1 - x0) * side_frac))
        fill_h = int(round((y1 - y0) * side_frac))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        cv2.rectangle(
            board,
            (cx - fill_w // 2, cy - fill_h // 2),
            (cx + fill_w // 2, cy + fill_h // 2),
            (0, 0, 0),
            -1,
        )
    return board


def test_blank_board_reads_near_zero_ink_in_every_cell():
    board = make_synthetic_board()

    ratios = measure_cells(board)

    assert len(ratios) == 9
    assert all(r < 0.01 for r in ratios)


def test_cell_bounds_are_inset_and_row_major():
    bounds = cell_bounds(600, inset=0.15)

    assert len(bounds) == 9
    # Cell 0 is top-left; margin should shrink it in from the 0..200 raw cell.
    x0, y0, x1, y1 = bounds[0]
    assert x0 == pytest.approx(30, abs=1)
    assert y0 == pytest.approx(30, abs=1)
    assert x1 == pytest.approx(170, abs=1)
    assert y1 == pytest.approx(170, abs=1)
    # Cell 4 is the center cell.
    cx0, cy0, cx1, cy1 = bounds[4]
    assert cx0 == pytest.approx(230, abs=1)
    assert cy0 == pytest.approx(230, abs=1)


def test_grid_lines_alone_do_not_register_as_ink():
    board = make_synthetic_board()  # grid only, no marks

    ratios = measure_cells(board, inset=0.15)

    assert max(ratios) < 0.01


def test_heavy_mark_classified_as_marked_others_unaffected():
    board = make_synthetic_board(marks={4: 0.6})

    ratios = measure_cells(board)
    marks = classify_cells(ratios, baseline=[0.0] * 9)

    assert marks[4] == "marked"
    assert all(m == "none" for i, m in enumerate(marks) if i != 4)


def test_classification_uses_delta_from_baseline_not_raw_ratio():
    """A cell whose baseline already carried that much ink (e.g. a
    committed mark) should not re-trigger just because its raw ratio is
    above the threshold (D1: it's the *delta* that's classified)."""
    board = make_synthetic_board(marks={0: 0.5})
    ratios = measure_cells(board)
    baseline = list(ratios)  # baseline already matches current ink exactly

    marks = classify_cells(ratios, baseline)

    assert all(m == "none" for m in marks)


def test_x_shaped_stroke_is_classified_marked():
    """A more realistic mark than a filled square: two diagonal strokes,
    like a hand-drawn X, still reads as ink."""
    board = make_synthetic_board()
    x0, y0, x1, y1 = cell_bounds(BOARD_SIZE)[4]
    cv2.line(board, (x0, y0), (x1, y1), (0, 0, 0), 6)
    cv2.line(board, (x1, y0), (x0, y1), (0, 0, 0), 6)

    ratios = measure_cells(board)
    result = classify_cell(ratios[4], baseline=0.0)

    assert result == "marked"


@pytest.mark.parametrize(
    "cell,coverage,expected",
    [
        (0, 0.0, "none"),
        (1, 0.0, "none"),
        (2, 0.005, "none"),
        (3, 0.01, "none"),
        (4, 0.015, "none"),
        (5, 0.018, "none"),
        (6, 0.025, "ambiguous"),
        (7, 0.03, "ambiguous"),
        (8, 0.035, "ambiguous"),
        (0, 0.04, "ambiguous"),
        (1, 0.045, "ambiguous"),
        (2, 0.048, "ambiguous"),
        (3, 0.06, "marked"),
        (4, 0.08, "marked"),
        (5, 0.1, "marked"),
        (6, 0.2, "marked"),
        (7, 0.3, "marked"),
        (8, 0.5, "marked"),
        (0, 0.7, "marked"),
        (1, 0.9, "marked"),
    ],
)
def test_twenty_synthetic_marks_classify_correctly(cell, coverage, expected):
    """Stands in for PRODUCT.md M2's '20 manual marks' acceptance check:
    20 marks of known coverage, spread across cells and across the
    none/ambiguous/marked bands defined by the default thresholds
    (ink_threshold_low=0.02, ink_threshold_high=0.05 in config.yaml).
    """
    board = make_synthetic_board(marks={cell: coverage} if coverage else None)

    ratios = measure_cells(board)
    result = classify_cell(ratios[cell], baseline=0.0)

    assert result == expected


def test_ink_ratio_accepts_grayscale_or_color_crop():
    board = make_synthetic_board(marks={4: 0.5})
    x0, y0, x1, y1 = cell_bounds(BOARD_SIZE)[4]
    color_crop = board[y0:y1, x0:x1]
    gray_crop = cv2.cvtColor(color_crop, cv2.COLOR_BGR2GRAY)

    assert ink_ratio(color_crop) == pytest.approx(ink_ratio(gray_crop), abs=1e-6)


# --- Stability gating (P5, P6) --------------------------------------------


def test_identical_frames_become_stable_after_n_consecutive_reads():
    gate = StabilityGate(stability_frames=3, motion_threshold=2.0)
    board = make_synthetic_board()

    results = [gate.update(board, markers_visible=True) for _ in range(3)]

    assert results[0] == (False, False)
    assert results[1] == (False, False)
    assert results[2] == (True, False)


def test_missing_markers_are_occluded_and_never_stable():
    gate = StabilityGate(stability_frames=3)
    board = make_synthetic_board()

    for _ in range(5):
        stable, occluded = gate.update(board, markers_visible=False)
        assert not stable
        assert occluded


def test_a_new_mark_between_frames_is_occluded_and_resets_the_count():
    gate = StabilityGate(stability_frames=3, motion_threshold=2.0)
    blank = make_synthetic_board()
    marked = make_synthetic_board(marks={4: 0.6})

    gate.update(blank, markers_visible=True)
    gate.update(blank, markers_visible=True)
    stable, occluded = gate.update(marked, markers_visible=True)  # motion: hand/mark appearing

    assert not stable
    assert occluded

    # Needs a fresh run of quiet frames after the disturbance.
    gate.update(marked, markers_visible=True)
    later_stable, later_occluded = gate.update(marked, markers_visible=True)
    assert not later_stable
    assert not later_occluded


# --- Perceiver: one Observation per frame (P1-P6, D1) ----------------------


def test_perceiver_reports_not_found_and_occluded_with_no_markers():
    perceiver = Perceiver()
    frame = make_synthetic_frame(omit={"bottom_right"})

    observation = perceiver.observe(frame, baseline=None, now=0.0)

    assert not observation.found
    assert observation.occluded
    assert not observation.stable
    assert observation.ratios is None
    assert observation.cell_marks is None


def test_perceiver_becomes_stable_after_repeated_quiet_frames():
    perceiver = Perceiver(stability=StabilityGate(stability_frames=3))
    frame = make_synthetic_frame()

    observations = [perceiver.observe(frame, baseline=None, now=float(i)) for i in range(3)]

    assert [o.found for o in observations] == [True, True, True]
    assert [o.stable for o in observations] == [False, False, True]
    assert all(o.ratios is not None for o in observations)
    assert all(o.cell_marks is None for o in observations)  # no baseline supplied yet


def test_perceiver_classifies_cell_marks_against_the_supplied_baseline():
    perceiver = Perceiver(stability=StabilityGate(stability_frames=1))
    frame = make_synthetic_frame()  # blank board behind the markers

    observation = perceiver.observe(frame, baseline=[0.0] * 9, now=0.0)

    assert observation.stable
    assert observation.cell_marks is not None
    assert all(mark == "none" for mark in observation.cell_marks)
