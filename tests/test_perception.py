"""Tests for board rectification (PRODUCT.md P1, P2).

Builds a synthetic scene with the four corner markers placed at known
positions and pushed through a perspective warp, so the expected result
is known without a camera. This is the "replay test" M1 asks for.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from inkwatch.perception import BoardTracker, CORNER_ROLES, compute_homography

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
