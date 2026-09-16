"""Board location and rectification.

Finds the four ArUco corner markers, computes the homography from the
inner (board-facing) marker corners to a fixed-size top-down image, and
warps the frame. Implements PRODUCT.md P1 and P2.

Perception never mutates game state; it only ever hands back pixels and
geometry. Move detection (ink per cell) is a separate module (M2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

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
