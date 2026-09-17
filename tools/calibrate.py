"""Debug view for board rectification and ink measurement (PRODUCT.md M1, M2).

Shows the raw camera feed and the rectified top-down board side by side,
with the found/missing-corner status and each cell's live ink ratio and
none/ambiguous/marked classification overlaid on the rectified view.

Press 'b' with a blank sheet in view to snapshot the current ratios as the
baseline (P4: classification is a delta from baseline, not a raw ratio),
then draw marks and watch each cell's label. This is the manual check
M2's milestone bar asks for ("correct per-cell none/ambiguous/marked on
20 manual marks"); there is still no session (M3) to own the baseline
across a real game, so this snapshot lives only in this tool, for this
run. Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import sys

import cv2

from inkwatch.perception import (
    DEFAULT_INK_HIGH,
    DEFAULT_INK_LOW,
    BoardTracker,
    cell_bounds,
    classify_cells,
    measure_cells,
)

_LABEL_COLOR = {
    "none": (0, 200, 0),
    "ambiguous": (0, 165, 255),
    "marked": (0, 0, 255),
}


def _draw_cell_overlay(rectified, inset: float, baseline: list[float], low: float, high: float) -> None:
    ratios = measure_cells(rectified, inset=inset)
    marks = classify_cells(ratios, baseline, low, high)
    for (x0, y0, x1, y1), ratio, mark in zip(cell_bounds(rectified.shape[0], inset), ratios, marks):
        color = _LABEL_COLOR[mark]
        cv2.rectangle(rectified, (x0, y0), (x1, y1), color, 1)
        cv2.putText(
            rectified,
            f"{ratio:.3f} {mark}",
            (x0 + 4, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="0", help="Camera index or stream URL")
    parser.add_argument("--size", type=int, default=600, help="Rectified output size (px)")
    parser.add_argument(
        "--cell-inset", type=float, default=0.15, help="Fraction each cell is shrunk by (config.yaml cell_inset)"
    )
    parser.add_argument("--ink-low", type=float, default=DEFAULT_INK_LOW, help="config.yaml ink_threshold_low")
    parser.add_argument("--ink-high", type=float, default=DEFAULT_INK_HIGH, help="config.yaml ink_threshold_high")
    args = parser.parse_args()

    camera = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Could not open camera '{camera}'", file=sys.stderr)
        sys.exit(1)

    tracker = BoardTracker(output_size=args.size)
    baseline = [0.0] * 9
    baseline_captured = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed", file=sys.stderr)
            break

        result = tracker.update(frame)

        display = frame.copy()
        if result.found:
            status = "board found" + (" (held over)" if result.reused else "")
            color = (0, 200, 0) if not result.reused else (0, 165, 255)
        else:
            missing = ", ".join(result.missing_corners) or "all corners"
            status = f"BOARD_LOST — missing: {missing}"
            color = (0, 0, 255)
        cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        baseline_status = "baseline: captured" if baseline_captured else "baseline: zero (press 'b' on a blank sheet)"
        cv2.putText(display, baseline_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
        cv2.imshow("camera", display)

        if result.rectified is not None:
            rectified = result.rectified.copy()
            _draw_cell_overlay(rectified, args.cell_inset, baseline, args.ink_low, args.ink_high)
            cv2.imshow("rectified", rectified)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b") and result.rectified is not None:
            baseline = measure_cells(result.rectified, inset=args.cell_inset)
            baseline_captured = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
