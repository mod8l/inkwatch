"""Debug view for board rectification and ink measurement (PRODUCT.md M1, M2).

Shows the raw camera feed and the rectified top-down board side by side,
with the found/missing-corner status and each cell's live ink ratio
overlaid on the rectified view. Press 'q' to quit.

Ratios shown are raw per-cell ink (P3, P4), not classified against a
baseline: there is no session yet to own an accepted baseline (M3). Use
this to eyeball that a blank sheet reads near zero everywhere and that a
drawn mark visibly jumps only in its own cell, per README.md's setup check.
"""

from __future__ import annotations

import argparse
import sys

import cv2

from inkwatch.perception import BoardTracker, cell_bounds, measure_cells


def _draw_cell_overlay(rectified, inset: float) -> None:
    ratios = measure_cells(rectified, inset=inset)
    for (x0, y0, x1, y1), ratio in zip(cell_bounds(rectified.shape[0], inset), ratios):
        cv2.rectangle(rectified, (x0, y0), (x1, y1), (255, 180, 0), 1)
        cv2.putText(
            rectified,
            f"{ratio:.3f}",
            (x0 + 4, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 180, 0),
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
    args = parser.parse_args()

    camera = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Could not open camera '{camera}'", file=sys.stderr)
        sys.exit(1)

    tracker = BoardTracker(output_size=args.size)

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
        cv2.imshow("camera", display)

        if result.rectified is not None:
            rectified = result.rectified.copy()
            _draw_cell_overlay(rectified, args.cell_inset)
            cv2.imshow("rectified", rectified)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
