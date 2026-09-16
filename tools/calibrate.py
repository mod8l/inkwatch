"""Debug view for board rectification (PRODUCT.md M1).

Shows the raw camera feed and the rectified top-down board side by side,
with the found/missing-corner status. Press 'q' to quit.

Per-cell ink ratios (M2) aren't wired in yet; this milestone only proves
the marker-detection -> homography -> warp pipeline against a real camera
and real lighting, which can't be checked without one.
"""

from __future__ import annotations

import argparse
import sys

import cv2

from inkwatch.perception import BoardTracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="0", help="Camera index or stream URL")
    parser.add_argument("--size", type=int, default=600, help="Rectified output size (px)")
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
            cv2.imshow("rectified", result.rectified)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
