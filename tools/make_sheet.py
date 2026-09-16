"""Generate the printable Inkwatch board: a 3x3 grid with an ArUco marker
at each corner, sized so a 100%-scale print puts the markers at a known
physical size.

Usage:
    python tools/make_sheet.py [--out assets/board.png] [--dpi 200]

Prints a PNG, not a PDF (see NOTES.md, "Build-time decisions": no PDF
library is in the approved stack). Most print dialogs let you print an
image directly; the sheet includes a 1-inch ruler mark so you can verify
your printer actually used 100% scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DICTIONARY_ID = cv2.aruco.DICT_4X4_50
MARKER_IDS = {"top_left": 0, "top_right": 1, "bottom_right": 2, "bottom_left": 3}

PAGE_SIZE_IN = (8.5, 11.0)  # US Letter, (width, height)
MARGIN_IN = 0.75
TITLE_AREA_IN = 1.3  # vertical space reserved above the board for instructions
MARKER_FRACTION = 0.11  # marker side as a fraction of the board side


def render_board(dpi: int) -> np.ndarray:
    page_w = round(PAGE_SIZE_IN[0] * dpi)
    page_h = round(PAGE_SIZE_IN[1] * dpi)
    margin = round(MARGIN_IN * dpi)
    title_h = round(TITLE_AREA_IN * dpi)

    canvas = np.full((page_h, page_w), 255, dtype=np.uint8)

    board_px = page_w - 2 * margin
    marker_px = round(board_px * MARKER_FRACTION)
    board_x0 = margin
    board_y0 = title_h

    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)
    positions = {
        "top_left": (board_y0, board_x0),
        "top_right": (board_y0, board_x0 + board_px - marker_px),
        "bottom_right": (board_y0 + board_px - marker_px, board_x0 + board_px - marker_px),
        "bottom_left": (board_y0 + board_px - marker_px, board_x0),
    }
    for name, marker_id in MARKER_IDS.items():
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        y, x = positions[name]
        canvas[y : y + marker_px, x : x + marker_px] = marker_img

    # Inner grid: the square strictly between the markers' inward corners,
    # matching what inkwatch.perception.BoardTracker rectifies (P1).
    grid_x0, grid_y0 = board_x0 + marker_px, board_y0 + marker_px
    grid_side = board_px - 2 * marker_px
    line_thickness = max(2, dpi // 80)

    cv2.rectangle(
        canvas,
        (grid_x0, grid_y0),
        (grid_x0 + grid_side, grid_y0 + grid_side),
        0,
        line_thickness,
    )
    for i in (1, 2):
        x = grid_x0 + round(grid_side * i / 3)
        cv2.line(canvas, (x, grid_y0), (x, grid_y0 + grid_side), 0, line_thickness)
        y = grid_y0 + round(grid_side * i / 3)
        cv2.line(canvas, (grid_x0, y), (grid_x0 + grid_side, y), 0, line_thickness)

    _draw_title(canvas, dpi, page_w, title_h)
    _draw_ruler(canvas, dpi, margin, page_h - margin // 2)
    return canvas


def _draw_title(canvas: np.ndarray, dpi: int, page_w: int, title_h: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = dpi / 220
    cv2.putText(
        canvas, "Inkwatch board", (round(dpi * 0.3), round(title_h * 0.35)),
        font, scale * 1.1, 0, max(1, dpi // 100), cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Print at 100% scale (no 'fit to page'). Check the 1-inch mark below with a ruler.",
        (round(dpi * 0.3), round(title_h * 0.65)),
        font, scale * 0.55, 0, max(1, dpi // 150), cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Keep all four corner squares visible and uncovered while playing.",
        (round(dpi * 0.3), round(title_h * 0.85)),
        font, scale * 0.55, 0, max(1, dpi // 150), cv2.LINE_AA,
    )


def _draw_ruler(canvas: np.ndarray, dpi: int, x0: int, y: int) -> None:
    x1 = x0 + dpi  # exactly one inch at the target DPI
    cv2.line(canvas, (x0, y), (x1, y), 0, max(2, dpi // 80))
    for x in (x0, x1):
        cv2.line(canvas, (x, y - 10), (x, y + 10), 0, max(2, dpi // 80))
    cv2.putText(
        canvas, "1 inch", (x0, y - 15), cv2.FONT_HERSHEY_SIMPLEX, dpi / 400, 0, 1, cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="assets/board.png", help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=200, help="Render resolution")
    args = parser.parse_args()

    canvas = render_board(args.dpi)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"Wrote {out_path} ({canvas.shape[1]}x{canvas.shape[0]} px at {args.dpi} dpi)")


if __name__ == "__main__":
    main()
