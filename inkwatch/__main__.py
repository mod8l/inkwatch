"""CLI entry point: wires perception, the session state machine, decision,
and output into the live camera loop (M3, "playable loop... happy path
only, no escalation"). `session.py`'s module docstring lists what's still
open for M4/M5 — this loop doesn't add any recovery of its own.

Needs a real camera to run, so it can't be exercised by an automated test;
see README.md for exactly what to try and what you should see.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

from inkwatch.output import Speaker, draw_overlay
from inkwatch.perception import (
    DEFAULT_CELL_INSET,
    DEFAULT_INK_HIGH,
    DEFAULT_INK_LOW,
    DEFAULT_MOTION_THRESHOLD,
    DEFAULT_STABILITY_FRAMES,
    Perceiver,
    StabilityGate,
)
from inkwatch.session import Session

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=None, help="Camera index or stream URL (overrides config.yaml)")
    parser.add_argument("--agent-first", action="store_true", default=None, help="Agent plays X and opens")
    parser.add_argument("--no-voice", action="store_true", help="Display only, no speech")
    parser.add_argument(
        "--no-escalation", action="store_true",
        help="Never call the vision model (the only mode M3 supports anyway — escalation is M5)",
    )
    parser.add_argument("--record", action="store_true", help="Save the raw stream for replay (not yet built, M5)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(Path(args.config))

    camera = args.camera if args.camera is not None else config.get("camera", 0)
    if isinstance(camera, str) and camera.isdigit():
        camera = int(camera)
    agent_first = args.agent_first if args.agent_first is not None else config.get("agent_first", False)
    voice = config.get("voice", True) and not args.no_voice

    if args.record:
        print("--record isn't built yet (M5); continuing without it.", file=sys.stderr)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Could not open camera '{camera}'", file=sys.stderr)
        sys.exit(1)

    perceiver = Perceiver(
        stability=StabilityGate(
            stability_frames=config.get("stability_frames", DEFAULT_STABILITY_FRAMES),
            motion_threshold=config.get("motion_threshold", DEFAULT_MOTION_THRESHOLD),
        ),
        cell_inset=config.get("cell_inset", DEFAULT_CELL_INSET),
    )
    ink_low = config.get("ink_threshold_low", DEFAULT_INK_LOW)
    ink_high = config.get("ink_threshold_high", DEFAULT_INK_HIGH)
    session = Session(agent_first=agent_first, reminder_s=tuple(config.get("reminder_s", (10, 20))))
    speaker = Speaker(enabled=voice)
    debug = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed", file=sys.stderr)
                break

            now = time.monotonic()
            observation = perceiver.observe(frame, session.baseline, now=now, low=ink_low, high=ink_high)
            result = session.update(observation, now)

            if result.message:
                print(result.message)
                speaker.say(result.message)

            if perceiver.last_rectified is not None:
                rectified = perceiver.last_rectified.copy()
                draw_overlay(
                    rectified,
                    board=result.board,
                    phase=result.phase.value,
                    message=result.message,
                    target_cell=result.target_cell,
                    confidence=result.confidence,
                    cell_marks=result.cell_marks,
                    debug=debug,
                )
                cv2.imshow("inkwatch", rectified)
            else:
                banner = frame.copy()
                text = "Board not found — " + (
                    ", ".join(observation.missing_corners) or "no markers"
                )
                cv2.putText(banner, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("inkwatch", banner)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug = not debug
    finally:
        speaker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
