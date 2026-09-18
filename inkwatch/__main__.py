"""CLI entry point: wires perception, the session state machine,
escalation, logging, and output into the live camera loop. `session.py`
owns all of §9's recovery behavior (M4); this loop's own recovery job is
narrower — the camera itself going away (§9 "Camera disconnects"), which
happens below the perception layer and so isn't something an
`Observation` can carry.

Per-frame perception -> session (-> escalation) is `replay.py`'s
`run_frame`, not duplicated here, so the live loop and the offline replay
pipeline (L4) can never drift into two different implementations of the
same step.

Needs a real camera to run, so it can't be exercised by an automated test;
see README.md for exactly what to try and what you should see.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

from inkwatch.escalation import DEFAULT_MAX_CALLS_PER_GAME, DEFAULT_MODEL, DEFAULT_TIMEOUT_S, Escalator
from inkwatch.events import Observation, SessionLogger
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
from inkwatch.replay import run_frame
from inkwatch.session import DEFAULT_OCCLUSION_REMINDER_S, Phase, Session, SessionResult

CAMERA_LOST_MESSAGE_S = 2.0  # §9: "Camera disconnects... No frames for 2 s"
CAMERA_RETRY_S = 2.0  # §9: "Retry every 2 s"

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
        help="Never call the vision model, even with an API key configured; every low-confidence "
        "read asks you directly instead (§9's own behavior for no model / no key)",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Save every raw camera frame (+ a manifest) under log_dir, for `python -m inkwatch.replay` later",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    return parser.parse_args(argv)


def _new_session(config: dict, agent_first: bool, ink_low: float, ink_high: float) -> Session:
    return Session(
        agent_first=agent_first,
        reminder_s=tuple(config.get("reminder_s", (10, 20))),
        occlusion_reminder_s=config.get("occlusion_reminder_s", DEFAULT_OCCLUSION_REMINDER_S),
        ink_low=ink_low,
        ink_high=ink_high,
    )


def _make_escalator(config: dict, disabled_by_flag: bool) -> Escalator:
    cfg = config.get("escalation") or {}
    return Escalator(
        enabled=bool(cfg.get("enabled", True)) and not disabled_by_flag,
        model=cfg.get("model", DEFAULT_MODEL),
        timeout_s=float(cfg.get("timeout_s", DEFAULT_TIMEOUT_S)),
        max_calls_per_game=int(cfg.get("max_calls_per_game", DEFAULT_MAX_CALLS_PER_GAME)),
    )


def _new_session_dir(log_dir: Path) -> Path:
    """One directory per game, so each game's events.jsonl is scored on
    its own by metrics.py. Two games started within the same second would
    otherwise share a timestamp-named directory and interleave one log."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    session_dir = log_dir / stamp
    n = 2
    while session_dir.exists():
        session_dir = log_dir / f"{stamp}-{n}"
        n += 1
    return session_dir


class _Recorder:
    """L3: `--record`'s raw-frame writer — every frame perception sees,
    with a manifest `replay.py` can read back in order. A no-op when
    `--record` wasn't passed, so the hot path never touches the disk."""

    def __init__(self, session_dir: Path | None) -> None:
        self.enabled = session_dir is not None
        self._count = 0
        if not self.enabled:
            return
        self._session_dir = session_dir
        (session_dir / "raw").mkdir(parents=True, exist_ok=True)
        self._manifest = (session_dir / "manifest.jsonl").open("a", encoding="utf-8")

    def record(self, frame: np.ndarray, frame_ts: float) -> None:
        if not self.enabled:
            return
        rel_path = f"raw/{self._count:06d}.png"
        cv2.imwrite(str(self._session_dir / rel_path), frame)
        self._manifest.write(json.dumps({"frame_ts": frame_ts, "path": rel_path}) + "\n")
        self._manifest.flush()
        self._count += 1

    def close(self) -> None:
        if self.enabled:
            self._manifest.close()


def _save_frame(session_dir: Path, rectified: np.ndarray, tag: str, frame_ts: float) -> str:
    """L2: a snapshot at a commit/escalation/question moment. Returns the
    path (relative to `session_dir`) for the log record to reference."""
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"frames/{tag}_{frame_ts:.3f}.png"
    cv2.imwrite(str(session_dir / rel_path), rectified)
    return rel_path


def _log_tick(
    logger: SessionLogger,
    session_dir: Path,
    rectified: np.ndarray | None,
    prev: SessionResult | None,
    result: SessionResult,
    frame_ts: float,
) -> None:
    """L1: one JSONL line per frame that changed something worth keeping
    -- a commit (board changed), a question (just entered ASK_HUMAN), or
    the final result (GAME_OVER). Escalations are logged separately, right
    where `run_frame` returns the real `EscalationOutcome` (latency,
    cost) — by the time a result gets here, a resolved escalation has
    already moved `phase` past `ESCALATE`, so this function would never
    actually see it. Quiet, unchanged frames while just waiting aren't
    logged; there can be thousands of those in a real game and they carry
    no information the transitions around them don't already capture.

    The move that ends the game changes the board AND enters GAME_OVER on
    the same tick; it logs both a "commit" and a "result" line. Without
    the commit line, metrics.py's commit-based accuracy never counts the
    winning move (every game's last ply would score as missed)."""
    board_changed = prev is not None and result.board != prev.board
    phase_changed = prev is not None and result.phase != prev.phase
    first_tick = prev is None
    entered_question = phase_changed and result.phase is Phase.ASK_HUMAN
    entered_game_over = phase_changed and result.phase is Phase.GAME_OVER
    if not (first_tick or board_changed or entered_question or entered_game_over):
        return

    def _write(event_type: str) -> None:
        frame_path = None
        if rectified is not None:
            frame_path = _save_frame(session_dir, rectified, event_type, frame_ts)
        logger.log(
            event_type,
            frame_ts=frame_ts,
            phase=result.phase,
            turn=result.turn,
            board=result.board,
            message=result.message,
            confidence=result.confidence,
            target_cell=result.target_cell,
            frame_path=frame_path,
        )

    if first_tick:
        _write("start")
    if board_changed:
        _write("commit")
    if entered_question:
        _write("question")
    if entered_game_over:
        _write("result")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()  # CLAUDE.md: OPENROUTER_API_KEY only ever comes from .env
    args = parse_args(argv)
    config = load_config(Path(args.config))

    camera = args.camera if args.camera is not None else config.get("camera", 0)
    if isinstance(camera, str) and camera.isdigit():
        camera = int(camera)
    agent_first = args.agent_first if args.agent_first is not None else config.get("agent_first", False)
    voice = config.get("voice", True) and not args.no_voice

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
    session = _new_session(config, agent_first, ink_low, ink_high)
    escalator = _make_escalator(config, disabled_by_flag=args.no_escalation)
    speaker = Speaker(enabled=voice)

    log_dir = Path(config.get("log_dir", "sessions/"))
    session_dir = _new_session_dir(log_dir)
    print(f"Logging this game to {session_dir}")
    logger = SessionLogger(session_dir, "events", enabled=True)
    recorder = _Recorder(session_dir if args.record else None)

    debug = False
    camera_lost_since: float | None = None
    camera_lost_announced = False
    prev_result: SessionResult | None = None

    try:
        while True:
            ok, frame = cap.read()
            now = time.monotonic()

            if not ok:
                # Below the perception layer entirely — there's no frame to
                # build an Observation from, so BOARD_LOST is driven with a
                # synthetic not-found one (§9: resumes via RESYNC once
                # frames come back, same path as a bumped page).
                if camera_lost_since is None:
                    camera_lost_since = now
                    camera_lost_announced = False
                elif not camera_lost_announced and now - camera_lost_since >= CAMERA_LOST_MESSAGE_S:
                    message = "I've lost the camera."
                    print(message, file=sys.stderr)
                    speaker.say(message)
                    camera_lost_announced = True

                session.update(
                    Observation(
                        frame_ts=now, found=False, stable=False, occluded=True,
                        missing_corners=(), ratios=None, cell_marks=None,
                    ),
                    now,
                )

                banner = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(
                    banner, "Camera disconnected -- retrying...", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                )
                cv2.imshow("inkwatch", banner)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                time.sleep(CAMERA_RETRY_S)
                # A dead capture doesn't hot-plug on read(): re-reading the
                # same VideoCapture after a real unplug fails forever.
                # Release and re-open, or §9's "retry every 2 s" never
                # actually recovers.
                cap.release()
                cap = cv2.VideoCapture(camera)
                continue

            camera_lost_since = None
            camera_lost_announced = False
            recorder.record(frame, now)

            observation, result, escalation_outcome = run_frame(
                perceiver, session, frame, now,
                escalator=escalator, ink_low=ink_low, ink_high=ink_high,
            )
            if escalation_outcome is not None:
                logger.log("escalation", frame_ts=now, **vars(escalation_outcome))

            if result.message:
                print(result.message)
                speaker.say(result.message)

            if perceiver.last_rectified is not None:
                rectified = perceiver.last_rectified.copy()
                _log_tick(logger, session_dir, rectified, prev_result, result, now)
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
                _log_tick(logger, session_dir, None, prev_result, result, now)
                banner = frame.copy()
                text = "Board not found — " + (
                    ", ".join(observation.missing_corners) or "no markers"
                )
                cv2.putText(banner, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("inkwatch", banner)

            prev_result = result

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("r"):
                session.force_resync()
            if key == ord("n"):
                # New game means a fresh Session AND a fresh per-game
                # escalation budget, event log, and recording — reusing
                # any of them silently merges two games into one metrics
                # view (and leaves every later game with no model calls
                # once the first game spent the budget).
                logger.close()
                recorder.close()
                escalator.close()
                session = _new_session(config, agent_first, ink_low, ink_high)
                escalator = _make_escalator(config, disabled_by_flag=args.no_escalation)
                session_dir = _new_session_dir(log_dir)
                print(f"Logging this game to {session_dir}")
                logger = SessionLogger(session_dir, "events", enabled=True)
                recorder = _Recorder(session_dir if args.record else None)
                prev_result = None
    finally:
        speaker.close()
        escalator.close()
        logger.close()
        recorder.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
