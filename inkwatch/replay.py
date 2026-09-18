"""Offline pipeline over a recorded session (L3, L4): replays the same
perception -> session (-> escalation, if wired in) steps `__main__.py`
runs live, reading frames from a recording instead of a camera, and
writes a fresh JSONL event log to diff against the original — how a
threshold or logic change gets checked against the same frames instead
of a new, uncontrolled game.

`run_frame` is the one piece of real per-frame logic here; `__main__.py`
imports it too, so the live loop and this offline pipeline can never
drift into two different implementations of the same step. Everything
else in this file is reading a recording (`--record`'s manifest + raw
PNG frames) and driving `run_frame` over it without a camera, a display,
or speech.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from inkwatch.escalation import Escalator
from inkwatch.events import EscalationOutcome, Observation, SessionLogger
from inkwatch.perception import (
    DEFAULT_CELL_INSET,
    DEFAULT_INK_HIGH,
    DEFAULT_INK_LOW,
    DEFAULT_MOTION_THRESHOLD,
    DEFAULT_STABILITY_FRAMES,
    Perceiver,
    StabilityGate,
)
from inkwatch.session import Phase, Session, SessionResult


def run_frame(
    perceiver: Perceiver,
    session: Session,
    frame: np.ndarray,
    now: float,
    *,
    escalator: Escalator | None = None,
    ink_low: float = DEFAULT_INK_LOW,
    ink_high: float = DEFAULT_INK_HIGH,
) -> tuple[Observation, SessionResult, EscalationOutcome | None]:
    """One frame through perception -> session, escalating inline (the
    same order `__main__.py`'s live loop uses) if this beat needs a
    vision-model read. `escalator=None` behaves exactly like M4: every
    escalation falls straight through to asking the human.

    Returns `(observation, result, escalation_outcome)`. `observation` is
    included since `__main__.py` still needs it for the "board not found"
    banner. `escalation_outcome` is `None` on every beat that didn't
    actually run a model call — `result.phase` alone can't tell a caller
    that, since a successful call already moves the phase past `ESCALATE`
    by the time this returns; L1's logging needs the real outcome
    (latency, cost) that produced that move, not just where it landed."""
    observation = perceiver.observe(frame, session.baseline, now=now, low=ink_low, high=ink_high)
    result = session.update(observation, now)
    outcome = None
    if result.phase == Phase.ESCALATE and escalator is not None and perceiver.last_rectified is not None:
        outcome = escalator.ask(perceiver.last_rectified, result.board, result.escalation_cells)
        result = session.apply_escalation(outcome, now)
    return observation, result, outcome


def load_manifest(session_dir: Path) -> list[dict]:
    """`--record`'s own format: one JSON object per raw frame, in capture
    order, `{"frame_ts": <float>, "path": "raw/000123.png"}` (path
    relative to `session_dir`)."""
    manifest_path = session_dir / "manifest.jsonl"
    with manifest_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def replay_session(
    session_dir: Path,
    *,
    agent_first: bool = False,
    ink_low: float = DEFAULT_INK_LOW,
    ink_high: float = DEFAULT_INK_HIGH,
    cell_inset: float = DEFAULT_CELL_INSET,
    stability_frames: int = DEFAULT_STABILITY_FRAMES,
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
    escalator: Escalator | None = None,
    out_path: Path | None = None,
) -> list[dict]:
    """Runs every recorded frame in `session_dir` through `run_frame` in
    order, returning (and optionally logging) one summary record per
    frame. A fresh `Session`/`Perceiver` each call, so re-running this on
    the same recording is deterministic (module the vision model, if a
    real `escalator` is passed). The perception/session parameters mirror
    `__main__.py`'s so a config or threshold change can be checked here
    against the exact same frames, per L4's own point."""
    frames = load_manifest(session_dir)
    perceiver = Perceiver(
        stability=StabilityGate(stability_frames=stability_frames, motion_threshold=motion_threshold),
        cell_inset=cell_inset,
    )
    session = Session(agent_first=agent_first, ink_low=ink_low, ink_high=ink_high)
    logger = (
        SessionLogger(out_path.parent, out_path.stem, enabled=True)
        if out_path is not None
        else None
    )

    records: list[dict] = []
    try:
        for entry in frames:
            frame = cv2.imread(str(session_dir / entry["path"]))
            if frame is None:
                continue
            _observation, result, outcome = run_frame(
                perceiver, session, frame, entry["frame_ts"],
                escalator=escalator, ink_low=ink_low, ink_high=ink_high,
            )
            record = {
                "frame_ts": entry["frame_ts"],
                "phase": result.phase.value,
                "board": result.board,
                "turn": result.turn,
                "message": result.message,
                "confidence": result.confidence,
            }
            records.append(record)
            if logger is not None:
                logger.log("replay_tick", **record)
                if outcome is not None:
                    logger.log("escalation", frame_ts=entry["frame_ts"], **vars(outcome))
    finally:
        if logger is not None:
            logger.close()

    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", help="A recorded session directory (from --record): raw/ + manifest.jsonl")
    parser.add_argument("--agent-first", action="store_true", help="Agent played X and opened, same as the original run")
    parser.add_argument("--ink-low", type=float, default=DEFAULT_INK_LOW)
    parser.add_argument("--ink-high", type=float, default=DEFAULT_INK_HIGH)
    parser.add_argument("--cell-inset", type=float, default=DEFAULT_CELL_INSET)
    parser.add_argument("--stability-frames", type=int, default=DEFAULT_STABILITY_FRAMES)
    parser.add_argument("--motion-threshold", type=float, default=DEFAULT_MOTION_THRESHOLD)
    parser.add_argument(
        "--no-escalation", action="store_true",
        help="Never call the vision model during replay, even with an API key set",
    )
    parser.add_argument(
        "--out", default=None,
        help="Where to write the replay's event log (default: <session_dir>/replay.jsonl)",
    )
    args = parser.parse_args(argv)

    load_dotenv()  # same as __main__.py: OPENROUTER_API_KEY only ever comes from .env
    session_dir = Path(args.session_dir)
    if not (session_dir / "manifest.jsonl").exists():
        print(f"No manifest.jsonl in {session_dir} — was this session run with --record?", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else session_dir / "replay.jsonl"
    escalator = Escalator(enabled=not args.no_escalation)
    try:
        records = replay_session(
            session_dir,
            agent_first=args.agent_first,
            ink_low=args.ink_low,
            ink_high=args.ink_high,
            cell_inset=args.cell_inset,
            stability_frames=args.stability_frames,
            motion_threshold=args.motion_threshold,
            escalator=escalator,
            out_path=out_path,
        )
    finally:
        escalator.close()
    print(f"Replayed {len(records)} frames -> {out_path}")


if __name__ == "__main__":
    main()
