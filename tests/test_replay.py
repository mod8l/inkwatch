"""Tests for the offline replay pipeline (L3, L4): a tiny synthetic
recording (raw frames + manifest.jsonl, `--record`'s own format) run
through `replay_session`/`run_frame`, without a camera. This is the
"replay test" §13.2 asks for, alongside test_perception.py's synthetic
Perceiver tests.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from inkwatch.perception import DEFAULT_OUTPUT_SIZE, Perceiver, cell_bounds
from inkwatch.replay import load_manifest, replay_session, run_frame
from inkwatch.session import Phase, Session
from tests.test_perception import BOARD_ORIGIN, BOARD_SIDE, MARKER_SIDE, make_synthetic_frame

RAW_GRID_ORIGIN = (BOARD_ORIGIN[0] + MARKER_SIDE, BOARD_ORIGIN[1] + MARKER_SIDE)
RAW_GRID_SIDE = BOARD_SIDE - 2 * MARKER_SIDE
_SCALE = RAW_GRID_SIDE / DEFAULT_OUTPUT_SIZE  # rectified-pixel -> raw-frame-pixel


def _to_raw(x: float, y: float) -> tuple[float, float]:
    return RAW_GRID_ORIGIN[0] + x * _SCALE, RAW_GRID_ORIGIN[1] + y * _SCALE


def make_raw_frame(marks: dict[int, float] | None = None) -> np.ndarray:
    """A synthetic camera frame (markers + perspective-correct grid area,
    per test_perception.py) with ink painted at the raw-frame location
    that rectifies into the given cells, so `Perceiver.observe()` reads
    it as real ink without ever touching a rectified image directly."""
    frame = make_synthetic_frame()
    bounds = cell_bounds(DEFAULT_OUTPUT_SIZE)
    for idx, coverage in (marks or {}).items():
        x0, y0, x1, y1 = bounds[idx]
        side_frac = coverage**0.5
        fill_w, fill_h = (x1 - x0) * side_frac, (y1 - y0) * side_frac
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx0, ry0 = _to_raw(cx - fill_w / 2, cy - fill_h / 2)
        rx1, ry1 = _to_raw(cx + fill_w / 2, cy + fill_h / 2)
        cv2.rectangle(frame, (int(rx0), int(ry0)), (int(rx1), int(ry1)), (0, 0, 0), -1)
    return frame


def _write_recording(session_dir, frames: list[np.ndarray]) -> None:
    """Writes a `--record`-shaped recording: raw/NNN.png + manifest.jsonl,
    one entry per frame, ts 1.0, 2.0, 3.0, ... in order."""
    raw_dir = session_dir / "raw"
    raw_dir.mkdir(parents=True)
    manifest_path = session_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for i, frame in enumerate(frames):
            rel_path = f"raw/{i:04d}.png"
            cv2.imwrite(str(session_dir / rel_path), frame)
            manifest.write(json.dumps({"frame_ts": float(i + 1), "path": rel_path}) + "\n")


# -- run_frame: the shared per-frame step -----------------------------------


def test_run_frame_matches_calling_perceiver_and_session_directly():
    frame = make_raw_frame()
    perceiver_a, session_a = Perceiver(), Session()
    perceiver_b, session_b = Perceiver(), Session()

    direct = None
    for ts in range(1, 12):
        observation = perceiver_a.observe(frame, session_a.baseline, now=float(ts))
        direct = session_a.update(observation, float(ts))

    via_run_frame = None
    for ts in range(1, 12):
        _observation, via_run_frame, _outcome = run_frame(perceiver_b, session_b, frame, float(ts))

    assert direct.phase == via_run_frame.phase == Phase.WAIT_HUMAN
    assert direct.message == via_run_frame.message


# -- load_manifest / replay_session: the offline pipeline --------------------


def test_load_manifest_reads_frames_in_order(tmp_path):
    _write_recording(tmp_path, [make_raw_frame(), make_raw_frame()])

    entries = load_manifest(tmp_path)

    assert [e["frame_ts"] for e in entries] == [1.0, 2.0]
    assert entries[0]["path"] == "raw/0000.png"


def test_replay_reaches_calibration_and_then_commits_a_move(tmp_path):
    blank = make_raw_frame()
    # A low stability_frames keeps this test's frame count small and fast
    # while exercising the exact same StabilityGate/D6 debounce logic a
    # real run does: 2 identical frames to become stable (calibrate),
    # then the new mark needs another 2 to re-stabilize plus 1 more for
    # D6's second confirming read.
    marked = make_raw_frame({0: 0.6})
    frames = [blank, blank] + [marked, marked, marked, marked]
    _write_recording(tmp_path, frames)

    records = replay_session(tmp_path, stability_frames=2, out_path=tmp_path / "replay.jsonl")

    assert len(records) == len(frames)
    assert records[-1]["board"][0] == "X"
    assert records[-1]["phase"] == Phase.WAIT_AGENT_INK.value


def test_replay_writes_a_jsonl_event_log(tmp_path):
    _write_recording(tmp_path, [make_raw_frame(), make_raw_frame()])
    out_path = tmp_path / "replay.jsonl"

    replay_session(tmp_path, stability_frames=2, out_path=out_path)

    assert out_path.exists()
    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["type"] == "replay_tick"
    assert "phase" in first and "board" in first


def test_replay_is_deterministic_across_runs(tmp_path):
    blank = make_raw_frame()
    marked = make_raw_frame({4: 0.6})
    _write_recording(tmp_path, [blank, blank, marked, marked, marked])

    first = replay_session(tmp_path, stability_frames=2)
    second = replay_session(tmp_path, stability_frames=2)

    assert first == second


def test_replay_fails_clearly_on_a_missing_recording(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path)  # no manifest.jsonl written
