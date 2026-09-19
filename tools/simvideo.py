"""Composite the final scenario video from the recorded simulation runs.

Per run directory (recordings/sim/<run>/, written by tools/simulate.py):
the simulated camera feed, the desktop recording with the app's window,
the audio capture (or a rebuilt espeak track), and the timeline
(captions, spoken lines, actions, verdicts). Output is one continuous
video: caption bar on top, simulated camera on the left, the unmodified
app's own window on the right, the spoken line as subtitle, verdicts
flashing as they land.

  python tools/simvideo.py                 -> recordings/inkwatch_scenarios.mp4
  python tools/simvideo.py recovery happy_path

Everything is aligned to the run's shared wall clock: the feed's
per-frame timestamps, the desktop recording's start time, the audio
capture's start time, and the timeline entries all live on the same
epoch, so the two panels and the speech stay in sync.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

SIM_ROOT = Path("recordings/sim")
OUT_PATH = Path("recordings/inkwatch_scenarios.mp4")
WORK = SIM_ROOT / "_video_work"

W, H = 1920, 1080
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FEED_POS = (40, 150, 900, 675)  # x, y, w, h
APP_POS = (1005, 150, 675, 675)
FFMPEG_STARTUP_S = 0.4  # measured process-spawn-to-first-frame slack

RUN_TITLES = {
    "happy_path": ("Happy path", "calibration → a full game → result + final re-read → auto-restart"),
    "recovery": ("Recovery gauntlet", "every §9 edge case: half-marks, scribbles, wrong cells, bumps, shadows, erasures, camera loss"),
    "escalation_live": ("Vision-model escalation", "two marks at once, resolved by the model (or the spec'd fallback)"),
    "agent_first": ("Agent plays first", "--agent-first: the agent opens as X"),
}

SILENT_DB = -45.0


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:6])}...\n{proc.stderr[-2000:]}")
    return proc


def _mean_volume_db(path: Path) -> float:
    proc = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    for line in proc.stderr.splitlines():
        if "mean_volume:" in line:
            return float(line.split("mean_volume:")[1].strip().rstrip(" dB"))
    return -99.0


def _spoken_entries(timeline: list[dict]) -> list[dict]:
    """Timeline lines that were actually spoken (printed result messages
    and the camera-lost line) — not the print-only bookkeeping lines."""
    skip = ("Logging this game to", "New page detected", "QFontDatabase", "Note that Qt", "Warning: Ignoring")
    out = []
    for entry in timeline:
        if entry.get("kind") != "app_line":
            continue
        text = entry["text"].strip()
        if text and not text.startswith(skip):
            out.append({"t": entry["t"], "text": text})
    return out


def _synth_audio(entries: list[dict], work: Path, tag: str) -> Path | None:
    """Rebuild the speech track with the same espeak engine the app uses
    (used when the live capture is missing or silent)."""
    if not entries:
        return None
    import pyttsx3

    parts = []
    engine = pyttsx3.init()
    for i, entry in enumerate(entries):
        part = work / f"{tag}_tts_{i}.wav"
        engine.save_to_file(entry["text"], str(part))
        parts.append((entry["t"], part))
    engine.runAndWait()
    parts = [(t, p) for t, p in parts if p.exists() and p.stat().st_size > 100]
    if not parts:
        return None

    inputs: list[str] = []
    filters = []
    for i, (t, part) in enumerate(parts):
        inputs += ["-i", str(part)]
        ms = max(0, int(t * 1000))
        filters.append(f"[{i}:a]aresample=48000,adelay={ms}|{ms}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(parts)))
    filters.append(f"{mix}amix=inputs={len(parts)}:normalize=0[aout]")
    out = work / f"{tag}_synth.wav"
    _run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[aout]", str(out)])
    return out


def _write_text(work: Path, tag: str, idx: int, text: str) -> Path:
    path = work / f"{tag}_txt_{idx}.txt"
    path.write_text(text)
    return path


def build_segment(run: str, run_dir: Path, work: Path) -> tuple[Path, float]:
    """One run -> one normalized 1920x1080 segment. Returns (path, duration)."""
    timeline = json.loads((run_dir / "timeline.json").read_text())
    recorders = json.loads((run_dir / "recorders.json").read_text()) if (run_dir / "recorders.json").exists() else {}
    t0_wall = recorders.get("t0_wall")

    feed_ts_path = run_dir / "feed_ts.jsonl"
    feed_off = 0.0
    if t0_wall and feed_ts_path.exists():
        first = json.loads(feed_ts_path.read_text().splitlines()[0])
        feed_off = first["t"] - t0_wall

    duration = (max(e["t"] for e in timeline) if timeline else 5.0) + 4.0

    filters: list[str] = [f"color=c=0x101418:s={W}x{H}:r=30:d={duration:.2f}[base]"]
    inputs: list[str] = ["-i", str(run_dir / "feed.mp4")]
    filters.append(f"[0:v]setpts=PTS-STARTPTS+{feed_off:.3f}/TB,fps=30,scale={FEED_POS[2]}:{FEED_POS[3]}[feed]")

    next_input = 1
    screen_path = run_dir / "screen.mkv"
    geo_path = run_dir / "window_geometry.json"
    have_screen = screen_path.exists() and geo_path.exists()
    if have_screen:
        geo = json.loads(geo_path.read_text())
        screen_off = (recorders.get("screen_start_wall", t0_wall) + FFMPEG_STARTUP_S - t0_wall) if t0_wall else 0.0
        inputs += ["-i", str(screen_path)]
        filters.append(
            f"[{next_input}:v]setpts=PTS-STARTPTS+{screen_off:.3f}/TB,fps=30,"
            f"crop={geo['w']}:{geo['h']}:{geo['x']}:{geo['y']},"
            f"scale={APP_POS[2]}:{APP_POS[3]}:force_original_aspect_ratio=decrease,"
            f"pad={APP_POS[2]}:{APP_POS[3]}:(ow-iw)/2:(oh-ih)/2:color=0x101418[app]"
        )
        next_input += 1
    filters.append("[base][feed]overlay=x={}:y={}:eof_action=repeat[v1]".format(*FEED_POS[:2]))
    if have_screen:
        filters.append("[v1][app]overlay=x={}:y={}:eof_action=repeat[v2]".format(*APP_POS[:2]))
        cur = "v2"
    else:
        cur = "v1"

    # --- text overlays
    def add_text(src: str, dst: str, textfile: Path, *, x: str, y: int, size: int, color: str,
                 start: float, end: float, bold: bool = False, center: bool = False) -> str:
        font = FONT_BOLD if bold else FONT
        xexpr = f"(w-text_w)/2" if center else x
        return (
            f"[{src}]drawtext=fontfile={font}:textfile={textfile}:x={xexpr}:y={y}:fontsize={size}:"
            f"fontcolor={color}:enable='between(t,{start:.2f},{end:.2f})'[{dst}]"
        )

    captions = [e for e in timeline if e.get("kind") == "caption"]
    spoken = _spoken_entries(timeline)
    verdicts = [e for e in timeline if e.get("kind") == "verdict"]
    actions = [e for e in timeline if e.get("kind") == "action"]

    step = 0
    for i, entry in enumerate(captions):
        end = captions[i + 1]["t"] if i + 1 < len(captions) else duration
        tf = _write_text(work, run, step := step + 1, entry["text"])
        filters.append(add_text(cur, f"v{step}", tf, x="40", y=42, size=34, color="white",
                                start=entry["t"], end=end, bold=True))
        cur = f"v{step}"
    # panel labels (always on)
    tf = _write_text(work, run, step := step + 1, "simulated camera feed")
    filters.append(add_text(cur, f"v{step}", tf, x=str(FEED_POS[0]), y=FEED_POS[1] - 32, size=22,
                            color="0x9aa4b0", start=0, end=duration))
    cur = f"v{step}"
    tf = _write_text(work, run, step := step + 1, "the app, unmodified" if have_screen else "app window not recorded")
    filters.append(add_text(cur, f"v{step}", tf, x=str(APP_POS[0]), y=APP_POS[1] - 32, size=22,
                            color="0x9aa4b0", start=0, end=duration))
    cur = f"v{step}"
    # subtitles: each spoken line holds until the next one
    for i, entry in enumerate(spoken):
        end = spoken[i + 1]["t"] if i + 1 < len(spoken) else entry["t"] + 6.0
        tf = _write_text(work, run, step := step + 1, entry["text"])
        filters.append(add_text(cur, f"v{step}", tf, x="", y=985, size=30, color="0xffe08a",
                                start=entry["t"], end=end, center=True))
        cur = f"v{step}"
    # verdicts flash for 5 s
    for entry in verdicts:
        ok = entry["text"].startswith("PASS")
        tf = _write_text(work, run, step := step + 1, entry["text"])
        filters.append(add_text(cur, f"v{step}", tf, x="", y=915, size=26,
                                color="0x4ade80" if ok else "0xf87171",
                                start=entry["t"], end=entry["t"] + 5.0, center=True, bold=True))
        cur = f"v{step}"
    # human actions, bottom-left, 4 s
    for entry in actions:
        tf = _write_text(work, run, step := step + 1, f"hand: {entry['text']}")
        filters.append(add_text(cur, f"v{step}", tf, x="40", y=1040, size=22, color="0x9aa4b0",
                                start=entry["t"], end=entry["t"] + 4.0))
        cur = f"v{step}"

    filters.append(f"[{cur}]format=yuv420p[vout]")

    # --- audio
    audio_path = run_dir / "audio.wav"
    audio_off = 0.0
    have_audio = audio_path.exists() and _mean_volume_db(audio_path) > SILENT_DB
    if have_audio:
        inputs += ["-i", str(audio_path)]
        audio_off = (recorders.get("audio_start_wall", t0_wall) + 0.2 - t0_wall) if t0_wall else 0.0
    else:
        synth = _synth_audio(spoken, work, run)
        if synth is not None:
            inputs += ["-i", str(synth)]
            audio_off = 0.0
        else:
            inputs += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            audio_off = 0.0
    if audio_off < 0:
        filters.append(f"[{next_input}:a]aresample=48000,atrim=start={-audio_off:.3f},apad=whole_dur={duration + 1:.2f}[aout]")
    else:
        filters.append(
            f"[{next_input}:a]aresample=48000,apad=whole_dur={duration + 1 + audio_off:.2f},"
            f"adelay={int(audio_off * 1000)}|{int(audio_off * 1000)}[aout]"
        )

    script = work / f"{run}_filter.txt"
    script.write_text(";\n".join(filters))
    segment = work / f"{run}_segment.mp4"
    _run([
        "ffmpeg", "-y", *inputs, "-filter_complex_script", str(script),
        "-map", "[vout]", "-map", "[aout]", "-t", f"{duration:.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", "30",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", str(segment),
    ])
    return segment, duration


def build_card(work: Path, name: str, title: str, subtitle: str, duration: float = 3.0) -> Path:
    tf_t = _write_text(work, name, 1, title)
    tf_s = _write_text(work, name, 2, subtitle)
    out = work / f"{name}_card.mp4"
    _run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x101418:s={W}x{H}:r=30:d={duration}",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-vf", (
            f"drawtext=fontfile={FONT_BOLD}:textfile={tf_t}:x=(w-text_w)/2:y=(h-text_h)/2-40:fontsize=58:fontcolor=white,"
            f"drawtext=fontfile={FONT}:textfile={tf_s}:x=(w-text_w)/2:y=(h-text_h)/2+50:fontsize=28:fontcolor=0x9aa4b0,"
            "format=yuv420p"
        ),
        "-t", f"{duration}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", "30",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest", str(out),
    ])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", default=["happy_path", "recovery", "escalation_live", "agent_first"])
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    pieces: list[Path] = []
    pieces.append(build_card(
        WORK, "000_opening",
        "Inkwatch — every specified scenario, on video",
        "a simulated human (real page, real pencil) plays against the unmodified app, fed through its documented camera URL",
        4.0,
    ))
    totals = {"pass": 0, "fail": 0}
    for i, run in enumerate(args.runs):
        run_dir = SIM_ROOT / run
        if not (run_dir / "timeline.json").exists():
            print(f"skipping {run}: no timeline.json (run tools/sim_scenarios.py {run} first)")
            continue
        title, subtitle = RUN_TITLES.get(run, (run, ""))
        pieces.append(build_card(WORK, f"{i + 1:03d}_{run}", title, subtitle))
        segment, _dur = build_segment(run, run_dir, WORK)
        pieces.append(segment)
        verdicts_path = run_dir / "verdicts.json"
        if verdicts_path.exists():
            for v in json.loads(verdicts_path.read_text()):
                totals["pass" if v["ok"] else "fail"] += 1

    verdict_line = f"scenario checks: {totals['pass']} PASS, {totals['fail']} FAIL"
    pieces.append(build_card(WORK, "999_closing", "Every check is the app's real behavior", verdict_line, 3.5))

    concat = WORK / "concat.txt"
    concat.write_text("".join(f"file '{p.resolve()}'\n" for p in pieces))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(out)])
    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-loglevel", "error", "-show_entries", "format=duration,size",
         "-of", "json", str(out)],
        capture_output=True, text=True,
    )
    print(f"wrote {out}: {probe.stdout.strip()}")
    print(verdict_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
