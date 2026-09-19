"""Scenario simulator: a virtual human playing against the real, unmodified
Inkwatch app.

The app is launched as a subprocess with `--camera http://127.0.0.1:<port>/stream`
— the MJPEG-over-HTTP input README documents for phone cameras ("IP camera /
phone stream"), so no app code changes and no root/virtual-camera kernel module
are needed. This script serves it frames rendered by `tools/simrender.py`
(real paper, real pencil, wobbly graphite strokes), paces them at 30 fps
(server-paced; measured end-to-end lag ~40-80 ms), and plays the human's side
by LISTENING to what the app says — every spoken line is also printed to
stdout (O4), which is the exact information a human player gets.

What it verifies (per scenario, as PASS/FAIL verdicts in timeline.json):
the app reacts to each §9 edge case with the specified message and recovery,
and state only ever changes through the page. Anything unexpected is a bug to
fix — the point of the exercise, not a test failure to route around.

Recording for the final video (`recordings/sim/<run>/`):
  feed.mp4 + feed_ts.jsonl   every frame served to the app, with wall times
  screen.mkv                 ffmpeg x11grab of the desktop (the app's window)
  audio.wav                  pw-record of the speaker monitor (TTS), if found
  app.log                    the app's stdout+stderr
  timeline.json              captions, app messages, event-log lines, verdicts

Usage: python tools/simulate.py [run_name ...]   (default: all runs)
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import queue
import re
import socketserver
# Only ever invoked below with fixed argv lists (fixed local binaries or
# sys.executable), never a shell string, never untrusted input.
import subprocess  # nosec B404
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simrender import FRAME_H, FRAME_W, RecordingBackground, Scene  # noqa: E402

from inkwatch.output import CELL_NAMES  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FPS = 30
CELL_INDEX = {name: i for i, name in enumerate(CELL_NAMES)}


# ------------------------------------------------------------------ MJPEG cam


class _StreamHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        server: "CameraServer" = self.server  # type: ignore[assignment]
        if server.dropping.is_set():
            return  # connection refused-ish: camera stays "unplugged"
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        frame_interval = 1.0 / FPS
        cid = server.claim_clock()
        try:
            while not server.dropping.is_set():
                t0 = time.monotonic()
                jpg = server.next_jpeg() if server.is_clock_owner(cid) else server.current_jpeg()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                dt = time.monotonic() - t0
                if dt < frame_interval:
                    time.sleep(frame_interval - dt)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args) -> None:
        pass


class CameraServer(socketserver.ThreadingTCPServer):
    """Serves scene.tick() frames as an MJPEG stream. Only the first
    connected client drives the scene clock (reconnects after a camera
    drop keep rendering the same scene). `dropping` simulates an unplug:
    the open connection is closed and new ones refused."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port: int, scene: Scene, feed_writer: "FeedRecorder") -> None:
        self.scene = scene
        self.feed = feed_writer
        self.dropping = threading.Event()
        self._tick_lock = threading.Lock()
        self._last_jpeg: bytes | None = None
        self._clock_owner: int = 0  # first client id drives the scene clock
        self._next_client_id = 0
        super().__init__(("127.0.0.1", port), _StreamHandler)

    def claim_clock(self) -> int:
        """Connections are numbered; the LATEST one owns the scene clock.
        Ownership must migrate on reconnect: the restarted app (or a
        re-plugged camera) is a new connection — if the first client kept
        ownership forever, every later client would get a frozen scene
        (found when B2's restarted app calibrated on a static frame and
        then nothing ever moved again)."""
        cid = self._next_client_id
        self._next_client_id += 1
        self._clock_owner = cid
        return cid

    def is_clock_owner(self, cid: int) -> bool:
        return cid == self._clock_owner

    def next_jpeg(self) -> bytes:
        with self._tick_lock:
            frame = self.scene.tick(1.0 / FPS)
        self.feed.write(frame)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        self._last_jpeg = jpg.tobytes()
        return self._last_jpeg

    def current_jpeg(self) -> bytes:
        """Latest frame WITHOUT advancing the scene — for any extra
        connection that isn't the clock owner (cv2 sometimes probes with
        a second connection; two ticking clients would double scene speed)."""
        with self._tick_lock:
            if self._last_jpeg is None:
                ok, jpg = cv2.imencode(".jpg", self.scene.tick(1.0 / FPS), [cv2.IMWRITE_JPEG_QUALITY, 88])
                self._last_jpeg = jpg.tobytes()
            return self._last_jpeg

    def scene_call(self, fn, *args, **kwargs):
        with self._tick_lock:
            return fn(*args, **kwargs)


class FeedRecorder:
    """Every frame served to the app -> feed.mp4 (+ wall time per frame)."""

    def __init__(self, out_dir: Path) -> None:
        self._writer = cv2.VideoWriter(
            str(out_dir / "feed.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (FRAME_W, FRAME_H)
        )
        self._ts = (out_dir / "feed_ts.jsonl").open("w")
        if not self._writer.isOpened():
            raise RuntimeError("cv2.VideoWriter mp4v unavailable")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
        self._ts.write(json.dumps({"t": time.time()}) + "\n")

    def close(self) -> None:
        self._writer.release()
        self._ts.close()


# -------------------------------------------------------------------- the app


class AppProcess:
    """The real `python -m inkwatch` as a subprocess. stdout+stderr lines
    are streamed to listeners; the session dir is parsed from its own
    'Logging this game to ...' line."""

    def __init__(self, port: int, extra_args: list[str], log_path: Path, voice: bool = True) -> None:
        cmd = [sys.executable, "-u", "-m", "inkwatch", "--camera", f"http://127.0.0.1:{port}/stream"]
        if not voice:
            cmd.append("--no-voice")
        cmd += extra_args
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(  # nosec B603
            cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        self.lines: queue.Queue[tuple[float, str]] = queue.Queue()
        self.session_dirs: list[Path] = []
        self._log = log_path.open("a")
        self._dead = threading.Event()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            self._log.write(line + "\n")
            self._log.flush()
            self.lines.put((time.time(), line))
            m = re.search(r"Logging this game to (sessions/\S+)", line)
            if m:
                self.session_dirs.append(REPO / m.group(1))
        self._dead.set()

    def poll_dead(self) -> bool:
        return self._dead.is_set() or self.proc.poll() is not None

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._log.close()

    @property
    def session_dir(self) -> Path | None:
        return self.session_dirs[-1] if self.session_dirs else None


# ------------------------------------------------------------- scenario engine


@dataclass
class Verdict:
    scenario: str
    check: str
    ok: bool
    detail: str = ""
    t: float = 0.0


@dataclass
class Timeline:
    out_dir: Path
    t0: float = field(default_factory=time.time)
    entries: list[dict] = field(default_factory=list)

    def add(self, kind: str, text: str, **extra) -> None:
        entry = {"t": time.time() - self.t0, "kind": kind, "text": text}
        entry.update(extra)
        self.entries.append(entry)

    def flush(self) -> None:
        (self.out_dir / "timeline.json").write_text(json.dumps(self.entries, indent=1))


class Director:
    """Runs one scenario run: scene + camera server + app + assertions."""

    def __init__(self, name: str, out_root: Path, app_args: list[str], seed: int = 7, voice: bool = True,
                 record_screen: bool = True) -> None:
        self.name = name
        self.out_dir = out_root / name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.app_args = app_args
        self.voice = voice
        self.record_screen = record_screen
        self.timeline = Timeline(self.out_dir)
        self.verdicts: list[Verdict] = []
        self.scenario = ""
        self._msg_offset = 0  # stdout lines consumed so far
        self._msgs: list[tuple[float, str]] = []
        self._event_offset = 0

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Director":
        self.scene = Scene(RecordingBackground(), seed=7)
        self.feed = FeedRecorder(self.out_dir)
        self.server = CameraServer(0, self.scene, self.feed)
        self.port = self.server.server_address[1]
        self._server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._server_thread.start()
        self.screen = None
        self.audio = None
        self.recorder_info: dict = {}
        if self.record_screen:
            self.audio_target = _find_monitor_source()
            if self.audio_target is not None:
                self.audio = subprocess.Popen(  # nosec B603 B607
                    ["pw-record", "--target", self.audio_target, str(self.out_dir / "audio.wav")],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self.recorder_info["audio_start_wall"] = time.time()
            self.win_capture = WindowCapture(self.out_dir)
            self.win_capture.start()
            time.sleep(1.0)  # let recorders settle before t0
        self.timeline.t0 = time.time()
        self.recorder_info["t0_wall"] = self.timeline.t0
        self.app = AppProcess(self.port, self.app_args, self.out_dir / "app.log", voice=self.voice)
        return self

    def __exit__(self, *exc) -> None:
        self.timeline.flush()
        (self.out_dir / "recorders.json").write_text(json.dumps(self.recorder_info))
        self.app.stop()
        self.server.shutdown()
        self.feed.close()
        if self.record_screen:
            self.win_capture.stop()
        if self.audio is not None:
            self.audio.terminate()
            try:
                self.audio.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.audio.kill()
        summary = [vars(v) for v in self.verdicts]
        (self.out_dir / "verdicts.json").write_text(json.dumps(summary, indent=1))
        fails = [v for v in self.verdicts if not v.ok]
        print(f"\n=== run {self.name}: {len(self.verdicts) - len(fails)} PASS, {len(fails)} FAIL")
        for v in fails:
            print(f"  FAIL [{v.scenario}] {v.check}: {v.detail}")

    # -- observation helpers -------------------------------------------------

    def _pump(self, seconds: float) -> None:
        """Let time pass (the server keeps rendering); collect app lines."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            self._drain()
            if self.app.poll_dead():
                raise RuntimeError("app died mid-run — see app.log")
            time.sleep(0.02)

    def _drain(self) -> None:
        while True:
            try:
                t, line = self.app.lines.get_nowait()
            except queue.Empty:
                return
            self._msgs.append((t, line))
            self.timeline.add("app_line", line)

    def wait_msg(self, patterns: str | list[str], timeout: float = 25.0) -> tuple[str, str] | None:
        """Wait for an app stdout line containing any pattern (spoken
        lines are printed, O4). Returns (pattern, line) or None on timeout.
        Lines are consumed in order: everything scanned before the match
        is past history for later waits."""
        if isinstance(patterns, str):
            patterns = [patterns]
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._drain()
            for i in range(self._msg_offset, len(self._msgs)):
                line = self._msgs[i][1]
                for pat in patterns:
                    if pat in line:
                        self._msg_offset = i + 1
                        return pat, line
            self._msg_offset = len(self._msgs)
            if self.app.poll_dead():
                raise RuntimeError(f"app died waiting for {patterns} — see app.log")
            time.sleep(0.02)
        return None

    def expect_silent(self, patterns: str | list[str], seconds: float) -> bool:
        """Assert NO app line matches any pattern for `seconds` (the
        debounce checks: nothing may commit while the pen is mid-mark)."""
        if isinstance(patterns, str):
            patterns = [patterns]
        deadline = time.time() + seconds
        while time.time() < deadline:
            self._drain()
            for i in range(self._msg_offset, len(self._msgs)):
                line = self._msgs[i][1]
                if any(pat in line for pat in patterns):
                    return False
            self._msg_offset = len(self._msgs)
            time.sleep(0.02)
        return True

    def events(self) -> list[dict]:
        """New lines of the app's own events.jsonl since the last call."""
        session_dir = self.app.session_dir
        if session_dir is None:
            return []
        path = session_dir / "events.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        new = lines[self._event_offset :]
        self._event_offset = len(lines)
        out = []
        for line in new:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out

    def wait_event(self, event_type: str, timeout: float = 25.0, **field_contains) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            for event in self.events():
                if event.get("type") != event_type:
                    continue
                if all(str(event.get(k, "")).find(v) >= 0 for k, v in field_contains.items()):
                    return event
            if self.app.poll_dead():
                raise RuntimeError(f"app died waiting for event {event_type} — see app.log")
            time.sleep(0.05)
        return None

    # -- scenario steps ------------------------------------------------------

    def caption(self, text: str) -> None:
        print(f"\n--- {text}")
        self.timeline.add("caption", text)

    def say_seen(self, line: str) -> None:
        print(f"  app: {line}")

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        verdict = Verdict(self.scenario, name, bool(ok), detail, time.time() - self.timeline.t0)
        self.verdicts.append(verdict)
        self.timeline.add("verdict", f"{'PASS' if ok else 'FAIL'}: {name}", detail=detail)
        print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail and not ok else ""))
        return ok

    def set_scenario(self, name: str, caption: str) -> None:
        self.scenario = name
        self.caption(caption)

    # scene wrappers (thread-safe against the streaming loop)

    def draw(self, cell: int, symbol: str) -> None:
        self.server.scene_call(self.scene.draw, cell, symbol)
        self._wait_idle()
        self.timeline.add("action", f"draw {symbol} at {CELL_NAMES[cell]}")

    def instant_mark(self, cell: int, symbol: str) -> None:
        def _stamp():
            self.scene.marks[cell] = self.scene._place_mark(cell, symbol)
        self.server.scene_call(_stamp)
        self.timeline.add("action", f"sneaky mark {symbol} at {CELL_NAMES[cell]} (unseen)")

    def half_draw(self, cell: int, symbol: str, pause: float = 2.2) -> None:
        self.server.scene_call(self.scene.half_draw_then_finish, cell, symbol, pause)
        self._wait_idle()

    def linger(self, seconds: float) -> None:
        self.server.scene_call(self.scene.linger, seconds)
        self._wait_idle()

    def bump(self, dx: float = 38, dy: float = -22, deg: float = 6) -> None:
        self.server.scene_call(self.scene.bump, dx, dy, deg)
        self._wait_idle()
        self.timeline.add("action", "page bumped")

    def erase(self, cell: int) -> None:
        self.server.scene_call(self.scene.erase, cell)
        self._wait_idle()
        self.timeline.add("action", f"erase {CELL_NAMES[cell]}")

    def shadow(self, cell: int, strength: float = 0.12) -> None:
        self.server.scene_call(self.scene.shadow, cell, "ambiguous", None, strength)
        self.timeline.add("action", f"shadow over {CELL_NAMES[cell]}")

    def set_shadow_strength(self, strength: float) -> None:
        self.server.scene_call(self.scene.set_shadow_strength, strength)
        self.timeline.add("action", f"the shadow deepens ({strength})")

    def shadow_ramp(self, cell: int, s1: float = 0.18, duration: float = 20.0) -> None:
        """Deepen a calibrated shadow continuously (a cloud moving in)."""
        self.server.scene_call(self.scene.shadow_ramp, cell, 0.12, s1, duration)
        self.timeline.add("action", f"shadow over {CELL_NAMES[cell]}, deepening")

    def stop_ramp(self) -> None:
        self.server.scene_call(self.scene.stop_ramp)

    def clear_shadow(self) -> None:
        self.server.scene_call(self.scene.clear_shadow)
        self._pump(0.5)

    def new_page(self) -> None:
        def _reset():
            self.scene.marks.clear()
            self.scene.erased.clear()
            self.scene.pose.update(dx=0.0, dy=0.0, deg=0.0)
        self.server.scene_call(_reset)
        self.timeline.add("action", "fresh blank page")

    def drop_camera(self, seconds: float) -> None:
        self.server.dropping.set()
        time.sleep(0.3)  # let the open connection notice
        self.server.shutdown()  # breaks the open handler loop
        self.server.server_close()  # free the port for the "plugged back in" server
        time.sleep(seconds)
        # a fresh server on the same port = the camera "plugged back in"
        self.server = CameraServer(self.port, self.scene, self.feed)
        self._server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._server_thread.start()
        self.timeline.add("action", f"camera unplugged {seconds}s then restored")

    def _wait_idle(self) -> None:
        deadline = time.time() + 45.0
        while True:
            busy = self.server.scene_call(lambda: self.scene.busy)
            if not busy:
                return
            if time.time() > deadline:
                raise RuntimeError("scene action never finished (stream stalled?)")
            self._pump(0.05)

    # -- compound helpers ------------------------------------------------------

    def calibrate(self, timeout: float = 30.0) -> bool:
        """Wait for the app's opening line after a fresh page is shown."""
        got = self.wait_msg("I can see the board", timeout=timeout)
        if got:
            self.say_seen(got[1])
        return self.check("calibration greets the player", got is not None,
                          f"no intro within {timeout}s" if not got else "")

    def await_agent_target(self, line: str) -> int | None:
        """'... I'll take top left. Please draw an O there.' -> cell index."""
        m = re.search(r"I'll take ([a-z ]+?)\.", line)
        if not m:
            return None
        return CELL_INDEX.get(m.group(1).strip())

    def restart_app(self, timeout: float = 40.0) -> bool:
        """Fresh app process + fresh page between scenarios that don't end
        in GAME_OVER (the blank-page auto-restart only fires at GAME_OVER;
        mid-game scenarios get a clean process instead). The stream server
        stays up: the new process reconnects like any re-opened camera."""
        self.app.stop()
        self.new_page()
        self._event_offset = 0
        self.app = AppProcess(self.port, self.app_args, self.out_dir / "app.log", voice=self.voice)
        if self.record_screen:
            self.win_capture.refresh()
        return self.calibrate(timeout=timeout)

    def restart_game(self, timeout: float = 40.0) -> bool:
        """Swap in a blank page and wait for the auto-restart + greeting."""
        self.new_page()
        got_restart = self.wait_msg("New page detected", timeout=timeout)
        if got_restart:
            self.say_seen(got_restart[1])
        ok1 = self.check("blank page restarts the game", got_restart is not None,
                         "no auto-restart after page swap" if not got_restart else "")
        got_hello = self.wait_msg("I can see the board", timeout=timeout) if got_restart else None
        if got_hello:
            self.say_seen(got_hello[1])
        ok2 = self.check("restarted game calibrates", got_hello is not None,
                         "no intro after restart" if not got_hello else "")
        return ok1 and ok2


def _find_monitor_source() -> str | None:
    """The default output sink's id — `pw-record --target <sink>` captures
    exactly what plays through the speakers (verified: the app's TTS at
    -22 dB mean). There is no separate 'monitor' node on this PipeWire
    setup; targeting the sink itself is the way."""
    try:
        out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5).stdout  # nosec B603 B607
        data = json.loads(out)
        for obj in data:
            props = obj.get("info", {}).get("props", {})
            if props.get("media.class") == "Audio/Sink":
                return str(obj["id"])
    except Exception:
        return None
    return None


# ------------------------------------------------------------- window capture


def _read_xwd(path: Path) -> np.ndarray | None:
    """Minimal XWD (X11 window dump) reader: 25 big-endian uint32 header
    fields, the window-name string (inside header_size), the colormap
    (ncolors × 12 bytes, present even for TrueColor dumps — skipping it
    is the difference between a board and rainbow noise), then ZPixmap
    pixels. Returns BGR, or None on anything unusual."""
    data = path.read_bytes()
    if len(data) < 100:
        return None
    header = np.frombuffer(data[:100], dtype=">u4")
    header_size, version, fmt, depth, w, h = header[0], header[1], header[2], header[3], header[4], header[5]
    byte_order, bpp, bytes_per_line, ncolors = header[7], header[11], header[12], header[19]
    if version != 7 or fmt != 2 or w == 0 or h == 0 or bpp not in (24, 32):
        return None
    offset = int(header_size) + int(ncolors) * 12  # colormap follows header+name
    pixels = np.frombuffer(data, dtype=np.uint8, offset=offset)
    need = int(bytes_per_line) * int(h)
    if len(pixels) < need:
        return None
    rows = pixels[:need].reshape(int(h), int(bytes_per_line))
    img = rows[:, : int(w) * (bpp // 8)].reshape(int(h), int(w), bpp // 8)
    if byte_order == 0:  # LSBFirst: B,G,R,(X) for the usual ff0000/ff00/ff masks
        bgr = img[..., :3]
    else:
        bgr = img[..., -3:][..., ::-1] if bpp == 32 else img[..., ::-1]
    return np.ascontiguousarray(bgr)


class WindowCapture:
    """The app's own cv2 window, captured frame by frame with xwd.

    Why not x11grab of the desktop: on this Wayland session the app's
    window lives in XWayland, which is rootless — the X root window is
    composited by Wayland, so a full-screen x11grab records pure black
    (verified live). Per-window `xwd -id` reads the window's own backing
    store and returns the real pixels (~60 dumps/s measured, so 30 fps
    capture has headroom). Frames are letterboxed onto a fixed canvas
    (the window resizes itself between the 600x600 overlay and the
    smaller banner views) and written to appwin.mp4 with per-frame wall
    times for exact sync with the rest of the footage."""

    CANVAS = (660, 660)
    # Window chrome above the board: WM titlebar (y 12-47), Qt's own
    # caption (52-63), Qt toolbar (64-99), white edge (100-104) — measured
    # row by row on a live dump; board content starts at y 105.
    CROP = (16, 105, 16, 16)  # left, top, right, bottom px

    def __init__(self, out_dir: Path, title: str = "inkwatch") -> None:
        self.out_dir = out_dir
        self.title = title
        self.window_id: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: cv2.VideoWriter | None = None
        self._ts = None
        self._xwd_tmp = out_dir / "_win.xwd"

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def refresh(self) -> None:
        """Drop the current window id (the app was restarted; its window
        is a new X window). The capture thread re-resolves on its own."""
        self.window_id = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._writer is not None:
            self._writer.release()
        if self._ts is not None:
            self._ts.close()

    def _resolve(self) -> str | None:
        try:
            out = subprocess.run(["xwininfo", "-name", self.title], capture_output=True, text=True, timeout=3).stdout  # nosec B603 B607
        except Exception:
            return None
        m = re.search(r"Window id: (0x[0-9a-fA-F]+)", out)
        return m.group(1) if m else None

    def _run(self) -> None:
        self._writer = cv2.VideoWriter(
            str(self.out_dir / "appwin.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, self.CANVAS
        )
        self._ts = (self.out_dir / "appwin_ts.jsonl").open("w")
        while not self._stop.is_set():
            t0 = time.monotonic()
            if self.window_id is None:
                self.window_id = self._resolve()
                if self.window_id is None:
                    time.sleep(0.3)
                    continue
            frame = self._grab()
            if frame is None:
                time.sleep(0.05)
                continue
            frame = self._letterbox(frame)
            self._writer.write(frame)
            self._ts.write(json.dumps({"t": time.time()}) + "\n")
            self._ts.flush()
            dt = time.monotonic() - t0
            if dt < 1.0 / FPS:
                time.sleep(1.0 / FPS - dt)

    def _grab(self) -> np.ndarray | None:
        assert self.window_id is not None
        try:
            rc = subprocess.run(  # nosec B603 B607
                ["xwd", "-silent", "-id", self.window_id, "-out", str(self._xwd_tmp)],
                capture_output=True, timeout=3,
            ).returncode
        except Exception:
            return None
        if rc != 0 or not self._xwd_tmp.exists():
            return None
        try:
            return _read_xwd(self._xwd_tmp)
        except Exception:
            return None

    def _letterbox(self, frame: np.ndarray) -> np.ndarray:
        l, t, r, b = self.CROP
        h, w = frame.shape[:2]
        if h > t + b + 50 and w > l + r + 50:
            frame = frame[t : h - b, l : w - r]
        h, w = frame.shape[:2]
        scale = min(self.CANVAS[0] / w, self.CANVAS[1] / h)
        if abs(scale - 1.0) > 0.02:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        canvas = np.full((self.CANVAS[1], self.CANVAS[0], 3), 16, np.uint8)
        y, x = (self.CANVAS[1] - h) // 2, (self.CANVAS[0] - w) // 2
        canvas[y : y + h, x : x + w] = frame
        return canvas
