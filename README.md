# Inkwatch

Play tic-tac-toe on paper against an AI agent that watches the page through a camera.

You draw every mark, yours and the agent's. The agent notices your move on its own, keeps the game state, picks its reply, and tells you where to draw it. Ink on the page is the move: the agent never assumes a move it didn't see drawn, and when it isn't sure, it asks instead of guessing.

**Demo video:** TODO link (unlisted YouTube / Drive)

| Doc | What's in it |
|---|---|
| [`PRODUCT.md`](PRODUCT.md) | What it does, requirements, edge cases, metrics |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | One-page design for a game-agnostic version |
| [`NOTES.md`](NOTES.md) | Decisions and why, known limits, how AI tools were used |
| [`AGENTS.md`](AGENTS.md) | Branch, PR, test, and review workflow |
| [`CODE_REVIEW.md`](CODE_REVIEW.md) | Review checklist: spec, hard rules, security, tests |
| [`COMPLIANCE.md`](COMPLIANCE.md) | SOC 2 / ISO 27001 gap analysis and readiness notes |

---

## Quick start (about 10 minutes)

### 1. What you need
- Python 3.11 or newer
- A webcam, or a phone used as a webcam (see [Camera setup](#camera-setup))
- A printer, or a tablet you can lay flat
- A dark pen or marker
- Speakers or headphones

### 2. Install

```bash
git clone <repo-url> inkwatch
cd inkwatch
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

**Linux only:** speech needs espeak: `sudo apt install espeak-ng`. macOS and Windows use the built-in voice.

### 3. Print the board

```bash
python tools/make_sheet.py
```

This writes `assets/board.png`: a US Letter page with a 3×3 grid and a black ArUco marker square in each corner. The markers are how the agent finds the board, so keep all four visible and uncovered. Print it at 100% scale (not "fit to page") and check the 1-inch mark on the sheet against a real ruler — the sheet is a PNG rather than a PDF (see `NOTES.md`), so your print dialog's scale isn't guaranteed automatically.

No printer? Open the PNG full-screen on a tablet laid flat and play with a stylus, or draw marks on a sheet of tracing paper over it.

### 4. Position the camera
Put the camera above the page, looking down, ideally no more than about 35° off vertical. The whole sheet should fill most of the frame with some margin. Avoid a lamp directly behind you that casts your hand's shadow onto the page.

Check it:

```bash
python tools/calibrate.py
```

You should see the straightened top-down board with a small ink number and a none/ambiguous/marked label in each cell. With a blank sheet in view, press `b` to snapshot the baseline, then draw a test mark in one cell: its label should flip to `marked` while the others stay `none`. Press `q` to close. Use a fresh sheet for the game.

### 5. (Optional) Enable the vision fallback
The game plays fully offline. A vision model is only called when the camera view is ambiguous. To enable it:

```bash
cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY=...
```

Without a key, the agent asks you directly whenever it's unsure.

### 6. Play

```bash
python -m inkwatch
```

The agent says *"I can see the board. You're X, you go first."* Then:

1. Draw an **X** in any cell and move your hand away.
2. The agent says which cell you played and where it will play, e.g. *"You played center. I'll take top left. Please draw an O there."*
3. Draw the **O** where it asked. It confirms and it's your turn again.
4. At the end it announces the result and checks the page matches.

Cells are named by row (**top / middle / bottom**) and column (**left / center / right**). The middle cell is just **center**. The target cell is also highlighted on screen.

---

## Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `n` | New game (use a fresh sheet) |
| `r` | Force a full re-read of the board |
| `d` | Toggle debug overlay (ink ratios, confidence) |
| `y` / `n` | Answer the agent's question, if it asks one you can't resolve on the page |

## Options

```bash
python -m inkwatch --agent-first        # agent plays X and opens
python -m inkwatch --camera 1           # pick another camera index
python -m inkwatch --camera <url>       # IP camera / phone stream
python -m inkwatch --no-voice           # display only
python -m inkwatch --no-escalation      # never call the vision model
python -m inkwatch --record             # save the raw stream for replay
```

All defaults live in [`config.yaml`](config.yaml).

## Camera setup

| Camera | How |
|---|---|
| Laptop webcam | Works if you can angle the screen down over the page; a small stand or stack of books helps. |
| iPhone + Mac | Continuity Camera shows up as a normal camera. Try `--camera 1` if index 0 is the built-in one. |
| Any phone | A webcam app (e.g. Camo, DroidCam, Iriun) exposes it as a camera or a stream URL for `--camera`. |

A phone on a gooseneck or propped on a glass above the desk gives the best angle.

---

## When things go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| *"I can't see the whole page"* | A corner marker is out of frame, covered, or blurred | Move the page or camera; keep fingers off the corners. |
| Agent never notices your move | Hand still in view, or pen too faint | Move your hand fully away; use a darker pen. Check numbers in `tools/calibrate.py`. |
| Agent reacts before you finish drawing | Very fast pen lift | It waits for two consistent reads; if it still happens, raise `stability_frames` in `config.yaml`. |
| Shadows counted as marks | Light source behind or beside you | Move the lamp in front of you or overhead. |
| No sound | TTS engine missing | Linux: install `espeak-ng`. Or run with `--no-voice`. |
| Camera index wrong or black window | Different device order | Try `--camera 1`, `--camera 2`. |

---

## Tests and replay

```bash
pytest
```

Covers rules, decision (the agent never loses), session state transitions, and ink detection on saved cell images. One short recorded game runs through the full pipeline as a replay test.

Replay any recorded session without a camera:

```bash
python -m inkwatch --record                      # play and record
python -m inkwatch.replay sessions/<session-id>  # re-run perception + session offline
```

Replay writes a fresh event log you can diff against the original. This is how threshold changes are checked against the same frames instead of new, uncontrolled games. See [`eval/README.md`](eval/README.md).

## Measured results

TODO: fill from recorded eval games. Targets are in [`PRODUCT.md` §13](PRODUCT.md).

| Metric | Target | Measured |
|---|---|---|
| Move detection accuracy | ≥ 98% | |
| False triggers per 10 games | ≤ 1 | |
| Escalation rate | < 5% of turns | |
| Time to detect (p50) | ≤ 1 s | |
| End-of-game desync | 0 | |
| Cost per game | ≈ $0 | |

## Project layout

```
inkwatch/            the application: one file per architecture component
  perception.py      board markers, rectification, stability, per-cell ink
  escalation.py      vision-model read on low confidence (timeout, budget)
  session.py         state machine; the only owner of game state
  rules.py           pure tic-tac-toe rules
  decision.py        minimax
  output.py          speech + overlay
  events.py          observation types and event log
  replay.py          offline pipeline over recordings
tools/               printable sheet generator, calibration view
tests/               unit and replay tests
eval/                ground-truth labels and metrics instructions
sessions/            runtime logs and frames (not committed)
```

## Known limits

The short version: needs the printed marker sheet, normal indoor light, and a dark pen; one game and one camera at a time. Full list with reasons in [`NOTES.md`](NOTES.md#known-limits).
