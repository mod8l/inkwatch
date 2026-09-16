# Notes

Working notes for the Inkwatch take-home: what I decided and why, what I dropped, what doesn't work, and how I used AI tools. Written during the work, not reconstructed afterwards.

> **TODO before submitting:** every `TODO` below is filled from what actually happened in the build. Nothing here should claim a result I didn't measure.

---

## 1. How I read the brief

- Tic-tac-toe is the vehicle. The hard part is turning a noisy physical scene into trustworthy events and keeping a shared state with a human without drifting. Game strength is solved.
- "A finished loop with honest notes beats an ambitious system" sets the priority order: working loop → recovery → evidence → extras.
- "Ink on the page is the move" means the agent must never infer a move it didn't see drawn, including its own.
- The ambiguity is intentional, so the assumptions are written down in [`PRODUCT.md` §5](PRODUCT.md) before building, not after.

**Time spent:** TODO (design / perception / loop / recovery / escalation / video / docs)

---

## 2. Decisions

Each entry: what I chose, what I considered, why, and what would make me change it.

### D1. Printed sheet with ArUco corner markers
- **Considered:** freehand grid found by contours or Hough lines; thick drawn border.
- **Why:** rectification is not the interesting part of the problem, and a marker sheet makes it deterministic under perspective, page movement and partial occlusion. It frees the time-box for the event loop and recovery, which is where the brief's difficulty actually sits.
- **Cost:** a printer (or tablet) is required. Stated as assumption A1.
- **Would change if:** the setting can't include a prepared sheet. Then: largest-quadrilateral contour for the outer border, then line detection for the inner grid, with the same downstream pipeline.

### D2. No X/O classification on the critical path
- **Considered:** a shape classifier or a VLM on every mark.
- **Why:** turn order already tells us which symbol to expect. Perception only has to answer "which empty cell got new ink, and is the scene stable." A cheap circularity check is advisory and only lowers confidence on mismatch.
- **Would change if:** games where the symbol matters independently of turn order (e.g. a player choosing which piece to place).

### D3. Classical CV on every frame; vision model only on low confidence
- **Considered:** sending frames to a VLM continuously or on every stable frame.
- **Why:** the per-frame question is binary and local ("did ink appear here"), which thresholding answers in milliseconds for free. A model is useful only when that answer is unclear, and then it gets a much easier task: the rectified board plus the known state. This gives ≈ $0 per game on the happy path and keeps latency independent of the network.
- **Would change if:** pieces are visually complex (chess, cards), where a model or detector is needed per read. Covered in `ARCHITECTURE.md` §4.6.

### D4. Stability gate before any evaluation
- **Considered:** evaluating every frame and filtering noise afterwards.
- **Why:** almost every false detection comes from a hand, pen or shadow in motion. Waiting for N quiet frames with all markers visible removes that whole class at a cost of ~0.6 s, which is inside the 1 s acknowledgement target. Two consistent stable reads before commit handles a pen lifted mid-stroke.
- **Tuned value:** TODO (stability_frames, thresholds as actually used)

### D5. Low confidence → model → human, never a silent guess
- **Why:** a wrong commit is worse than a question. The page and the state would diverge, and every later move would be built on it. The agent never changes committed state on its own; resolution comes from a consistent page, a model read that agrees with the ink data, or the human.
- **Consequence:** the game is fully playable with no API key; the model only reduces how often the human is asked.

### D6. Arm the agent's cell and wait for its ink
- **Why:** the agent's move isn't real until it's on paper. Arming one cell makes "the human drew it somewhere else" detectable and keeps page and state equal at every turn boundary, which is the brief's success criterion.

### D7. Minimax for decisions, not an LLM
- **Why:** perfect play, sub-10 ms, deterministic, fully unit-testable. An LLM would be slower, cost money and play worse. Ties broken by a fixed preference order so tests and demos are reproducible.

### D8. Spoken + displayed output, no voice input
- **Why:** the player is looking at the paper, so speech is the primary channel; the overlay confirms the cell visually (guards against mishearing) and makes the video readable. Voice input adds latency and a second perception problem for little gain in the time-box; recovery happens through the page, with keyboard `y`/`n` as a last resort.
- **Would change if:** hands-free play is a hard requirement. Then streaming ASR for yes/no/repeat only.

### D9. Replay harness over win rate as the evaluation method
- **Why:** win rate measures the human, not the agent, and a perfect player can't improve at play anyway. What can improve is perception, and that has to be measured on the same frames before and after a change. Recording plus replay also doubles as regression tests.

### D10. Keep the codebase small, one file per architecture component
- **Why:** the brief says any file may be pointed at. Each file maps to one box in the architecture diagram, and the rules and decision modules have no I/O so the "new game = rules module + perception spec" claim is visible in code.

### Build-time decisions
TODO: add entries as they happen, same format. Likely candidates: threshold calibration method, cell inset, how occlusion is detected, TTS threading, escalation prompt shape, anything that turned out different from the plan.

#### M1.1 Printable sheet is a PNG, not a PDF
- **Considered:** a PDF via `reportlab` or `img2pdf`, which guarantees physical size when printed.
- **Why:** the approved stack (`CLAUDE.md`) doesn't include a PDF library, and this milestone didn't seem worth asking to extend it for. A PNG printed from any image viewer preserves pixel dimensions; the real risk is a print dialog silently rescaling ("fit to page"). Mitigated with a 1-inch ruler mark printed on the sheet itself, checked against a real ruler.
- **Would change if:** the ruler check turns out to be unreliable in practice, or Gad would rather just add a small PDF dependency.
- **Note to Gad:** flagging this explicitly since `README.md` originally pointed at `assets/board.pdf`; I changed it to `assets/board.png` and generate it via `tools/make_sheet.py` rather than committing a binary. Happy to switch to a PDF if you'd rather.

#### M1.2 Marker layout and homography convention
- **Chosen:** four `DICT_4X4_50` markers, ids 0/1/2/3 assigned clockwise from top-left. The homography source points are each marker's board-facing inner corner (not its center), mapped to the four corners of the rectified image, so the 600×600 output is exactly the grid area (P1) with the markers themselves cropped out.
- **Considered:** using marker centers as the four homography points.
- **Why:** inner corners give a tighter, more precise fit for the grid PRODUCT.md's ink measurement (M2) will run on; centers would leave the rectified crop's exact framing dependent on marker size relative to the board, which is fiddly to keep aligned with the grid drawn on the sheet.
- **Would change if:** markers are frequently partially occluded near their inner corner specifically (center would be more robust to that one failure mode).

#### Process: AGENTS.md, CODE_REVIEW.md, and CI added alongside the milestone work
- **Chosen:** a separate `AGENTS.md` (branch/PR/test workflow) and `CODE_REVIEW.md` (review checklist: spec conformance, the `CLAUDE.md` hard rules, security, tests, docs, style), plus a minimal GitHub Actions workflow (`.github/workflows/tests.yml`) that runs `pytest` on every push/PR to `main`.
- **Considered:** folding this into `CLAUDE.md` itself.
- **Why separate files:** `CLAUDE.md` is the fixed build brief (what to build, milestone process); the git/PR/CI mechanics and the review checklist are a different concern read at a different time (opening a PR, reviewing one) and change on a different cadence, so keeping them apart avoids either doc growing unfocused.
- **Would change if:** Gad wants this folded into one doc, or wants GitHub branch protection actually turned on (that's a repo Settings change, not a file — noted as not-yet-done in `AGENTS.md`, needs someone with admin access to flip it).

#### Process: CI now runs a Python version matrix, pip-audit, and bandit; a real CVE was fixed
- **Chosen:** split CI into three required checks — `pytest` matrixed over Python 3.11/3.12, `pip-audit` (dependency vulnerability scan / SCA), and `bandit` (SAST) — plus least-privilege `permissions: contents: read` and a `concurrency` group so a new push cancels a stale run.
- **Considered:** leaving dependency/static-analysis scanning as a documented gap only (as the first cut of `COMPLIANCE.md` did).
- **Why:** these are ephemeral CI-only tools (installed inside the runner, never added to `pyproject.toml`'s `dependencies`), so they don't touch the approved runtime stack `CLAUDE.md` gates — closing two of `COMPLIANCE.md`'s SDLC gaps at effectively no cost.
- **What happened running it:** `pip-audit` found a real, live vulnerability — `PYSEC-2026-3447` in `setuptools` 79.0.1 (a build-time dependency, not one of the app's runtime deps) — before the check was ever wired into CI. Fixed by pinning `setuptools>=83` in `pyproject.toml`'s `[build-system]`; confirmed clean with a second `pip-audit` run. `bandit -r inkwatch tools` was clean on the first run (256 lines scanned at the time, M1's `perception.py` plus the two `tools/` scripts).
- **Would change if:** a future dependency legitimately can't be upgraded past a flagged version (e.g. no fix released yet) — then the specific finding gets an explicit, commented suppression in the workflow rather than turning the whole check off.

#### Packaging/runtime: pip + pyproject.toml (with a pip-tools lockfile), no Poetry, no Docker for the app
- **Chosen:** stay on plain `pip install -e .` against `pyproject.toml` for local dev and `README.md`'s quick start; add `requirements.txt`/`requirements-dev.txt` as pinned, reproducible locks (generated with `pip-tools`) for CI and for matching a reviewer's environment exactly, with a CI check that fails if they drift from `pyproject.toml` (see `AGENTS.md` "Dependency lockfile"). No Docker image for the interactive app.
- **Considered:** Poetry as the package/dependency manager; Docker for local dev and/or a "prod" deployment.
- **Why not Poetry:** it solves problems this project doesn't have — publishing to PyPI, a large/complex dependency graph needing a real SAT resolver. Six runtime dependencies and `pyproject.toml` + `pip` already cover it. What Poetry would have actually bought (a lockfile) is available more cheaply via `pip-tools`, without adopting a second package manager or asking `CLAUDE.md`'s "ask before adding anything else" question for something as heavyweight as a full tool swap.
- **Why not Docker for the app itself:** `PRODUCT.md` explicitly rules out a hosted service (non-goal) — there is no "prod" for this deliverable, only "a stranger's laptop," which is also what `README.md`'s 10-minute setup target assumes. The app needs simultaneous camera, GUI (`cv2.imshow`), and audio (TTS) access, all of which are fragile or unsupported through Docker's device/display passthrough (especially on macOS/Windows, where Docker Desktop runs in a VM with no native camera access). Containerizing would work against the setup-time goal, not for it.
- **Where Docker could still make sense:** `replay.py` (M5+) is headless — no camera, GUI, or TTS — so it's a legitimate, much narrower container use case (reproducible eval/regression runs) if the eval harness ever needs it. Not built now; noted for `NOTES.md` §6.
- **Would change if:** this project ever did become a hosted, multi-tenant product (`ARCHITECTURE.md`'s dashed future boxes) — then the *server-side* pieces (Session service, Model gateway) would need real deployment tooling, Docker very plausibly included. The camera-facing edge client would still run on the user's own device either way.

#### M1.3 BOARD_LOST hold-over is a perception-layer concern, session state machine not built yet
- `BoardTracker.update()` implements P1/P2 directly: it returns `found=False` once the 0.5 s hold window (`hold_seconds`) expires with fewer than 4 markers. The actual `BOARD_LOST` / `RESYNC` states in PRODUCT.md §8 belong to `session.py`, which isn't built yet (that's M3/M4). For now `tools/calibrate.py` just prints the found/lost status directly.

---

## 3. What I tried and dropped

TODO: fill honestly during the build. Format:

| Tried | What happened | Replaced with |
|---|---|---|
| | | |

---

## 4. Known limits

Stated up front; to be confirmed or corrected after testing.

| Limit | Why it exists | What it would take |
|---|---|---|
| Needs the printed marker sheet | D1: deterministic rectification over robustness to arbitrary grids | Contour + line-based grid finding as a fallback path |
| Faint pencil is unreliable | Ink ratio in the ambiguous band; escalates or asks often | Per-session calibration from the first marks; stronger local contrast normalization |
| Strong side lighting and hand shadows | Shadows change pixel darkness like ink does | Compare against baseline in a lighting-normalized space; shadow-invariant features |
| Erasing a mark is treated as an error, not a move | Assumption A5: marks only get added | Region-state perception (full re-read per turn), as needed for chess/checkers |
| One game, one camera, one person | Scope | Session service per connection, as in `ARCHITECTURE.md` |
| Agent always plays perfectly | Non-goal: no difficulty levels | Depth-limited or randomized policy |
| English only | Scope | TTS voice + cell vocabulary per locale |
| Measured on my desk, my lighting, my handwriting | Eval set size | More users and rooms in the recorded eval set |

TODO: add anything found during testing; remove anything that turned out not to be a problem.

---

## 5. How I used AI tools

The brief says the bar is directing the tool well and editing sharply. This section is the record.

### What I delegated
- Drafting the design documents (`PRODUCT.md`, `ARCHITECTURE.md`, this file's structure, `README.md`) from my direction, then reviewing and editing them.
- TODO: code generation for which modules, and how it was scoped (e.g. one module at a time against the requirement IDs in `PRODUCT.md`).

### What I kept for myself
- The core framing: perception as event detection, the model only on the low-confidence path, session as the single owner of truth.
- Every assumption and every cut in scope.
- Threshold tuning against my actual camera and lighting.
- TODO: confirm / extend.

### Where I overrode or rewrote the output
TODO: specific, honest examples. Format:

| What the AI produced | What was wrong or weak | What I did instead |
|---|---|---|
| | | |

### How I checked AI-written code
TODO: e.g. unit tests written against the spec rather than the implementation, replay test on recorded frames, reading every file before commit.

---

## 6. What I'd do next

In priority order, if this became a real product:

1. **Grow the eval set** across users, rooms and pens before changing anything else. Every other change depends on being able to measure it.
2. **Per-user calibration that persists across sessions**, validated on held-out recordings (see `PRODUCT.md` §17).
3. **Marker-free board finding** as a second path, so a hand-drawn grid works.
4. **Split perception and session into separate processes** with the `Observation` event as the wire contract, as in the architecture page.
5. **Second game with removals** (e.g. checkers) to force the region-state perception spec and prove the game-agnostic boundary is real rather than drawn.
6. **SOC 2 / ISO 27001 readiness** if this ever became a hosted, multi-tenant product — see `COMPLIANCE.md` for the gap analysis against both frameworks and the prioritized list of design changes that would actually matter (auth design for the per-tenant session service, vendor review of the model gateway's downstream provider, a real retention policy for `sessions/`). Explicitly not built now: `PRODUCT.md`'s non-goals rule out a hosted service for this take-home, and compliance is an audited property of an organization over time, not something a repo can claim on its own.

---

## 7. Open questions for Vistral

1. Is a prepared marker sheet an acceptable assumption, or is a freehand grid expected?
2. Should the agent ever let the human win?
3. Is page-based recovery enough, or is spoken confirmation from the human expected?
