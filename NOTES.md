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
- **What happened running it:** `pip-audit` found a real, live vulnerability — `PYSEC-2026-3447` in `setuptools` 79.0.1 (a build-time dependency, not one of the app's runtime deps) — before the check was ever wired into CI. First pass "fixed" it by pinning `setuptools>=83` in `pyproject.toml`'s `[build-system]` and confirmed clean locally — but that pin only governs the isolated build-backend environment, not whatever `setuptools` a fresh venv or CI runner already has installed ambiently; the local "clean" result was actually from a separate `pip install --upgrade setuptools` run directly into that venv, not from the `pyproject.toml` edit. Once this went through actual CI (PR #1), `pip-audit` failed for real with the same finding, on the runner's own pre-installed `setuptools` 79.0.1. Real fix: an explicit `pip install --upgrade pip setuptools` step added to every CI job that installs the package (`.github/workflows/tests.yml`), reproduced and confirmed locally by installing `setuptools==79.0.1` into a clean venv first, then confirming the upgrade step alone (no `pyproject.toml` involved) took `pip-audit` from 1 finding to 0. `bandit -r inkwatch tools` was clean on the first run (256 lines scanned at the time, M1's `perception.py` plus the two `tools/` scripts).
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

#### M2.1 Ink measurement stays in `perception.py`; classification takes an explicit baseline, doesn't own one
- **Chosen:** `cell_bounds`, `ink_ratio`, `measure_cells`, `classify_cell`, `classify_cells` all added to `perception.py` per `PRODUCT.md` §14's own layout comment (`perception.py # markers, homography, stability, per-cell ink`), rather than a new module. Adaptive thresholding (`cv2.adaptiveThreshold`, Gaussian, block size ~half the cell) rather than a single global cutoff, so uneven lighting across the page doesn't bias one cell against another. `classify_cell`/`classify_cells` take the baseline ratio(s) as a plain argument; they don't store or own a baseline themselves.
- **Considered:** giving `BoardTracker` (or a new class) ownership of the "accepted baseline" mentioned in P4/D7.
- **Why not:** the baseline is session state — it changes only when `session.py` commits a move (D7) — and perception must never own game state (`CLAUDE.md` hard rule). Keeping `classify_cell` a pure function of `(ratio, baseline, low, high)` means the baseline can live in `session.py` once it exists, without perception needing to change.
- **Would change if:** M3 finds passing 9 baseline floats around every frame awkward; a small `Baseline` dataclass could wrap the list without changing where it's owned.

#### M2.2 `tools/calibrate.py` overlays raw ink ratios, not baseline-relative classification
- **Chosen:** the rectified view now draws each cell's inset box and its live `ink_ratio` (a plain number), matching the check `README.md` §4 already described ("a small ink number in each cell... draw a test mark, its number should jump").
- **Considered:** wiring in `classify_cell` against a snapshot-the-blank-board-as-baseline captured on a keypress, so the tool shows none/ambiguous/marked directly.
- **Why not now:** that needs a per-cell baseline and a capture UX, which is session-state territory (`session.py`, M3) rather than a debug tool's job; the raw-ratio view is what the setup check actually needs (numbers near zero, one jumps on a mark) and needed no new concepts.
- **Would change if:** manual testing on the real camera (`tools/calibrate.py`, 20 marks against `PRODUCT.md`'s M2 acceptance bar) shows raw ratios aren't legible enough to judge by eye; then add the baseline-snapshot + classify overlay.
- **Tuned value:** TODO — `ink_threshold_low`/`ink_threshold_high`/`cell_inset` as actually confirmed against the real camera and pen (currently the `config.yaml` defaults, unverified beyond synthetic tests).
- **Superseded by M2.3 below:** on review, M2's actual "done when" (`PRODUCT.md` §15: "correct per-cell none/ambiguous/marked on 20 manual marks") isn't checkable from raw ratios alone — it asks whether the *classification* is correct, not whether the number moves. Added the baseline snapshot after all.

#### M2.3 `tools/calibrate.py` gets a baseline snapshot (`b` key) and shows none/ambiguous/marked
- **Chosen:** press `b` with a blank sheet in view to capture the current 9 ratios as a baseline (a plain list local to the tool's `main()` loop); every frame after that calls the existing `classify_cells(ratios, baseline, low, high)` and the overlay shows the label (color-coded) next to the ratio instead of the ratio alone. Baseline starts at `[0.0] * 9` so the tool is still usable before pressing `b` (matches a freshly printed, un-inked sheet). Also exposes `--ink-low`/`--ink-high` as CLI flags mirroring `config.yaml`, so the actual manual 20-mark check can be run at the configured thresholds instead of only the module defaults.
- **Why now, not left for M3:** `PRODUCT.md`'s M2 milestone is specifically "done when" a human can confirm classification, not ink measurement, is correct on real marks. `session.py` owning the *game* baseline (updated on each committed move, D7) is still M3+; this is a separate, throwaway baseline that lives only in the calibration tool's own loop for the length of one manual check — it doesn't give `perception.py` or the tool ownership of any game state, so the perception/session boundary (`CLAUDE.md` hard rule) is unaffected.
- **Would change if:** the manual check on a real camera shows the zero-baseline default is confusing before the first `b` press (e.g. paper texture already reads non-trivial ink and cells show `ambiguous`/`marked` on a blank sheet) — would then default to capturing on the tool's first frame automatically instead of waiting for a keypress.

#### M3.1 `decision.py` imports `rules.py` (one-way, not a cycle); CLAUDE.md's hard rule read as isolating the pure pair from the I/O modules, not from each other
- **The question:** CLAUDE.md says `rules.py` and `decision.py` have "no imports from other `inkwatch` modules except `events.py` types." Taken literally, `decision.py` couldn't import `rules.py` — but minimax (G3) has to walk the game tree with `rules.py`'s legal-move/apply-move/terminal logic at every node, not just the root's legal-move list. Re-implementing that inside `decision.py` too would duplicate the rules engine and risk the two disagreeing about what's legal.
- **Asked Gad directly rather than guessing** (this is exactly the "stop and say so" case, not a call I should make silently on my own hard rule). **Chosen:** yes, `decision.py` can import `rules.py`; the rule is read as isolating this pure pair from the I/O-having modules (`session.py`, `perception.py`, `output.py`, `escalation.py`), not from each other. Both stay free of `cv2`, file/network I/O, and any import of an I/O module.
- **One-directional, not circular:** the dependency only runs one way — `decision.py` imports from `rules.py`; `rules.py` imports nothing from `inkwatch` at all and has no idea `decision.py` exists. (Earlier phrasing here and in the M3 commit message said "import each other," which reads as a cycle — it isn't one; `session.py` will import `rules.py` the same one-way for the same reason.)
- **Would change if:** Gad later wants the literal reading (decision.py imports nothing from rules.py) — then decision.py needs its own minimal, duplicated terminal/legal-move check, accepting two places that must agree on tic-tac-toe rules.

#### M3.2 Stability gating (P5, P6) lands in `perception.py`, fixing an unresolved doc/code mismatch from M1
- **Found on review:** M1's `perception.py` docstring said stability gating belongs in `session.py` ("M3+"), but `PRODUCT.md` §14's own repo-layout line assigns `perception.py` "markers, homography, **stability**, per-cell ink," and neither doc was ever reconciled with the other — a silent divergence CLAUDE.md says to stop and flag, not carry forward.
- **Chosen:** implemented as `StabilityGate` in `perception.py` (P5: N consecutive quiet frames with all markers currently visible; P6: motion above threshold or missing/held-over markers marks the scene occluded), matching §14. `Perceiver` composes `BoardTracker` + `StabilityGate` + ink measurement into one `observe()` call that returns an `Observation` (`events.py`) — the one thing `session.py` (or `__main__.py` on its behalf) calls per frame. `perception.py`'s docstring is corrected to match.
- **Why here and not session.py:** stability is a property of the pixel stream (is the scene moving, are markers visible) — the same kind of per-frame, stateful-but-not-game-state tracking `BoardTracker` already does for the homography hold-over (P2). D2/D3's confidence rules (does *this* classified frame count as a move) stay in `session.py`, since that's genuinely state-machine logic.
- **New config value:** `motion_threshold: 2.0` (mean abs pixel diff, 0-255 scale, between consecutive rectified frames) added to `config.yaml`/`PRODUCT.md` §12 — there was no prior tunable for inter-frame motion. Unverified beyond synthetic tests, same caveat as the ink thresholds.

#### M3.3 EVALUATE and THINK are folded into the same `update()` call, not persisted as their own resting phase
- **Chosen:** `session.py`'s `Phase` enum matches `PRODUCT.md` §8 in full (all 10 states), but `EVALUATE` and `THINK` never actually persist between frames in the implementation — a stable observation is evaluated and, on a high-confidence read, committed within the same `update()` call; committing the human's move immediately computes and arms the agent's reply in that same call too, matching §6.2's "immediately." Both phases have no external input of their own (no observation is needed to decide the agent's move), so there's nothing for the state machine to actually wait on in between.
- **Consequence:** the two "You played X" / "I'll take Y" utterances that §6.2 describes as two steps are spoken as one combined utterance, not two separate `Speaker.say()` calls — calling `say()` twice back-to-back would race with `Speaker`'s own O2 cancel-unsaid-utterance behavior and could drop the first one before it's ever heard.
- **Would change if:** M4/M5 give THINK real latency worth showing (e.g. a visible "thinking" state during an escalation call) — then it would need to become a real, persisted phase.

#### M3.4 `Speaker` takes an injectable `speak_fn`; overlay/vocabulary logic lives in `output.py`, not `session.py`
- **Chosen:** `output.py`'s `Speaker` wraps a background thread + queue around a `speak_fn: Callable[[str], None]`, defaulting to `pyttsx3` with a `say`-command fallback on macOS, but overridable — tests inject a fake `speak_fn` to check queuing/cancellation (O2) and non-blocking `say()` (O1) without a real TTS engine or audio hardware (neither exists in CI or this sandbox). `session.py` never touches `Speaker` or `cv2` directly: `Session.update()` returns a `SessionResult` (message text, phase, board, target cell); `__main__.py` is the only thing that calls `Speaker.say()` and `draw_overlay()`.
- **Why:** keeps `test_session.py` a pure, synthetic-observation test of the state machine (CLAUDE.md's testing section), with no TTS engine or `cv2.imshow` window to mock out. `cell_name()`/`describe_line()` (§6.3 vocabulary) live in `output.py` since they're a presentation concern, imported by `session.py` to build message text.
- **Would change if:** a future milestone wants session to control speech timing more precisely (e.g. cutting off a stale utterance mid-turn) — then `Session` might need a narrower speech-control interface rather than just returning strings.

#### M3.5 What M3 deliberately leaves for M4/M5 (not a silent gap — tracked here and in `session.py`'s docstring)
- Anything that isn't a clean D2 high-confidence read (two marks at once, an ambiguous cell, ink in an occupied cell, ink outside the agent's armed cell) currently just **waits** for a cleaner read rather than escalating or asking — never guesses, per the hard rule, but doesn't yet recover with a spoken question either. `ESCALATE`/`ASK_HUMAN`/`BOARD_LOST`/`RESYNC` are defined in `Phase` (matching §8) but `Session.update()` never sets them yet.
- G5's post-game full-board re-read (verifying the final page matches the reported state) isn't implemented — `_commit`'s terminal check only looks at the board `session.py` already believes, not a fresh read of all nine cells.
- `--record` and `--no-escalation` are accepted by `__main__.py` for forward compatibility with the README's documented flags, but `--record` currently just prints that it isn't built yet (M5); `--no-escalation` is a no-op since there's no escalation path to disable yet.
- **Tuned value:** TODO — `motion_threshold` and the debounce/reminder timings, as actually confirmed against a real camera and a real hand drawing (currently config defaults, unverified beyond synthetic `Observation`s and synthetic frames).

#### M4.1 `ESCALATE` has no model yet, so it's a deliberate one-frame stub, not a stand-in for M5
- **Chosen:** the two situations §9 says should "Escalate" first (two new marks, a persistently ambiguous cell) enter `Phase.ESCALATE` for exactly one `update()` beat (silent, `confidence="escalating"` so the overlay shows it), then fall straight through to `ASK_HUMAN` on the next call. There's nowhere for a real model call to await a result yet.
- **Why this shape, not skipping ESCALATE entirely:** §9 explicitly lists "vision model timeout or no API key → skip straight to asking the human" as its own specified behavior, not an unhandled gap. M4 with no `escalation.py` wired in is permanently in that state, so implementing ESCALATE as "always takes the no-model fallback path" is the *correct* M4 behavior per the spec, not a placeholder to feel bad about. It also leaves the obvious seam for M5: `_resolve_escalate` is the one place a real model call goes, and it already knows which situation (`_ask_context`) and which cells (`_ask_cells`) to send.
- **Would change if:** M5 wants ESCALATE to persist across multiple frames while a model call is in flight (e.g. to show a "checking..." state for longer than one beat) — today's one-beat version assumes the call will be near-instant relative to a frame, which won't be true for a real network round trip.

#### M4.2 `ASK_HUMAN` resolves through the page, not through keyboard `y`/`n`
- **Considered:** wiring `ARCHITECTURE.md`'s "keyboard y/n, last-resort answer" into `__main__.py` for M4, since §9's edge cases are explicitly the milestone's scope.
- **Why not:** every `ASK_HUMAN` case this build produces has a page-based resolution that's already fully specified and testable: two marks at once resolves when only one is left on the page (or none, meaning withdrawn); an ambiguous/shadowed cell resolves the same way normal D2/D6 confidence already does, once conditions clear; a wrong-cell mark resolves once the *armed* cell gets ink (D5: never guesses that the stray mark was meant as the move); RESYNC resolves once the page matches state again. D8 already frames the page as the primary channel and keyboard as "last resort" — none of M4's cases actually need the last resort. "Which of two marks is your move?" doesn't map onto `y`/`n` cleanly anyway (it's not a yes/no question), so building that well is a separable piece of work, not a few extra lines bolted onto this milestone.
- **Consequence:** `README.md`'s controls table documents `y`/`n` as **not built**, and flags a pre-existing name collision in that same table (a `n` "new game" key and a `y`/`n` "answer" key can't both be literally `n`) for whoever picks this up.
- **Confirmed with Gad:** wait for M5. Only one of the three `ASK_HUMAN` cases (the persistent-ambiguous/shadow one) is actually a yes/no question ("is that your move in {cell}?"); the other two ("which of two cells" and "please fix the page") don't map onto `y`/`n` at all, so this isn't a uniform "add y/n" feature — it's a scoped addition worth building alongside M5's real escalation model rather than half now, half later.

#### M4.3 RESYNC and the terminal re-read (G5) both classify against a second, never-moved baseline
- **Chosen:** `session.py` now keeps two baselines: `baseline` (moves forward on every commit, per D7 — used for D2/D3's frame-to-frame delta classification) and a new `_blank_baseline` (captured once, the moment CALIBRATING first sees the board, and never touched again). RESYNC and G5's post-game re-read both classify the current ratios against `_blank_baseline` and compare *absolute* ink presence per cell to what `session.board` says is occupied, rather than trusting `baseline`, which §8 itself calls "no longer trustworthy" once the page has been lost or bumped.
- **Considered:** having `perception.py` own this second baseline and expose an extra `Observation` field (e.g. `absolute_marks`) instead of `session.py` reaching for `classify_cells` itself.
- **Why session.py, not a new Observation field:** `PRODUCT.md` §14 already describes `session.py` as "state machine, **reconciler**, baseline ownership" — RESYNC/G5 are exactly that reconciliation job, and `classify_cell(s)` (unlike the rest of `perception.py`) is pure — no `cv2`, no I/O. **Correction while writing this up:** I'd first written this as "keeps session.py import-time cv2-free, matching the same local-import precedent in `output.py`'s `draw_overlay`" — that's wrong on inspection. `session.py` already imports `output.py` at module level for `cell_name`/`describe_line`, and `output.py` imports `cv2` at *its* top level, so `cv2` is already loaded transitively the moment `session.py` is imported, local import or not (checked: `import inkwatch.session; 'cv2' in sys.modules` is `True`). `output.py`'s own comment on its `cell_bounds` import makes the identical false claim, pre-existing from M3 — not fixed here since it's outside this milestone's diff, but flagged in the Known limits table below for whoever touches that file next. The local import in `_resync_mismatches` is kept anyway, for the more modest reason actually stated in its comment: it keeps perception's larger surface out of `session.py`'s own import list. `test_session.py` needing no real camera was never actually about avoiding the `cv2` import (which already happens); it's about not needing a working camera device or GUI, which is unaffected either way.
- **Would change if:** a real camera bump also shifts the homography enough that `_blank_baseline`'s absolute ratios stop being comparable (e.g. lighting changed too) — untested beyond synthetic `Observation`s, flagged below.

#### M4.4 Two divergences from a literal reading of §8's diagram — one confirmed and folded back into PRODUCT.md, one still just noted here
- `RESYNC --> WAIT_HUMAN` was the diagram's only drawn edge out of a successful resync. If the board was lost mid-`WAIT_AGENT_INK` (the agent's move already armed and announced), resuming into `WAIT_HUMAN` would silently drop that armed turn and wait on the wrong side — worse than what it's supposed to fix. `_resume_after_resync` instead resumes whichever phase matches whose turn it already was, re-arming the *same* target cell (no new `best_move` call) rather than picking a fresh one. **Flagged per CLAUDE.md ("stop and say so"), Gad confirmed it — `PRODUCT.md` §8's diagram and prose are updated in the same commit** to draw `RESYNC --> WAIT_AGENT_INK` alongside `RESYNC --> WAIT_HUMAN`, so the diagram and the code now agree; this is no longer a divergence.
- The diagram still doesn't draw `CALIBRATING --> BOARD_LOST` as reachable, and this build agrees by omission: there's no committed state to lose before the first board is even found, and `__main__.py`'s existing "board not found" banner already covers that wait. `CALIBRATING` stays immune to `BOARD_LOST`. Not yet raised as its own explicit question to Gad — noted here rather than silently changing the diagram a second time.

#### M4.5 `n` (new game) and `r` (force a re-read) are trivial to wire in immediately, `y`/`n` isn't
- **Why now:** `README.md` already documented these three keys as a set with a shared "Built?" column; `n` is just constructing a fresh `Session(...)` with the same startup config, and `r` is a one-line `Session.force_resync()` that jumps straight to the same `RESYNC` phase a lost board would — both reuse machinery this milestone already built, at effectively zero incremental cost. `y`/`n` is the one that needed its own design discussion (see M4.2), so it's the one actually deferred.

#### M5.1 `session.py` still never touches the network or a rectified crop — `escalation.py`'s result crosses the boundary as a plain `EscalationOutcome`, the same way `Observation` already does
- **The constraint:** `events.py`'s docstring (M1) already commits to `Observation` never carrying raw pixels, "raw frames never cross to the session" (ARCHITECTURE.md §2). D4 needs the rectified crop sent to a vision model, but session.py is the only thing that knows *when* to escalate (it owns the confidence logic) and never sees pixels at all.
- **Chosen:** entering `ESCALATE` (`Session._enter_escalate`) is silent and one-beat, same shape as M4's stub. `SessionResult` gains `escalation_cells` (which cells need an opinion); the caller — `__main__.py` or `replay.py`, never session.py — is expected to see `phase == ESCALATE`, call `escalation.py`'s `Escalator.ask(crop, board, escalation_cells)` with the rectified crop it already has, and hand the result to a new `Session.apply_escalation(outcome, now)` before the next frame. `outcome` is `EscalationOutcome` (events.py): `cell`, `error`, `latency_s`, `cost` — a plain data type, not a raw image, so the "no pixels to session" rule holds even though the model call is now real.
- **Why not have session.py call `escalation.py` itself:** it would need the rectified crop as an argument either way (defeating the point), and would pull network I/O and a blocking 3 s call into the one module this whole build has kept synchronous, pure, and camera-free-to-test. `test_session.py` still needs no network, no `httpx`, no mock model — it drives `apply_escalation` with a synthetic `EscalationOutcome` exactly like it drives `update()` with a synthetic `Observation`.
- **The safety net:** if a caller enters `ESCALATE` and never calls `apply_escalation` before the next `update()` (no escalator wired in, a bug, whatever), `update()` itself auto-resolves with an `EscalationOutcome(cell=None, error="not_applied")` — the same "skip straight to asking" fallback M4 always used. The session can never get stuck waiting on a caller that doesn't cooperate.

#### M5.2 `run_frame` (in `replay.py`, imported by `__main__.py`) is the one place "perception → session → escalate if needed" is written
- **Considered:** keeping `__main__.py`'s loop and `replay.py`'s offline pipeline as two separate implementations of that sequence, since they were already separate files.
- **Why not:** L4's whole point is checking a change against *the same frames* a real game produced — that only means something if replay runs the identical per-frame logic the live loop does. Two hand-maintained copies would drift the first time either one changed (a new escalate condition, a different confidence field) without the other noticing.
- **Chosen:** `run_frame(perceiver, session, frame, now, escalator=...)` lives in `replay.py` (since replay's whole job already is "run this pipeline over frames without a camera") and `__main__.py` imports it. Slightly backwards-looking naming (`__main__.py` depending on `replay.py`), but the alternative — a third shared module neither file is really "about" — seemed like more indirection for a three-line function.
- **Would change if:** `run_frame` grows enough non-replay-specific concerns that it stops feeling like it belongs in a file called `replay.py`; a small `pipeline.py` would be the natural extraction point.

#### M5.3 D5's "consistent with the ink data" gate: the model's cell has to be one perception already flagged
- **Chosen:** `Session._resolve_escalate` accepts `outcome.cell` only if it's a member of `self._ask_cells` (the exact candidate set that triggered this escalation) and the cell is still empty. Any other answer — a cell outside the candidates, `None`, or a failure — asks the human.
- **Why:** the model only ever sees one cropped, possibly low-quality photo; treating "which of these flagged cells is it" as a multiple-choice question (rather than trusting the model to name an arbitrary cell from scratch) is a much easier, more reliable task, and it structurally can't let a hallucinated answer commit a move nowhere near what perception actually saw. This is the literal reading of D5's "consistent with the ink data," not just an implementation convenience.
- **Would change if:** the two-marks case wants the model to distinguish "actually neither of these, it's a third cell" — not built, since perception's own ink-delta scan already found every candidate cell before escalating; a cell the model names that perception didn't flag is far more likely a hallucination than a real ink shape than perception's own thresholding missed.

#### M5.4 Escalation is a synchronous, blocking call in the frame loop — not threaded
- **Considered:** running `Escalator.ask()` on a background thread (like `Speaker`'s TTS queue) so camera capture and display keep running during the up-to-3 s model call.
- **Why not (for now):** CLAUDE.md's hard rule is specific to TTS ("TTS must not block the frame loop"); nothing in PRODUCT.md requires escalation itself to be non-blocking, and D4's own "Timeout 3 s" reads as an acceptable synchronous wait, not a background one. Escalation is also rare by design (target < 5% of turns), so the UX cost is a brief pause on an already-uncommon path, not a steady-state problem. Threading it would mean `Session` gaining a "waiting on escalation" resting phase (unlike today's one-beat `ESCALATE`), a cancellation story if the page changes mid-call, and a way to keep drawing/speaking while a call is in flight — real complexity for a path this build's own acceptance criteria don't require to be smooth.
- **Consequence:** during the (rare) escalation path, the preview window and camera capture visibly pause for up to `timeout_s` (default 3 s). Not measured against a real camera yet.
- **Would change if:** manual testing on a real camera finds this pause worse in practice than it reads on paper (e.g., because the camera buffer backs up and the next frame is stale) — then threading this the same way `Speaker` already is would be the fix, with `Session` gaining a real "awaiting escalation" phase to match.

#### M5.5 Logging (L1) is one JSONL line per *meaningful* tick, not every frame
- **Considered:** logging every single `session.update()` call, matching L1's literal "observations" wording.
- **Why not:** a real game runs at whatever the camera's frame rate is for however many minutes it takes to play; logging every quiet, nothing-changed frame while the session is just waiting would make the log enormous and mostly noise — nothing distinguishes frame 400 from frame 401 while both just say "still waiting." `__main__.py`'s `_log_tick` logs a `"start"` event on the first tick, a `"commit"` whenever the board changes, a `"question"` on entering `ASK_HUMAN`, and a `"result"` on `GAME_OVER` — every phase transition and commit L1 actually cares about, without a line for every camera frame in between. Escalations are logged separately, right where `run_frame` returns the real `EscalationOutcome` (see M5.1/M5.2) — by the time a resolved escalation reaches `_log_tick`, `phase` has already moved past `ESCALATE`, so `_log_tick` itself would never see it anyway.
- **Would change if:** the eval harness (M6) turns out to need frame-level ink-ratio history for a metric, not just the committed events — then a separate, opt-in raw-observation stream (distinct from this event log) would be the addition, not changing this log's shape.

#### M5.6 `--record` saves one PNG per raw frame, not a video container
- **Considered:** `cv2.VideoWriter` (an `.mp4`/`.avi`).
- **Why not:** video codecs are the single most platform/build-dependent part of an OpenCV install (missing codecs, container support varying by OS and how `opencv-contrib-python` was built) — exactly the kind of setup fragility `README.md`'s "stranger, ten minutes" target is trying to avoid. A directory of timestamped PNGs plus a `manifest.jsonl` needs nothing beyond what's already a hard dependency (`cv2.imwrite`/`cv2.imread`), is trivially diffable/inspectable by hand, and reads back through the exact same `cv2.imread` path `replay.py` already uses for `--record`'s output.
- **Cost:** far more disk space and inodes than a compressed video for the same length of footage, especially since **every** raw frame is saved (not throttled) — throttling would silently change what `StabilityGate` sees on replay (its quiet-frame count is a frame count, not a duration), which would break L4's actual point of replaying *the same* frames. Flagged as a known limit below.
- **Would change if:** disk usage on a real multi-minute game turns out to be a real problem (not yet measured) — the fix would be a compact per-frame format (e.g. JPEG instead of PNG) rather than throttling, to keep frame-for-frame replay fidelity.

#### M5.7 Escalation "cost" in the log is a token count, not a dollar figure
- **Chosen:** `EscalationOutcome.cost` is `usage.total_tokens` from OpenRouter's response when present, `0.0` otherwise — not converted to a dollar amount.
- **Why:** a real cost needs a per-model price table (prompt vs. completion token rates, which vary by model and change over time); hardcoding one `google/gemini-flash` rate would silently go stale or be wrong the moment `config.yaml`'s `escalation.model` changes. A token count is at least honest about what's actually known from the response.
- **Would change if:** M6's metrics pass wants a real "$ per game" number — then either a small, explicitly-dated price table keyed by model id, or reading price data OpenRouter's own API can provide per-model.

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
| No keyboard `y`/`n` answer channel | Deferred — every `ASK_HUMAN` case resolves through the page instead (M4.2); "which of two cells" doesn't map onto yes/no cleanly | A real answer channel, or reword multi-choice questions as a yes/no sequence |
| RESYNC/G5's absolute-ink check assumes the blank-baseline snapshot from calibration is still valid | Only one blank baseline is ever captured, at CALIBRATING; a lighting change after a page bump isn't accounted for (M4.3) | Re-capture a blank baseline whenever a bump is severe enough, or normalize for lighting instead of using a raw ratio delta |
| §9's persistent-ambiguous-cell escalation threshold (`AMBIGUOUS_ESCALATE_READS = 3`) and the occlusion reminder's 15 s aren't confirmed against a real camera | Same as the other tuned values (D4) — set from the spec's own numbers where it gives one, or a reasonable guess otherwise | Manual test with a real hand, real shadows; adjust once measured |
| `output.py`'s comment on its `cell_bounds` import claims it "avoids a hard cv2 dependency" | Wrong — `output.py` imports `cv2` at its own top level, so that's already true regardless (found while writing M4.3) | A one-line comment fix; not touched here since it's outside M4's actual diff |
| Escalation blocks the frame loop for up to `timeout_s` (default 3 s) | Synchronous call, not threaded (M5.4) — a deliberate scope cut, not an oversight | Run it on a background thread like `Speaker`, with `Session` gaining a real "awaiting escalation" phase |
| `escalation.py`'s prompt and reply-parsing (`_parse_cell_reply`) are only tested against a mocked HTTP transport, never a real vision model | No API key configured for this build environment; `test_escalation.py` proves the request/response *plumbing*, not that a real model reliably answers with a bare cell number the way the prompt asks | Manual test with a real `OPENROUTER_API_KEY` and a real ambiguous mark; adjust the prompt wording if replies come back noisier than expected |
| `--record` saves one PNG per raw camera frame (no throttling, no video container) | Keeps replay frame-for-frame faithful to `StabilityGate`'s quiet-frame counting (M5.6); trades disk space for that | A compact per-frame format (e.g. JPEG) if disk usage on a real game turns out to be a problem |
| Escalation "cost" in the log is a token count (`usage.total_tokens`), not a dollar figure | No per-model price table wired in (M5.7) | A small, dated price table keyed by model id, or a real pricing lookup |
| `SessionLogger`/`--record`/L2 frame-saving are only exercised by direct unit calls (see the manual smoke-test note in this milestone's summary), not through a full live `__main__.py` run | `__main__.py` needs a real camera per CLAUDE.md, so its own loop can't be automated-tested; the logging/recording helper functions it calls are plain functions and were checked directly instead | Confirm on a real camera: `sessions/<timestamp>/events.jsonl`, `frames/`, and (with `--record`) `raw/`+`manifest.jsonl` all appear and `python -m inkwatch.replay` runs cleanly on them |

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
