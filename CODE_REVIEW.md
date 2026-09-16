# CODE_REVIEW.md: review checklist for Inkwatch

Use this for every PR before merging into `main` (see `AGENTS.md` for the
branch/PR mechanics this plugs into). Check every box that applies; call
out anything skipped and why in the review. This is deliberately stricter
than "does it run" — most of this project's hard rules (`CLAUDE.md`) fail
silently unless someone checks for them on purpose.

## 1. Spec conformance

- [ ] Changed behavior maps to specific requirement IDs in `PRODUCT.md`
      (P#, D#, G#, O#, L#) — cite them in the PR description.
- [ ] If this PR covers part of §9 (edge cases), each relevant row's
      "Agent behavior" is what the code actually does — verified by a
      test, not just by reading the code.
- [ ] If the code needed to differ from `PRODUCT.md` / `ARCHITECTURE.md`
      / `README.md`, the doc was updated **in the same PR**, not left to
      diverge silently (`CLAUDE.md`'s explicit rule).
- [ ] State machine changes match `PRODUCT.md` §8 exactly — no new
      transitions invented ad hoc.

## 2. Hard rules (CLAUDE.md) — check these explicitly, they're easy to break by accident

- [ ] No secret appears in a diff, log line, `print()`, or exception
      message. `OPENROUTER_API_KEY` is only ever read from `.env` via
      `python-dotenv`.
- [ ] `rules.py` and `decision.py` have zero `import cv2`, zero file/
      network I/O, and import nothing from other `inkwatch` modules
      except types from `events.py`.
      (`grep -n "^import\|^from" inkwatch/rules.py inkwatch/decision.py`
      — every line should be stdlib or `events`.)
- [ ] `perception.py` never writes to session/game state directly — it
      only returns `Observation`-shaped data; only `session.py` commits a
      move.
- [ ] No code path lets the agent change already-committed state without
      going through escalate → ask-human. Grep for state mutation outside
      `session.py`.
- [ ] The game still runs start-to-finish with no `OPENROUTER_API_KEY`
      set and no network (`--no-escalation`, or unset the key and confirm
      escalation disables itself per config).
- [ ] TTS calls are non-blocking — the frame loop doesn't stall on
      `output.py`; confirm it's on its own thread/queue, not a synchronous
      call in the hot path.
- [ ] Every payload sent to a model (`escalation.py`) is a rectified
      board crop, never a raw camera frame — check the actual bytes being
      sent, not just the function name.

## 3. Security

- [ ] Dependencies are exactly the approved stack (`CLAUDE.md` "Stack")
      or were explicitly asked about and agreed; no silent new dependency
      in `pyproject.toml`.
- [ ] No `eval` / `exec` / `pickle.loads` on anything derived from a
      frame, a model response, or a file on disk.
- [ ] `httpx` calls to OpenRouter (`escalation.py`) have an explicit
      timeout (`PRODUCT.md` D4: 3s), don't follow redirects to arbitrary
      hosts, and the URL isn't built from unsanitized input.
- [ ] Anything written under `sessions/` (logs, frames) never includes
      the API key or other secrets; `sessions/` stays git-ignored.
- [ ] File paths built from CLI args or config (`--camera`, `--record`,
      session IDs consumed by `replay.py`) can't escape the intended
      `sessions/` / `assets/` directories (no unsanitized `..` traversal).
- [ ] Any `subprocess` / `os.system` call (e.g. the macOS `say` fallback)
      uses an argument list, not a shell string built from variable
      input.
- [ ] No verbose/debug mode that would print camera frames, keys, or full
      request/response bodies (including the key) during normal
      operation.

## 4. Tests

- [ ] `pytest` run in full, output pasted into the PR, all green.
- [ ] New behavior has a new test named after the behavior
      (`test_two_new_marks_escalates`, not `test_evaluate_2`), per
      `CLAUDE.md`.
- [ ] Tests were written from the requirement IDs / spec, not derived
      from reading the implementation and asserting what it happens to
      do.
- [ ] `test_decision.py` still proves the agent never loses from any
      reachable position (exhaustive over reachable states, not a
      handful of examples).
- [ ] `test_session.py` still covers every §9 row that doesn't need a
      camera.
- [ ] Anything that does need the camera/paper is called out in the PR
      description as manually verified (with exactly what to try and
      what to expect to see) or explicitly left unverified — never marked
      done without one or the other.

## 5. Non-functional targets (PRODUCT.md §10) — spot-check when the change touches the hot path

- [ ] Per-frame perception work is still cheap: no new per-frame model or
      network call, no unbounded loop.
- [ ] Nothing added to the escalation path risks the 3s timeout or the
      5-calls-per-game budget.
- [ ] No new blocking call on the main frame loop thread.

## 6. Docs and notes

- [ ] `NOTES.md` "Build-time decisions" has an entry for any nontrivial
      choice made in this PR (chosen / considered / why / would change
      if).
- [ ] `NOTES.md` §3 has an entry for anything tried and dropped.
- [ ] `NOTES.md` §5 ("How I used AI tools") and the time-spent line were
      **not** touched — that's Gad's to write.
- [ ] No number appears anywhere (`README.md`'s measured-results table,
      `NOTES.md`) that wasn't produced by an actual run in this PR;
      unmeasured values stay `TODO`.

## 7. Style

- [ ] Code stays boring and explainable — no cleverness Gad couldn't
      defend line-by-line if asked about it directly.
- [ ] One file per architecture component (`ARCHITECTURE.md`); no
      component's logic leaking into another file.
- [ ] No premature abstraction for hypothetical future games — this is
      tic-tac-toe; the "game-agnostic" boundary is the existing
      file/type boundary, not extra indirection layered on top.

---

Sign off in the PR review with which sections were checked, which were
skipped (and why — usually "needs a camera"), and any follow-up items
filed.
