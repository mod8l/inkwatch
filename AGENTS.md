# AGENTS.md: branches, PRs, tests, and review on Inkwatch

Companion to `CLAUDE.md` (what to build, milestone process) and
`CODE_REVIEW.md` (what a review checks). This file is the mechanics of
getting a change from a local edit into `main` — for a human contributor
or an AI coding agent alike.

## Branches

- `main` is the reference branch. Treat it as always-deployable: don't
  push directly to it, don't force-push it, don't rewrite its history.
- Do all work on a short-lived branch, one per coherent unit of work — in
  practice, one per milestone (`m1-sheet-rectification`,
  `m2-ink-detection`, ...) while M1–M7 are in progress, then one per
  fix/follow-up afterwards. Branch off an up-to-date `main`.
- Naming: `mN-<short-description>` for milestone work (e.g.
  `m2-per-cell-ink`), `fix-<short-description>` for post-milestone work.
- Commits stay small and milestone-prefixed, per `CLAUDE.md`'s commit
  convention (`M2: per-cell ink ratio with inset`) — one coherent step per
  commit, so review and `git bisect` are actually useful.
- Never rewrite a branch's history once it's pushed and may have been
  looked at by a reviewer — add new commits instead of amending or
  rebasing shared history. Rebasing your own not-yet-reviewed branch onto
  an updated `main` before opening the PR is fine.

## Pull requests

- Open a PR from your branch into `main` for every milestone, and for any
  follow-up change once the milestones are done.
- Don't merge your own PR. This is a human-evaluated take-home; `main`
  only moves forward with Gad's review, even if every check is green.
- PR description should cover, mirroring the milestone stop-and-review
  habit in `CLAUDE.md`:
  - What was built, against which requirement IDs / milestone.
  - The `pytest` summary (paste it).
  - What's untested or uncertain — especially anything needing the
    physical camera/paper that can't be verified from the branch, stated
    as "here's exactly what to try and what you should see."
  - Any doc changes made alongside (`PRODUCT.md` / `ARCHITECTURE.md` /
    `README.md` / `NOTES.md`) and why, if the implementation needed to
    diverge from what was written.
- Link the PR to the milestone it closes (M1–M7) in the description.
- Merge-commit or squash-merge is fine; avoid rebase-merge since it
  rewrites the commits that were actually reviewed.

## Tests

- Run `pytest` locally before opening a PR. Every test must pass — a red
  local run means the PR isn't ready to open.
- `tests/test_rules.py` and `tests/test_decision.py` need no camera and
  must always run clean.
- `tests/test_session.py` drives the state machine with synthetic
  observations (per `CLAUDE.md`'s testing section) — also camera-free,
  always runs.
- Perception logic that would otherwise need a real camera and real
  lighting is instead tested against synthetic or recorded frames (see
  `tests/test_perception.py` for the pattern), so the geometry/threshold
  logic is still checked by CI even though the actual camera/lighting
  behavior can't be.
- CI (`.github/workflows/tests.yml`) runs the full `pytest` suite on
  every push and on every PR targeting `main`. A PR shouldn't merge with
  a red check.

## Code review

- Every PR gets a code review before merge — see `CODE_REVIEW.md` for the
  checklist (spec conformance, hard-rule conformance, security, full test
  run, docs in sync).
- Review against the spec (`PRODUCT.md` requirement IDs, §9 edge cases),
  not just "does it run." This project's hard rules (`CLAUDE.md`) are the
  kind of thing that fails silently if nobody deliberately checks for it —
  e.g. a raw camera frame sneaking into an escalation call, or `rules.py`
  picking up an accidental `cv2` import.
- If the reviewer finds something, push more commits addressing it —
  don't force-push over the history that was already reviewed.

## Local setup (for review/CI parity)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Not yet set up

GitHub branch protection (require the CI check and a review before
`main` accepts a merge) isn't configured from this repo's code — that's a
repository setting, not a file, so it needs someone with admin access on
`github.com/<owner>/inkwatch` → Settings → Branches to turn it on. Until
then, the rules above are process, not an enforced gate.
