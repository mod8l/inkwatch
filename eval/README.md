# Eval set: recording, labeling, scoring

This is the M6 "Evidence" milestone. Everything in here needs the real
camera and the real page — none of it can be done from the sandbox this
was built in, so `inkwatch/metrics.py` and its tests are as far as this
build gets on its own. What's below is exactly what to do next.

## 1. Record at least 10 games

```bash
python -m inkwatch --record
```

Note the printed session id (`sessions/<timestamp>/`) for each game. Per
`PRODUCT.md` §13.1, the eval set should deliberately include each of
these at least once, not just 10 ordinary games:

- two marks drawn in the same beat
- a mark in the wrong cell
- the page bumped or rotated mid-game
- a lingering hand over the board
- a shadow or lighting change over a cell

Play some of them with escalation disabled (`--no-escalation`) too, since
that path needs to stay provably playable (§13.2's acceptance checklist).

Also record video (phone, screen capture, whatever) of at least one game
for the submission's acceptance criteria — that's a separate ask from
this scoring pipeline and isn't covered by anything below.

## 2. Label ground truth for each game

For each `sessions/<timestamp>/`, write `eval/labels/<timestamp>.yaml` by
hand, watching what's actually in `sessions/<timestamp>/frames/` (and
`raw/`, since this was recorded with `--record`) against what really
happened on the page. See `labels/example.yaml` for the schema; the
short version:

```yaml
session_id: "<timestamp>"        # must match the sessions/ directory name
true_moves:                      # every real mark, human and agent, in order
  - cell: 4                      # 0-8, row-major
    hand_left_frame_ts: 172731.842   # see below
false_trigger_commits: 0         # commits in events.jsonl with no real mark behind them
final_board: [X, null, O, null, X, null, null, null, O]   # the page's real final state
notes: "lingering hand over cell 4 partway through"
```

**`hand_left_frame_ts`** is the `frame_ts` value of the frame where the
hand has just cleared the page after drawing that mark — read it straight
off the filename of the matching image under `sessions/<timestamp>/frames/`
or `raw/` (e.g. `commit_172731.842.png` → `172731.842`). It's the same
clock `events.jsonl` already uses (`time.monotonic()` from that game's
own run), so there's no video-to-log time conversion to get wrong —
just read the number off the right frame.

**`final_board`** is the real page's state at the end of the game
(9 entries, `X`/`O`/`null`), used to check for end-of-game desync (G5).

**`false_trigger_commits`** is a plain count from watching the game: how
many `commit` events in `events.jsonl` correspond to no real mark on the
page. This is written down directly rather than re-derived from the
frames, since deriving it would mean re-running some version of the same
perception logic being evaluated.

## 3. Score it

```bash
python -m inkwatch.metrics sessions/<timestamp-1> sessions/<timestamp-2> ...
```

Reads each session's `events.jsonl` (what the live game actually logged,
not a replay) against `eval/labels/<timestamp>.yaml`, and prints the
`PRODUCT.md` §13.1 table with real numbers. Paste it into README.md's
"Measured results" section, replacing the `TODO`.

A session with no matching label file is skipped (with a note on
stderr), so it's fine to run this over `sessions/*` even before every
game is labeled.

## 4. What this can't check

- **Cost per game** is reported as an average token count
  (`escalation.py` logs `usage.total_tokens`, not a dollar figure — see
  `NOTES.md` D-M5.7). Convert to dollars yourself using OpenRouter's
  current price for whatever `config.yaml`'s `escalation.model` is set
  to, if a $ figure is wanted; not built here since a hardcoded price
  table would just go stale.
- Everything else in `PRODUCT.md` §13.2's acceptance checklist that isn't
  one of the six §13.1 metrics (the demo video, the README dry-run timing,
  known limits being written up honestly) is a separate, manual step —
  this scorer only covers §13.1.
