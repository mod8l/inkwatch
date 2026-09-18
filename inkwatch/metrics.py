"""Eval scoring (M6): turns one recorded game's event log plus a small,
hand-written ground-truth label into the six PRODUCT.md §13.1 metrics,
and aggregates several games into the table README.md's "Measured
results" section asks for.

Deliberately scores `events.jsonl` -- what `__main__.py` already wrote
while the game was actually played -- rather than re-running perception
on the recording. Scoring the perception pipeline against its own
re-derived output would prove nothing; the point is checking it against
what really happened on the page.

The one thing this module can't do is watch the recording and produce
that ground truth itself: every `GameLabel` field is something Gad has to
read off the game (the saved frames, or the final page) and write down
by hand -- exactly the camera-and-paper verification CLAUDE.md says can't
be done here. See `eval/README.md` for the how-to and the label schema.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from inkwatch.rules import Board


@dataclass(frozen=True)
class TrueMove:
    """One real mark a human labeler saw on the page. `hand_left_frame_ts`
    is in the *same clock* as `events.jsonl`'s `frame_ts` (`time.monotonic()`
    from that game's own process) -- read directly off the saved frame
    filenames (`sessions/<id>/frames/`, or `raw/` with `--record`) rather
    than converted from a separately-timed video, so there's no
    clock-alignment step to get wrong."""

    cell: int
    hand_left_frame_ts: float


@dataclass(frozen=True)
class GameLabel:
    """Ground truth for one recorded session, written by hand. `notes`
    is free text for which §13.1 edge case this game exercises (two
    marks, wrong cell, page bump, lingering hand, shadow)."""

    session_id: str
    true_moves: tuple[TrueMove, ...]
    false_trigger_commits: int
    final_board: Board | None
    notes: str = ""


@dataclass(frozen=True)
class GameScore:
    """One game's raw counts, before aggregating across the eval set."""

    session_id: str
    total_true_moves: int
    matched_moves: int
    false_triggers: int
    escalations: int
    questions: int
    turns: int
    desynced: bool | None  # None: no "result" event, so unscoreable
    detect_latencies_s: tuple[float, ...]
    escalation_tokens: tuple[float, ...]


def load_events(session_dir: Path) -> list[dict]:
    """Reads a live game's `events.jsonl` (L1, written by `__main__.py`),
    not `replay.jsonl` -- the label describes what really happened on the
    page during that run, not a later replay of the same frames."""
    path = Path(session_dir) / "events.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_label(path: Path) -> GameLabel:
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    true_moves = tuple(
        TrueMove(cell=int(m["cell"]), hand_left_frame_ts=float(m["hand_left_frame_ts"]))
        for m in raw.get("true_moves", [])
    )
    final_board = raw.get("final_board")
    return GameLabel(
        session_id=raw["session_id"],
        true_moves=true_moves,
        false_trigger_commits=int(raw.get("false_trigger_commits", 0)),
        final_board=tuple(final_board) if final_board is not None else None,
        notes=raw.get("notes", ""),
    )


def _committed_cells(commits: Sequence[dict]) -> list[tuple[float, int]]:
    """One `(frame_ts, cell)` per commit event, `cell` being whichever
    board index went from empty to filled since the previous commit.
    Starts from an empty board, since the agent's own move is only ever
    committed once it's actually drawn (D6) -- there is no earlier state
    to diff against."""
    prev_board: Board = (None,) * 9
    out: list[tuple[float, int]] = []
    for commit in commits:
        board = tuple(commit["board"])
        newly_filled = [i for i, (before, after) in enumerate(zip(prev_board, board)) if before is None and after is not None]
        if newly_filled:
            out.append((commit["frame_ts"], newly_filled[0]))
        prev_board = board
    return out


def score_game(events: Sequence[dict], label: GameLabel) -> GameScore:
    """§13.1's per-game numbers. `turns` (the denominator for escalation
    rate and human-question rate) is one count per commit event -- §13.1
    doesn't define "turn" precisely, and a commit is the one thing every
    turn definitely produces if it resolves at all; see NOTES.md."""
    commits = [e for e in events if e["type"] == "commit"]
    escalations = [e for e in events if e["type"] == "escalation"]
    questions = [e for e in events if e["type"] == "question"]
    results = [e for e in events if e["type"] == "result"]

    committed = _committed_cells(commits)
    matched = 0
    latencies: list[float] = []
    for true_move, (frame_ts, cell) in zip(label.true_moves, committed):
        if cell == true_move.cell:
            matched += 1
            latencies.append(frame_ts - true_move.hand_left_frame_ts)

    desynced: bool | None = None
    if results:
        final_board = tuple(results[-1]["board"])
        desynced = label.final_board is not None and final_board != label.final_board

    return GameScore(
        session_id=label.session_id,
        total_true_moves=len(label.true_moves),
        matched_moves=matched,
        false_triggers=label.false_trigger_commits,
        escalations=len(escalations),
        questions=len(questions),
        turns=len(commits),
        desynced=desynced,
        detect_latencies_s=tuple(latencies),
        escalation_tokens=tuple(e.get("cost", 0.0) for e in escalations),
    )


def _percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def aggregate(scores: Sequence[GameScore]) -> dict:
    """Combines every scored game into the six §13.1 numbers (plus the
    non-functional cost figure README's table also carries). Any ratio
    with an empty denominator reports `None` rather than dividing by
    zero -- an eval set that never escalated, say, has an undefined
    escalation rate, not a 0% one."""
    total_true_moves = sum(s.total_true_moves for s in scores)
    total_matched = sum(s.matched_moves for s in scores)
    total_false_triggers = sum(s.false_triggers for s in scores)
    total_turns = sum(s.turns for s in scores)
    total_escalations = sum(s.escalations for s in scores)
    total_questions = sum(s.questions for s in scores)
    latencies = [lat for s in scores for lat in s.detect_latencies_s]
    tokens = [t for s in scores for t in s.escalation_tokens]
    scoreable_desync = [s for s in scores if s.desynced is not None]
    desynced_games = [s for s in scoreable_desync if s.desynced]

    return {
        "num_games": len(scores),
        "move_detection_accuracy": (total_matched / total_true_moves) if total_true_moves else None,
        "false_triggers_per_10_games": (total_false_triggers * 10 / len(scores)) if scores else None,
        "escalation_rate": (total_escalations / total_turns) if total_turns else None,
        "human_question_rate": (total_questions / total_turns) if total_turns else None,
        "time_to_detect_p50_s": _percentile(latencies, 50) if latencies else None,
        "desync_games": len(desynced_games),
        "desync_scoreable_games": len(scoreable_desync),
        "avg_escalation_tokens": (sum(tokens) / len(tokens)) if tokens else None,
    }


def format_table(summary: dict) -> str:
    def pct(value: float | None) -> str:
        return f"{value * 100:.1f}%" if value is not None else "—"

    rows = [
        ("Move detection accuracy", "≥ 98%", pct(summary["move_detection_accuracy"])),
        (
            "False triggers per 10 games",
            "≤ 1",
            f"{summary['false_triggers_per_10_games']:.1f}" if summary["false_triggers_per_10_games"] is not None else "—",
        ),
        ("Escalation rate", "< 5% of turns", pct(summary["escalation_rate"])),
        ("Human-question rate", "< 3% of turns", pct(summary["human_question_rate"])),
        (
            "Time to detect (p50)",
            "≤ 1 s",
            f"{summary['time_to_detect_p50_s']:.2f} s" if summary["time_to_detect_p50_s"] is not None else "—",
        ),
        ("End-of-game desync", "0", f"{summary['desync_games']}/{summary['desync_scoreable_games']}"),
        (
            "Cost per game",
            "≈ $0",
            f"{summary['avg_escalation_tokens']:.0f} tokens/game (avg)" if summary["avg_escalation_tokens"] is not None else "0 (no escalations)",
        ),
    ]
    lines = [f"Scored {summary['num_games']} game(s).", "", "| Metric | Target | Measured |", "|---|---|---|"]
    lines += [f"| {name} | {target} | {measured} |" for name, target, measured in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dirs", nargs="+", help="One or more sessions/<id> directories (each needs events.jsonl)")
    parser.add_argument("--labels-dir", default="eval/labels", help="Directory holding <session_id>.yaml ground-truth labels")
    args = parser.parse_args(argv)

    scores = []
    for raw_dir in args.session_dirs:
        session_dir = Path(raw_dir)
        session_id = session_dir.name
        label_path = Path(args.labels_dir) / f"{session_id}.yaml"
        if not label_path.exists():
            print(f"No label file at {label_path} -- skipping {session_id}", file=sys.stderr)
            continue
        events = load_events(session_dir)
        label = load_label(label_path)
        scores.append(score_game(events, label))

    if not scores:
        print("No scored games -- nothing to report.", file=sys.stderr)
        sys.exit(1)

    print(format_table(aggregate(scores)))


if __name__ == "__main__":
    main()
