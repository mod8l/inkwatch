"""Tests for the eval scorer (M6, PRODUCT.md §13.1), against synthetic
`events.jsonl`-shaped dicts and hand-built `GameLabel`s -- no camera, no
recording, no real vision-model call. These are the same six numbers
`eval/README.md` asks Gad to fill into README.md's "Measured results"
table from real recordings; this file only proves the arithmetic.
"""

from __future__ import annotations

import json

from inkwatch.metrics import (
    GameLabel,
    TrueMove,
    aggregate,
    format_table,
    load_events,
    load_label,
    main,
    score_game,
)

EMPTY = [None] * 9


def _board(**cells: str) -> list[str | None]:
    board = list(EMPTY)
    for index, symbol in cells.items():
        board[int(index)] = symbol
    return board


def _commit(frame_ts: float, board: list) -> dict:
    return {"type": "commit", "frame_ts": frame_ts, "board": board}


def _escalation(frame_ts: float, cost: float = 12.0) -> dict:
    return {"type": "escalation", "frame_ts": frame_ts, "cell": None, "error": None, "cost": cost}


def _question(frame_ts: float) -> dict:
    return {"type": "question", "frame_ts": frame_ts, "board": EMPTY}


def _result(frame_ts: float, board: list) -> dict:
    return {"type": "result", "frame_ts": frame_ts, "board": board}


def _label(session_id: str, true_moves: list[tuple[int, float]], **kwargs) -> GameLabel:
    final_board = kwargs.get("final_board")
    return GameLabel(
        session_id=session_id,
        true_moves=tuple(TrueMove(cell=c, hand_left_frame_ts=ts) for c, ts in true_moves),
        false_trigger_commits=kwargs.get("false_trigger_commits", 0),
        final_board=tuple(final_board) if final_board is not None else None,
        notes=kwargs.get("notes", ""),
    )


# -- score_game: one game's raw counts ---------------------------------------


def test_move_detection_counts_a_matching_committed_cell():
    events = [_commit(10.0, _board(**{"4": "X"}))]
    label = _label("g1", [(4, 9.0)])

    score = score_game(events, label)

    assert score.total_true_moves == 1
    assert score.matched_moves == 1


def test_move_detection_does_not_count_a_wrong_cell_commit():
    events = [_commit(10.0, _board(**{"7": "X"}))]  # committed cell 7, ground truth says 4
    label = _label("g1", [(4, 9.0)])

    score = score_game(events, label)

    assert score.matched_moves == 0


def test_committed_cells_diff_consecutive_boards_in_order():
    events = [
        _commit(10.0, _board(**{"4": "X"})),
        _commit(20.0, _board(**{"4": "X", "0": "O"})),
    ]
    label = _label("g1", [(4, 9.0), (0, 19.0)])

    score = score_game(events, label)

    assert score.matched_moves == 2
    assert score.detect_latencies_s == (1.0, 1.0)


def test_time_to_detect_is_commit_frame_ts_minus_hand_left_frame_ts():
    events = [_commit(11.5, _board(**{"4": "X"}))]
    label = _label("g1", [(4, 10.0)])

    score = score_game(events, label)

    assert score.detect_latencies_s == (1.5,)


def test_false_triggers_come_straight_from_the_label():
    score = score_game([], _label("g1", [], false_trigger_commits=2))

    assert score.false_triggers == 2


def test_turns_counts_commit_events_only():
    events = [
        _commit(1.0, _board(**{"4": "X"})),
        _question(2.0),
        _escalation(3.0),
        _commit(4.0, _board(**{"4": "X", "0": "O"})),
    ]
    label = _label("g1", [])

    score = score_game(events, label)

    assert score.turns == 2
    assert score.escalations == 1
    assert score.questions == 1


def test_desync_true_when_final_board_differs_from_label():
    events = [_result(5.0, _board(**{"4": "X"}))]
    label = _label("g1", [], final_board=_board(**{"4": "X", "0": "O"}))

    score = score_game(events, label)

    assert score.desynced is True


def test_desync_false_when_final_board_matches_label():
    board = _board(**{"4": "X", "0": "O"})
    events = [_result(5.0, board)]
    label = _label("g1", [], final_board=board)

    score = score_game(events, label)

    assert score.desynced is False


def test_desync_is_none_without_a_result_event():
    events = [_commit(1.0, _board(**{"4": "X"}))]
    label = _label("g1", [], final_board=_board(**{"4": "X"}))

    score = score_game(events, label)

    assert score.desynced is None


def test_escalation_tokens_are_read_from_the_log():
    events = [_escalation(1.0, cost=7.0), _escalation(2.0, cost=9.0)]
    label = _label("g1", [])

    score = score_game(events, label)

    assert score.escalation_tokens == (7.0, 9.0)


# -- aggregate: combining several games ---------------------------------------


def test_aggregate_move_detection_accuracy_across_games():
    s1 = score_game([_commit(1.0, _board(**{"4": "X"}))], _label("g1", [(4, 0.0)]))
    s2 = score_game([_commit(1.0, _board(**{"7": "X"}))], _label("g2", [(4, 0.0)]))  # missed

    summary = aggregate([s1, s2])

    assert summary["move_detection_accuracy"] == 0.5


def test_aggregate_false_triggers_per_10_games_normalizes():
    s1 = score_game([], _label("g1", [], false_trigger_commits=1))
    s2 = score_game([], _label("g2", [], false_trigger_commits=1))

    summary = aggregate([s1, s2])

    assert summary["false_triggers_per_10_games"] == 10.0  # 2 false triggers over 2 games -> 10 per 10 games


def test_aggregate_escalation_and_question_rate_over_total_turns():
    events = [
        _commit(1.0, _board(**{"4": "X"})),
        _escalation(1.5),
        _commit(2.0, _board(**{"4": "X", "0": "O"})),
    ]
    score = score_game(events, _label("g1", []))

    summary = aggregate([score])

    assert summary["escalation_rate"] == 0.5  # 1 escalation over 2 turns
    assert summary["human_question_rate"] == 0.0


def test_aggregate_time_to_detect_p50_over_all_matched_moves():
    events = [_commit(11.0, _board(**{"4": "X"}))]
    score = score_game(events, _label("g1", [(4, 10.0)]))

    summary = aggregate([score])

    assert summary["time_to_detect_p50_s"] == 1.0


def test_aggregate_desync_counts_only_scoreable_games():
    scoreable = score_game([_result(1.0, _board(**{"4": "X"}))], _label("g1", [], final_board=_board(**{"0": "O"})))
    unscoreable = score_game([], _label("g2", [], final_board=_board(**{"0": "O"})))

    summary = aggregate([scoreable, unscoreable])

    assert summary["desync_games"] == 1
    assert summary["desync_scoreable_games"] == 1


def test_aggregate_ratio_is_none_without_a_denominator():
    score = score_game([], _label("g1", []))

    summary = aggregate([score])

    assert summary["move_detection_accuracy"] is None
    assert summary["escalation_rate"] is None
    assert summary["time_to_detect_p50_s"] is None


# -- format_table: the README-shaped output ------------------------------------


def test_format_table_lists_all_six_spec_metrics():
    summary = aggregate([score_game([], _label("g1", []))])

    table = format_table(summary)

    for metric in (
        "Move detection accuracy",
        "False triggers per 10 games",
        "Escalation rate",
        "Human-question rate",
        "Time to detect (p50)",
        "End-of-game desync",
    ):
        assert metric in table


def test_format_table_renders_undefined_ratios_as_a_dash():
    summary = aggregate([score_game([], _label("g1", []))])

    table = format_table(summary)

    assert "—" in table  # no true moves, no turns -> nothing to divide by


# -- load_events / load_label: reading real files ------------------------------


def test_load_events_reads_one_dict_per_jsonl_line(tmp_path):
    session_dir = tmp_path / "20260101T000000"
    session_dir.mkdir()
    (session_dir / "events.jsonl").write_text(
        json.dumps(_commit(1.0, EMPTY)) + "\n" + json.dumps(_result(2.0, EMPTY)) + "\n"
    )

    events = load_events(session_dir)

    assert [e["type"] for e in events] == ["commit", "result"]


def test_load_label_reads_the_documented_yaml_schema(tmp_path):
    label_path = tmp_path / "20260101T000000.yaml"
    label_path.write_text(
        "session_id: \"20260101T000000\"\n"
        "true_moves:\n"
        "  - cell: 4\n"
        "    hand_left_frame_ts: 12.3\n"
        "false_trigger_commits: 1\n"
        "final_board: [X, null, null, null, null, null, null, null, null]\n"
        "notes: lingering hand\n"
    )

    label = load_label(label_path)

    assert label.session_id == "20260101T000000"
    assert label.true_moves == (TrueMove(cell=4, hand_left_frame_ts=12.3),)
    assert label.false_trigger_commits == 1
    assert label.final_board == ("X",) + (None,) * 8
    assert label.notes == "lingering hand"


def test_load_label_defaults_are_permissive(tmp_path):
    label_path = tmp_path / "minimal.yaml"
    label_path.write_text('session_id: "minimal"\n')

    label = load_label(label_path)

    assert label.true_moves == ()
    assert label.false_trigger_commits == 0
    assert label.final_board is None


# -- main: the CLI end to end ---------------------------------------------------


def test_main_prints_a_table_for_a_labeled_session(tmp_path, capsys):
    session_dir = tmp_path / "sessions" / "20260101T000000"
    session_dir.mkdir(parents=True)
    (session_dir / "events.jsonl").write_text(json.dumps(_commit(1.0, _board(**{"4": "X"}))) + "\n")

    labels_dir = tmp_path / "eval" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "20260101T000000.yaml").write_text(
        'session_id: "20260101T000000"\ntrue_moves:\n  - cell: 4\n    hand_left_frame_ts: 0.5\n'
    )

    main([str(session_dir), "--labels-dir", str(labels_dir)])

    out = capsys.readouterr().out
    assert "Scored 1 game(s)." in out
    assert "Move detection accuracy" in out


def test_main_exits_nonzero_when_no_session_has_a_label(tmp_path, capsys):
    session_dir = tmp_path / "sessions" / "20260101T000000"
    session_dir.mkdir(parents=True)
    (session_dir / "events.jsonl").write_text("")

    try:
        main([str(session_dir), "--labels-dir", str(tmp_path / "no-labels-here")])
        raised = False
    except SystemExit as exc:
        raised = exc.code != 0

    assert raised
