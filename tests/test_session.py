"""Tests for the turn state machine (PRODUCT.md §8), driven with synthetic
`Observation`s so no camera is needed. Covers the M3 happy path (CALIBRATING
through a full game to GAME_OVER, the D6 debounce, D2/D3 confidence split)
and M4's recovery behavior for every §9 edge case that doesn't need a real
camera or vision model: two marks at once, a persistently ambiguous cell,
an occupied-cell redraw or erasure, ink in the wrong armed cell, and
BOARD_LOST/RESYNC. See session.py's module docstring for what's still
deliberately out of scope (keyboard y/n, a real vision-model call).
"""

from __future__ import annotations

from inkwatch.events import CellMark, Observation
from inkwatch.output import cell_name
from inkwatch.session import Phase, Session

BLANK_RATIOS = (0.0,) * 9
BLANK_MARKS: tuple[CellMark, ...] = ("none",) * 9


def marks(**marked_cells: CellMark) -> tuple[CellMark, ...]:
    """9 cell marks, "none" by default, overridden by cell index kwargs
    (e.g. marks(c0="marked", c5="ambiguous"))."""
    result = list(BLANK_MARKS)
    for key, value in marked_cells.items():
        result[int(key[1:])] = value
    return tuple(result)


def obs(
    *,
    found: bool = True,
    stable: bool = True,
    occluded: bool = False,
    ratios=BLANK_RATIOS,
    cell_marks=None,
    frame_ts: float = 0.0,
) -> Observation:
    return Observation(
        frame_ts=frame_ts,
        found=found,
        stable=stable,
        occluded=occluded,
        missing_corners=(),
        ratios=tuple(ratios) if ratios is not None else None,
        cell_marks=tuple(cell_marks) if cell_marks is not None else None,
    )


def calibrate(session: Session, now: float = 0.0) -> None:
    session.update(obs(ratios=BLANK_RATIOS, cell_marks=None, frame_ts=now), now)


# -- CALIBRATING -------------------------------------------------------


def test_calibrating_waits_until_board_is_found_and_stable():
    session = Session()

    result = session.update(obs(found=False, stable=False, ratios=None, cell_marks=None), now=0.0)

    assert session.phase == Phase.CALIBRATING
    assert result.message is None


def test_calibrating_sets_baseline_and_hands_off_to_the_human_by_default():
    session = Session()

    result = session.update(obs(ratios=BLANK_RATIOS, cell_marks=None), now=0.0)

    assert session.phase == Phase.WAIT_HUMAN
    assert session.baseline == [0.0] * 9
    assert "you go first" in result.message.lower()


def test_agent_first_calibration_immediately_arms_the_agents_opening_move():
    session = Session(agent_first=True)

    result = session.update(obs(ratios=BLANK_RATIOS, cell_marks=None), now=0.0)

    assert session.phase == Phase.WAIT_AGENT_INK
    assert session.target_cell == 4  # center, the deterministic opening move
    assert "i'll go first" in result.message.lower()
    assert "draw an x there" in result.message.lower()


# -- WAIT_HUMAN: D2/D3/D6 -------------------------------------------------


def _calibrated_session(agent_first: bool = False) -> Session:
    session = Session(agent_first=agent_first)
    calibrate(session)
    return session


def test_a_single_marked_cell_needs_two_consecutive_stable_reads_to_commit():
    session = _calibrated_session()
    candidate = marks(c0="marked")

    first = session.update(obs(ratios=BLANK_RATIOS, cell_marks=candidate, frame_ts=1.0), now=1.0)
    assert session.phase == Phase.WAIT_HUMAN
    assert session.board[0] is None
    assert first.message is None

    second = session.update(obs(ratios=BLANK_RATIOS, cell_marks=candidate, frame_ts=2.0), now=2.0)
    assert session.board[0] == "X"
    assert second.message is not None


def test_two_new_marks_at_once_does_not_commit():
    session = _calibrated_session()
    two_marks = marks(c0="marked", c1="marked")

    session.update(obs(cell_marks=two_marks, frame_ts=1.0), now=1.0)
    session.update(obs(cell_marks=two_marks, frame_ts=2.0), now=2.0)

    # M4: two consistent reads of two new marks escalates rather than
    # waiting forever — see test_two_new_marks_escalates_then_asks below.
    assert session.phase == Phase.ESCALATE
    assert all(c is None for c in session.board)


def test_an_ambiguous_cell_does_not_commit():
    session = _calibrated_session()
    ambiguous = marks(c3="ambiguous")

    session.update(obs(cell_marks=ambiguous, frame_ts=1.0), now=1.0)
    session.update(obs(cell_marks=ambiguous, frame_ts=2.0), now=2.0)

    assert session.phase == Phase.WAIT_HUMAN
    assert all(c is None for c in session.board)


def test_ink_change_in_an_already_occupied_cell_does_not_commit():
    session = _calibrated_session()
    session.board = ("X",) + session.board[1:]  # pretend cell 0 is already taken

    changed_occupied = marks(c0="marked")
    session.update(obs(cell_marks=changed_occupied, frame_ts=1.0), now=1.0)
    session.update(obs(cell_marks=changed_occupied, frame_ts=2.0), now=2.0)

    assert session.phase == Phase.WAIT_HUMAN


def test_switching_candidate_between_reads_restarts_the_debounce():
    session = _calibrated_session()
    cand_a, cand_b = marks(c0="marked"), marks(c1="marked")

    session.update(obs(cell_marks=cand_a, frame_ts=1.0), now=1.0)   # pending = 0
    session.update(obs(cell_marks=cand_b, frame_ts=2.0), now=2.0)   # pending = 1 (restarted)
    assert all(c is None for c in session.board)

    session.update(obs(cell_marks=cand_b, frame_ts=3.0), now=3.0)   # confirms 1

    assert session.board[1] == "X"
    assert session.board[0] is None


def test_an_unstable_frame_is_never_evaluated():
    session = _calibrated_session()
    candidate = marks(c0="marked")

    session.update(obs(stable=False, cell_marks=candidate, frame_ts=1.0), now=1.0)
    session.update(obs(stable=False, cell_marks=candidate, frame_ts=2.0), now=2.0)

    assert session.phase == Phase.WAIT_HUMAN
    assert all(c is None for c in session.board)


# -- Commit -> THINK -> WAIT_AGENT_INK -----------------------------------


def _commit_human_move(session: Session, cell: int, now_start: float = 1.0, ratios=BLANK_RATIOS):
    candidate = marks(**{f"c{cell}": "marked"})
    session.update(obs(cell_marks=candidate, ratios=ratios, frame_ts=now_start), now=now_start)
    return session.update(obs(cell_marks=candidate, ratios=ratios, frame_ts=now_start + 1), now=now_start + 1)


def board_ratios(board) -> tuple[float, ...]:
    """A plausible ink ratio per cell for a given board: enough ink to
    read as "marked" against the zero blank baseline where occupied,
    none where empty. Used to give G5's terminal re-read something
    realistic to check against a hand-set test board."""
    return tuple(0.3 if cell is not None else 0.0 for cell in board)


def test_committing_the_humans_move_immediately_arms_the_agents_reply():
    session = _calibrated_session()

    result = _commit_human_move(session, cell=0)

    assert session.board[0] == "X"
    assert session.phase == Phase.WAIT_AGENT_INK
    assert session.target_cell is not None
    assert session.board[session.target_cell] is None
    assert "you played top left" in result.message.lower()
    assert "please draw an o there" in result.message.lower()


def test_baseline_moves_forward_after_a_commit():
    session = _calibrated_session()
    new_ratios = tuple(0.5 if i == 0 else 0.0 for i in range(9))
    candidate = marks(c0="marked")

    session.update(obs(ratios=new_ratios, cell_marks=candidate, frame_ts=1.0), now=1.0)
    session.update(obs(ratios=new_ratios, cell_marks=candidate, frame_ts=2.0), now=2.0)

    assert session.baseline == list(new_ratios)


def test_agent_ink_in_the_armed_cell_commits_and_returns_turn_to_human():
    session = _calibrated_session()
    _commit_human_move(session, cell=0)
    target = session.target_cell
    agent_marks = marks(**{f"c{target}": "marked"})

    session.update(obs(cell_marks=agent_marks, frame_ts=10.0), now=10.0)
    result = session.update(obs(cell_marks=agent_marks, frame_ts=11.0), now=11.0)

    assert session.board[target] == "O"
    assert session.phase == Phase.WAIT_HUMAN
    assert session.turn == "X"
    assert result.message == "Got it."


def test_ink_outside_the_armed_cell_does_not_commit():
    session = _calibrated_session()
    _commit_human_move(session, cell=0)
    target = session.target_cell
    wrong_cell = next(i for i in range(9) if i != target and session.board[i] is None)
    wrong_marks = marks(**{f"c{wrong_cell}": "marked"})

    session.update(obs(cell_marks=wrong_marks, frame_ts=10.0), now=10.0)
    session.update(obs(cell_marks=wrong_marks, frame_ts=11.0), now=11.0)

    # M4: two consistent reads of ink in the wrong cell asks the human
    # rather than waiting forever — see test_wrong_cell_ink_asks_* below.
    assert session.phase == Phase.ASK_HUMAN
    assert session.board[target] is None


# -- O3 reminders ----------------------------------------------------------


def test_agent_move_is_repeated_after_ten_seconds_of_silence_then_every_twenty():
    session = _calibrated_session()
    _commit_human_move(session, cell=0, now_start=0.0)

    quiet = obs(cell_marks=marks(), frame_ts=0.0)  # nothing new happening on the page
    assert session.update(quiet, now=5.0).message is None
    reminder_1 = session.update(quiet, now=11.0)
    assert reminder_1.message is not None
    assert session.update(quiet, now=25.0).message is None
    reminder_2 = session.update(quiet, now=32.0)
    assert reminder_2.message == reminder_1.message


# -- Terminal states -------------------------------------------------------
#
# These set `session.board`/`baseline` directly to a one-move-from-terminal
# position rather than playing a full game through the agent's real
# replies: the agent plays perfectly (test_decision.py proves it never
# loses), so a human can only ever draw or lose against it, never win —
# a hand-played-out "human wins" sequence with real minimax replies isn't
# a reachable game at all. G5 (terminal detection + message) only cares
# about the commit that crosses into a terminal board, so setting that
# board up directly is the more honest test. `ratios=board_ratios(...)`
# gives G5's own full-board re-read (M4) something consistent to confirm
# against, the same way a real final frame would.


def test_a_terminal_human_move_ends_the_game_with_the_right_message():
    session = _calibrated_session()
    session.board = ("X", "X", None, "O", "O", None, None, None, None)
    session.baseline = [0.0] * 9
    final_board = ("X", "X", "X", "O", "O", None, None, None, None)

    result = _commit_human_move(session, cell=2, now_start=1.0, ratios=board_ratios(final_board))

    assert session.phase == Phase.GAME_OVER
    assert "you win" in result.message.lower()
    assert "top row" in result.message.lower()
    assert "doesn't quite match" not in result.message.lower()


def test_a_terminal_agent_move_ends_the_game_with_the_right_message():
    session = _calibrated_session()
    session.board = ("O", "O", None, "X", "X", None, None, None, None)
    session.baseline = [0.0] * 9
    session.phase = Phase.WAIT_AGENT_INK
    session.target_cell = 2
    session.turn = "O"
    final_ratios = board_ratios(("O", "O", "O", "X", "X", None, None, None, None))

    agent_marks = marks(c2="marked")
    session.update(obs(cell_marks=agent_marks, ratios=final_ratios, frame_ts=1.0), now=1.0)
    result = session.update(obs(cell_marks=agent_marks, ratios=final_ratios, frame_ts=2.0), now=2.0)

    assert session.phase == Phase.GAME_OVER
    assert "i win" in result.message.lower()
    assert "top row" in result.message.lower()
    assert "doesn't quite match" not in result.message.lower()


def test_a_full_board_with_no_winner_ends_in_a_draw():
    session = _calibrated_session()
    session.board = ("X", "O", "X", "X", "O", "O", "O", "X", None)
    session.baseline = [0.0] * 9
    final_board = ("X", "O", "X", "X", "O", "O", "O", "X", "X")

    result = _commit_human_move(session, cell=8, now_start=1.0, ratios=board_ratios(final_board))

    assert session.phase == Phase.GAME_OVER
    assert result.message == "It's a draw."


def test_terminal_message_flags_a_page_state_mismatch():
    """G5: the post-game re-read is a real check, not a rubber stamp —
    a cell the page doesn't actually back up gets called out by name."""
    session = _calibrated_session()
    session.board = ("X", "X", None, "O", "O", None, None, None, None)
    session.baseline = [0.0] * 9
    final_board = ("X", "X", "X", "O", "O", None, None, None, None)
    bad_ratios = tuple(0.0 if i == 4 else r for i, r in enumerate(board_ratios(final_board)))

    result = _commit_human_move(session, cell=2, now_start=1.0, ratios=bad_ratios)

    assert session.phase == Phase.GAME_OVER
    assert "you win" in result.message.lower()
    assert "doesn't quite match" in result.message.lower()
    assert "center" in result.message.lower()


# -- M4: two marks at once (§9) -----------------------------------------


def test_two_new_marks_escalates_then_asks_which_one():
    session = _calibrated_session()
    two_marks = marks(c0="marked", c1="marked")

    session.update(obs(cell_marks=two_marks, frame_ts=1.0), now=1.0)
    escalated = session.update(obs(cell_marks=two_marks, frame_ts=2.0), now=2.0)
    assert escalated.phase == Phase.ESCALATE
    assert escalated.confidence == "escalating"
    assert escalated.message is None

    asked = session.update(obs(cell_marks=two_marks, frame_ts=3.0), now=3.0)
    assert asked.phase == Phase.ASK_HUMAN
    assert asked.confidence == "asking"
    assert "which one" in asked.message.lower()
    assert all(c is None for c in session.board)


def test_two_new_marks_resolved_by_narrowing_to_one_commits_it():
    session = _calibrated_session()
    two_marks = marks(c0="marked", c1="marked")
    for ts in (1.0, 2.0, 3.0):
        session.update(obs(cell_marks=two_marks, frame_ts=ts), now=ts)
    assert session.phase == Phase.ASK_HUMAN

    one_mark = marks(c0="marked")
    session.update(obs(cell_marks=one_mark, frame_ts=4.0), now=4.0)
    session.update(obs(cell_marks=one_mark, frame_ts=5.0), now=5.0)

    assert session.board[0] == "X"
    assert session.phase == Phase.WAIT_AGENT_INK


def test_two_new_marks_withdrawn_returns_to_wait_human():
    session = _calibrated_session()
    two_marks = marks(c0="marked", c1="marked")
    for ts in (1.0, 2.0, 3.0):
        session.update(obs(cell_marks=two_marks, frame_ts=ts), now=ts)
    assert session.phase == Phase.ASK_HUMAN

    session.update(obs(cell_marks=marks(), frame_ts=4.0), now=4.0)

    assert session.phase == Phase.WAIT_HUMAN
    assert all(c is None for c in session.board)


# -- M4: persistent ambiguous ink / shadow (§9) --------------------------


def test_briefly_ambiguous_cell_does_not_escalate_within_two_reads():
    session = _calibrated_session()
    faint = marks(c4="ambiguous")

    session.update(obs(cell_marks=faint, frame_ts=1.0), now=1.0)
    result = session.update(obs(cell_marks=faint, frame_ts=2.0), now=2.0)

    assert result.phase == Phase.WAIT_HUMAN


def test_persistent_ambiguous_cell_escalates_then_asks_about_light_or_page():
    session = _calibrated_session()
    faint = marks(c4="ambiguous")

    result = None
    for ts in (1.0, 2.0, 3.0):
        result = session.update(obs(cell_marks=faint, frame_ts=ts), now=ts)
    assert result.phase == Phase.ESCALATE

    asked = session.update(obs(cell_marks=faint, frame_ts=4.0), now=4.0)
    assert asked.phase == Phase.ASK_HUMAN
    assert "light" in asked.message.lower() and "page" in asked.message.lower()


# -- M4: occupied-cell redraw and erasure (§9) ---------------------------


def test_ink_in_an_occupied_cell_warns_and_keeps_the_old_state():
    session = _calibrated_session()
    session.board = ("X",) + session.board[1:]
    session.baseline = [0.0] * 9
    changed = marks(c0="marked")

    first = session.update(obs(cell_marks=changed, frame_ts=1.0), now=1.0)
    assert first.message is None

    second = session.update(obs(cell_marks=changed, frame_ts=2.0), now=2.0)
    assert second.message is not None
    assert "already taken" in second.message.lower()
    assert session.board[0] == "X"

    # doesn't repeat every frame while the same ink persists
    third = session.update(obs(cell_marks=changed, frame_ts=3.0), now=3.0)
    assert third.message is None


def test_an_erased_mark_warns_that_it_disappeared():
    session = _calibrated_session()
    session.board = ("X",) + session.board[1:]
    session.baseline = [0.3] + [0.0] * 8  # cell 0 had real ink when committed
    faded_ratios = (0.0,) * 9  # the ink is gone now
    clean_marks = marks()  # classify_cell reads a decrease as "none", same as blank

    first = session.update(obs(ratios=faded_ratios, cell_marks=clean_marks, frame_ts=1.0), now=1.0)
    assert first.message is None

    second = session.update(obs(ratios=faded_ratios, cell_marks=clean_marks, frame_ts=2.0), now=2.0)
    assert second.message is not None
    assert "disappeared" in second.message.lower()
    assert "top left" in second.message.lower()
    assert session.board[0] == "X"  # never guesses a removal into a committed change


# -- M4: ink in the wrong armed cell (§9) --------------------------------


def test_wrong_cell_ink_asks_then_resolves_when_target_cell_gets_ink():
    session = _calibrated_session()
    _commit_human_move(session, cell=0)
    target = session.target_cell
    wrong_cell = next(i for i in range(9) if i != target and session.board[i] is None)
    wrong_marks = marks(**{f"c{wrong_cell}": "marked"})

    session.update(obs(cell_marks=wrong_marks, frame_ts=10.0), now=10.0)
    asked = session.update(obs(cell_marks=wrong_marks, frame_ts=11.0), now=11.0)

    assert asked.phase == Phase.ASK_HUMAN
    assert cell_name(target) in asked.message.lower()
    assert cell_name(wrong_cell) in asked.message.lower()

    right_marks = marks(**{f"c{target}": "marked"})
    session.update(obs(cell_marks=right_marks, frame_ts=12.0), now=12.0)
    session.update(obs(cell_marks=right_marks, frame_ts=13.0), now=13.0)

    assert session.board[target] == "O"
    assert session.phase == Phase.WAIT_HUMAN


def test_wrong_cell_ink_withdrawn_stays_armed_when_the_stray_mark_clears():
    session = _calibrated_session()
    _commit_human_move(session, cell=0)
    target = session.target_cell
    wrong_cell = next(i for i in range(9) if i != target and session.board[i] is None)
    wrong_marks = marks(**{f"c{wrong_cell}": "marked"})

    session.update(obs(cell_marks=wrong_marks, frame_ts=10.0), now=10.0)
    session.update(obs(cell_marks=wrong_marks, frame_ts=11.0), now=11.0)
    assert session.phase == Phase.ASK_HUMAN

    session.update(obs(cell_marks=marks(), frame_ts=12.0), now=12.0)

    assert session.phase == Phase.WAIT_AGENT_INK
    assert session.target_cell == target


# -- M4: BOARD_LOST / RESYNC (§9 "page bumped or rotated") ---------------


def test_board_not_found_enters_board_lost():
    session = _calibrated_session()

    result = session.update(obs(found=False, stable=False, ratios=None, cell_marks=None, frame_ts=1.0), now=1.0)

    assert result.phase == Phase.BOARD_LOST


def test_resync_resumes_wait_human_when_the_page_matches_state():
    session = _calibrated_session()
    session.update(obs(found=False, stable=False, ratios=None, cell_marks=None, frame_ts=1.0), now=1.0)
    session.update(obs(found=True, stable=False, ratios=None, cell_marks=None, frame_ts=2.0), now=2.0)
    assert session.phase == Phase.RESYNC

    matching = obs(found=True, stable=True, ratios=BLANK_RATIOS, cell_marks=None, frame_ts=3.0)
    result = session.update(matching, now=3.0)

    assert result.phase == Phase.WAIT_HUMAN


def test_resync_asks_when_the_page_differs_from_state():
    session = _calibrated_session()
    session.board = ("X",) + session.board[1:]  # session believes top left is taken

    session.update(obs(found=False, stable=False, ratios=None, cell_marks=None, frame_ts=1.0), now=1.0)
    session.update(obs(found=True, stable=False, ratios=None, cell_marks=None, frame_ts=2.0), now=2.0)
    assert session.phase == Phase.RESYNC

    # the physical page shows nothing at cell 0 -- disagrees with session.board
    mismatched = obs(found=True, stable=True, ratios=BLANK_RATIOS, cell_marks=None, frame_ts=3.0)
    result = session.update(mismatched, now=3.0)

    assert result.phase == Phase.ASK_HUMAN
    assert "top left" in result.message.lower()


def test_board_lost_mid_agent_turn_resumes_wait_agent_ink_with_the_same_target():
    session = _calibrated_session()
    _commit_human_move(session, cell=0)
    target = session.target_cell
    assert session.phase == Phase.WAIT_AGENT_INK

    session.update(obs(found=False, stable=False, ratios=None, cell_marks=None, frame_ts=20.0), now=20.0)
    assert session.phase == Phase.BOARD_LOST
    session.update(obs(found=True, stable=False, ratios=None, cell_marks=None, frame_ts=21.0), now=21.0)
    assert session.phase == Phase.RESYNC

    matching_ratios = tuple(0.3 if session.board[i] is not None else 0.0 for i in range(9))
    result = session.update(
        obs(found=True, stable=True, ratios=matching_ratios, cell_marks=None, frame_ts=22.0), now=22.0
    )

    assert result.phase == Phase.WAIT_AGENT_INK
    assert session.target_cell == target


# -- M4: hand lingering over the page (§9) -------------------------------


def test_hand_lingering_during_wait_human_reminds_after_the_occlusion_timeout():
    session = _calibrated_session()
    unstable = obs(stable=False, ratios=None, cell_marks=None, frame_ts=0.0)

    assert session.update(unstable, now=1.0).message is None
    reminder = session.update(unstable, now=16.0)  # default occlusion_reminder_s is 15

    assert reminder.message is not None
    assert "take your time" in reminder.message.lower()
