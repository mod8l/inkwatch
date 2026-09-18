"""A full synthetic game, calibration through GAME_OVER, driven through
`run_frame` -- the exact per-frame step `__main__.py`'s live camera loop
and `replay.py`'s offline pipeline both use.

This is the strongest coverage possible without a real camera, and it is
still not an end-to-end test in the sense a reviewer might expect: every
mark is painted onto a synthetic frame by code (`make_raw_frame`), not
drawn by a real hand with a real pen under real lighting, and nothing
here calls a real vision model, plays real audio, or opens a real
`cv2.imshow` window. CLAUDE.md is explicit that verifying any of that
needs the physical camera and page, which this sandbox doesn't have --
see `README.md`'s Quick Start and `NOTES.md`'s Known limits for exactly
what's still unconfirmed. What this file adds beyond `test_replay.py`
(a couple of moves) and `test_session.py` (direct `Observation`/board
poking, including terminal positions set up by hand) is a *complete*
game played move by move through real perception, never assuming or
hardcoding what the agent replies with -- only that whatever it plays
gets detected and committed like any other mark.
"""

from __future__ import annotations

import pytest

from inkwatch.perception import DEFAULT_CELL_INSET, Perceiver, StabilityGate
from inkwatch.replay import run_frame
from inkwatch.rules import is_terminal
from inkwatch.session import Phase, Session
from test_replay import make_raw_frame

# A fixed priority order for the human's moves. The agent (`decision.py`)
# plays perfectly (`test_decision.py` proves it), so scripting moves that
# force a *human* win isn't possible against a real reply -- see
# test_session.py's own note on this. This test doesn't try: it only
# checks that a full game, whatever its actual outcome, is played out
# correctly end to end.
_HUMAN_PRIORITY = (4, 0, 2, 6, 8, 1, 3, 5, 7)


def _play_full_game(agent_first: bool) -> tuple[Session, object]:
    perceiver = Perceiver(stability=StabilityGate(stability_frames=3), cell_inset=DEFAULT_CELL_INSET)
    session = Session(agent_first=agent_first)
    marks: dict[int, float] = {}
    now = 0.0

    def settle():
        nonlocal now
        frame = make_raw_frame(marks)
        result = None
        result_with_message = None
        # Comfortably more frames than stability_frames*2 + 1 (P5/D6)
        # needs to both re-stabilize and get its second confirming read,
        # for any mark this test draws. `SessionResult.message` is only
        # set on the one tick something happened (M5.5's log-worthy-tick
        # rule applies here too), so it has to be grabbed as it goes by
        # rather than read off the final, since-quiet frame.
        for _ in range(10):
            now += 1.0
            _observation, result, _outcome = run_frame(perceiver, session, frame, now)
            if result.message:
                result_with_message = result
        return result_with_message if result_with_message is not None else result

    result = settle()  # calibrate on the blank board
    assert result.phase in (Phase.WAIT_HUMAN, Phase.WAIT_AGENT_INK)

    for plies in range(9):
        if result.phase == Phase.GAME_OVER:
            break
        if result.phase == Phase.WAIT_HUMAN:
            cell = next(c for c in _HUMAN_PRIORITY if c not in marks)
        elif result.phase == Phase.WAIT_AGENT_INK:
            cell = result.target_cell
        else:
            pytest.fail(f"unexpected phase {result.phase} after {plies} plies in a clean synthetic game")
        marks[cell] = 0.6
        result = settle()

    return session, result


def test_full_game_human_first_reaches_game_over_through_real_perception():
    session, result = _play_full_game(agent_first=False)

    assert result.phase == Phase.GAME_OVER
    assert session.phase == Phase.GAME_OVER
    assert is_terminal(result.board)
    assert result.message
    # G5's post-game re-read agreed with what was actually drawn.
    assert "doesn't quite match" not in result.message.lower()


def test_full_game_agent_first_reaches_game_over_through_real_perception():
    session, result = _play_full_game(agent_first=True)

    assert result.phase == Phase.GAME_OVER
    assert is_terminal(result.board)
    assert result.message
    assert "doesn't quite match" not in result.message.lower()
