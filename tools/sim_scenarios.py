"""The scenario runs: every PRODUCT.md §8/§9 behavior, played against the
real app by the Director (tools/simulate.py). Each run is one continuous
app session and one continuous piece of footage; runs are independent.

  happy_path       calibration, a full game to a result with a clean G5
                   final re-read, blank-page auto-restart (§6, §13.2)
  recovery         the §9 gauntlet: half-drawn mark, occupied cell, two
                   marks at once, wrong-cell O, lingering hand, page bump
                   (human turn / agent turn / page changed while away),
                   shadow, erased mark, armed-cell reminder, camera drop
  escalation_live  two marks with the real vision model enabled (M5)
  agent_first      --agent-first: the agent opens as X (A2)

Every check is a verdict: the scenario asserts the SPECIFIED app behavior
(spoken line + state visible in events.jsonl), so a FAIL is a genuine app
bug to fix, not a script hiccup to paper over.

Usage: python tools/sim_scenarios.py [run ...]   (default: all)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import CELL_INDEX, CELL_NAMES, Director  # noqa: E402

OUT_ROOT = Path("recordings/sim")


# ------------------------------------------------------------------ helpers


def play_human_move(d: Director, cell: int, ply: str, symbol: str = "X") -> int | None:
    """Draw the human's mark; expect 'You played <cell>. I'll take <target>.'
    Returns the agent's armed target cell (or None on failure)."""
    d.draw(cell, symbol)
    got = d.wait_msg("You played", timeout=30)
    if got:
        d.say_seen(got[1])
    if not d.check(f"{ply}: human move acknowledged", got is not None, "timeout after drawing"):
        return None
    target = d.await_agent_target(got[1])
    d.check(f"{ply}: agent reply armed", target is not None, f"could not parse target from '{got[1]}'")
    return target


def play_agent_ink(d: Director, target: int, ply: str, symbol: str = "O") -> bool:
    """Draw the agent's mark in its armed cell; expect 'Got it.' (or a
    terminal line, which is also a fine answer)."""
    d.draw(target, symbol)
    got = d.wait_msg(["Got it.", "I win", "It's a draw", "You win"], timeout=30)
    if got:
        d.say_seen(got[1])
    return d.check(f"{ply}: agent ink confirmed", got is not None, "no confirmation after drawing agent mark")


def app_board(d: Director) -> list:
    """The app's own latest board, peeked from its events.jsonl (all
    None if nothing has been committed yet)."""
    import json
    session_dir = d.app.session_dir
    if session_dir is not None and (session_dir / "events.jsonl").exists():
        for line in reversed((session_dir / "events.jsonl").read_text().splitlines()):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("board"):
                return event["board"]
    return [None] * 9


def empty_cell_other_than(d: Director, *used: int) -> int:
    """A cell currently empty on the app's board, avoiding `used`."""
    board = app_board(d)
    for cell in range(9):
        if cell in used:
            continue
        if board[cell] is None:
            return cell
    return (used[-1] + 1) % 9


# ------------------------------------------------------------------ run A


def run_happy(d: Director) -> None:
    d.set_scenario("A1 calibration", "A fresh page — calibration and the opening greeting")
    if not d.calibrate():
        return

    d.set_scenario("A2 full game", "A full game on the happy path — every move noticed, every reply drawn")
    human_plan = [4, 0, 2, 6, 8, 1, 3, 5, 7]
    result_line = None
    for ply in range(12):
        if result_line:
            break
        # human's turn (the agent's ink was confirmed by 'Got it.' -> WAIT_HUMAN)
        board = app_board(d)
        cell = next((c for c in human_plan if board[c] is None), None)
        if cell is None:
            break
        d.draw(cell, "X")
        pat, line = d.wait_msg(["You played", "I win", "It's a draw", "You win"], timeout=35) or (None, None)
        if pat is None:
            d.check(f"ply {ply}: app answers the human's move", False, "timeout — move not acknowledged")
            break
        d.say_seen(line)
        if pat != "You played":
            result_line = line
            break
        d.check(f"ply {ply}: human move acknowledged", True)
        target = d.await_agent_target(line)
        if target is None:
            d.check(f"ply {ply}: agent reply armed", False, line)
            break
        d.draw(target, "O")
        pat, line = d.wait_msg(["Got it.", "I win", "It's a draw", "You win"], timeout=35) or (None, None)
        if pat is None:
            d.check(f"ply {ply}: agent ink confirmed", False, "timeout after drawing agent mark")
            break
        d.say_seen(line)
        if pat != "Got it.":
            result_line = line

    d.check("the game reaches a result", result_line is not None, "never saw a terminal line")
    if result_line:
        d.check("final re-read agrees with the page (G5)", "doesn't quite match" not in result_line, result_line)

    d.set_scenario("A3 auto-restart", "Finished board swapped for a blank page — a new game starts itself")
    d.restart_game()


# ------------------------------------------------------------------ run B


def run_recovery(d: Director) -> None:
    d.set_scenario("B1 half-drawn mark", "Pen lifts mid-mark — no commit until the mark is finished (D6)")
    if not d.calibrate():
        return
    d.server.scene_call(d.scene.half_draw_then_finish, 4, "X", 2.2)
    time.sleep(1.3)  # 55% drawn, pencil has just left the page
    silent = d.expect_silent("You played", seconds=1.5)  # still mid-pause
    d.check("no commit while the mark is half-drawn", silent, "committed a half-drawn mark")
    d._wait_idle()
    got = d.wait_msg("You played center", timeout=30)
    if got:
        d.say_seen(got[1])
    d.check("finished mark commits", got is not None, "completed X never committed")
    target = d.await_agent_target(got[1]) if got else None
    if target is not None:
        play_agent_ink(d, target, "B1")

    d.set_scenario("B2 occupied cell", "Drawing over a taken cell — warned, old state kept, game goes on")
    if not d.restart_app():
        return
    target = play_human_move(d, 4, "B2 setup")
    if target is None or not play_agent_ink(d, target, "B2 setup"):
        return
    d.server.scene_call(d.scene.scribble, 4)
    d._wait_idle()
    got = d.wait_msg("already taken", timeout=20)
    if got:
        d.say_seen(got[1])
    d.check("occupied-cell scribble is warned", got is not None, "no 'already taken' warning")
    target = play_human_move(d, 2, "B2 recovery")
    d.check("game continues with ink left in the taken cell", target is not None,
            "DEADLOCK: further moves never evaluate while the extra ink stays")

    d.set_scenario("B3 two marks at once", "Two new marks in one glance — escalate, then ask which one")
    if not d.restart_app():
        return
    d.draw(4, "X")
    d.draw(0, "X")
    got = d.wait_msg("Which one is your move?", timeout=30)
    if got:
        d.say_seen(got[1])
    d.check("two marks trigger the which-one question", got is not None,
            "expected 'I see two new marks. Which one is your move?'")
    d.erase(0)
    got = d.wait_msg("You played center", timeout=30)
    if got:
        d.say_seen(got[1])
    d.check("removing one mark commits the other", got is not None, "lone remaining mark never committed")
    target = d.await_agent_target(got[1]) if got else None
    if target is not None:
        play_agent_ink(d, target, "B3")

    d.set_scenario("B4 wrong cell", "The agent's O drawn in the wrong cell — asked to fix the page")
    if not d.restart_app():
        return
    target = play_human_move(d, 4, "B4 setup")
    if target is None:
        return
    wrong = empty_cell_other_than(d, target)
    d.draw(wrong, "O")
    got = d.wait_msg("but I see a mark in", timeout=30)
    if got:
        d.say_seen(got[1])
    ok = got is not None and f"I asked for {CELL_NAMES[target]}" in got[1]
    d.check("wrong-cell ink is asked about, naming the armed cell", ok,
            got[1] if got else "no wrong-cell question")
    d.erase(wrong)
    d.check("wrong cell resolved by fixing the page", play_agent_ink(d, target, "B4"))

    d.set_scenario("B5 lingering hand", "Hand stays over the page — patient, then 'Take your time...' (15 s)")
    if not d.restart_app():
        return
    target = play_human_move(d, 4, "B5 setup")
    if target is None or not play_agent_ink(d, target, "B5 setup"):
        return
    d.server.scene_call(d.scene.linger, 18.0)
    got = d.wait_msg("Take your time", timeout=25)
    if got:
        d.say_seen(got[1])
    d.check("lingering hand gets the patient reminder", got is not None, "no reminder after ~15 s occluded")
    d._wait_idle()
    target2 = play_human_move(d, 0, "B5 resume")
    d.check("game resumes when the hand leaves", target2 is not None, "never evaluated a move after the linger")

    d.set_scenario("B6 page bump (human turn)", "Page shoved mid-game — BOARD_LOST, RESYNC, same game back")
    if not d.restart_app():
        return
    target = play_human_move(d, 4, "B6 setup")
    if target is None or not play_agent_ink(d, target, "B6 setup"):
        return
    d.bump()
    got = d.wait_msg("Okay, I can see the board again.", timeout=30)
    if got:
        d.say_seen(got[1])
    d.check("bumped page resyncs back into the game", got is not None, "no resync greeting after bump")
    target2 = play_human_move(d, 0, "B6 resume")
    d.check("moves evaluate again after the bump", target2 is not None, "no move committed after resync")

    d.set_scenario("B7 page bump (agent turn)", "Bumped while the agent's mark is armed — same cell re-armed")
    if not d.restart_app():
        return
    target = play_human_move(d, 4, "B7 setup")
    if target is None:
        return
    d.bump()
    got = d.wait_msg("Okay, I can see the board again.", timeout=30)
    if got:
        d.say_seen(got[1])
    ok = got is not None and f"I'll take {CELL_NAMES[target]}" in got[1]
    d.check("resync re-arms the SAME agent target", ok, got[1] if got else "no resync greeting")
    if got:
        d.check("agent ink still commits after the bump", play_agent_ink(d, target, "B7"))

    d.set_scenario("B8 page changed while lost", "A mark appears while the board is away — resync catches it")
    if not d.restart_app():
        return
    target = play_human_move(d, 4, "B8 setup")
    if target is None:
        return
    d.server.scene_call(d.scene.bump, 38, -22, 6)
    time.sleep(1.4)  # mid-bump: board undetectable
    d.instant_mark(0, "X")
    d._wait_idle()
    got = d.wait_msg("The page doesn't match what I have", timeout=30)
    if got:
        d.say_seen(got[1])
    d.check("sneaky mark is caught by the resync re-read", got is not None, "resync missed the changed page")
    d.erase(0)
    got = d.wait_msg("Okay, I can see the board again.", timeout=30)
    if got:
        d.say_seen(got[1])
    d.check("fixing the page resumes the game", got is not None, "no resume after erasing the sneaky mark")
    if got:
        d.check("armed agent ink still commits", play_agent_ink(d, target, "B8"))

    d.set_scenario("B9 shadow over a cell", "A shadow parks on a cell — ambiguous reads escalate, then ask")
    if not d.restart_app():
        return
    target = play_human_move(d, 4, "B9 setup")
    if target is None or not play_agent_ink(d, target, "B9 setup"):
        return
    d.shadow(8)
    got = d.wait_msg("check the light or the page", timeout=30)
    if got:
        d.say_seen(got[1])
    d.check("persistent shadow escalates to a light/page question", got is not None,
            "no light-or-page question for the shadowed cell")
    d.clear_shadow()
    d._pump(1.0)
    target2 = play_human_move(d, 0, "B9 resume")
    d.check("game resumes once the shadow lifts", target2 is not None, "no move committed after shadow cleared")

    d.set_scenario("B10 erased mark", "A committed mark is erased — the agent notices it vanished")
    if not d.restart_app():
        return
    target = play_human_move(d, 4, "B10 setup")
    if target is None or not play_agent_ink(d, target, "B10 setup"):
        return
    d.erase(4)
    got = d.wait_msg("seems to have disappeared", timeout=30)
    if got:
        d.say_seen(got[1])
    d.check("erased committed mark is reported", got is not None, "no disappeared-mark warning")
    d.draw(4, "X")
    d._pump(2.0)
    target2 = play_human_move(d, 0, "B10 resume")
    d.check("redrawing the mark lets the game continue", target2 is not None,
            "game did not continue after the mark was redrawn")

    d.set_scenario("B11 armed-cell reminder", "Agent's cell stays empty — it repeats the instruction (10 s)")
    if not d.restart_app():
        return
    target = play_human_move(d, 4, "B11 setup")
    if target is None:
        return
    got = d.wait_msg(f"I'll take {CELL_NAMES[target]}", timeout=25)
    if got:
        d.say_seen(got[1])
    d.check("armed move is repeated after 10 s of silence", got is not None, "no reminder within 25 s")
    d.check("game continues after the reminder", play_agent_ink(d, target, "B11"))

    d.set_scenario("B12 camera disconnect", "Camera unplugged — announced, retried, game resyncs on return")
    d.drop_camera(3.5)
    got = d.wait_msg("I've lost the camera", timeout=15)
    if got:
        d.say_seen(got[1])
    d.check("camera loss is announced", got is not None, "no lost-camera message")
    got = d.wait_msg("Okay, I can see the board again.", timeout=40)
    if got:
        d.say_seen(got[1])
    d.check("game resyncs when the camera returns", got is not None, "no resync after reconnect")


# ------------------------------------------------------------------ run C


def run_escalation_live(d: Director) -> None:
    d.set_scenario("C1 escalation", "Two marks at once WITH the vision model — the model is asked which")
    if not d.calibrate():
        return
    d.draw(4, "X")
    d.draw(0, "X")
    got = d.wait_msg(["You played", "Which one is your move?"], timeout=40)
    if got:
        d.say_seen(got[1])
    d.check("the two-mark situation resolves", got is not None, "neither a commit nor a question")
    escalation = d.wait_event("escalation", timeout=10)
    d.check("the model call is logged with latency", escalation is not None,
            "no escalation event in events.jsonl")
    if got and "Which one" in got[1]:
        d.caption("Model disagreed or timed out — the spec'd fallback asks the human")
        d.erase(0)
        got2 = d.wait_msg("You played center", timeout=30)
        if got2:
            d.say_seen(got2[1])
        d.check("human resolution commits the remaining mark", got2 is not None,
                "lone mark never committed after the fallback ask")
    elif got:
        d.caption("The vision model picked the fresh mark out of the two candidates")


# ------------------------------------------------------------------ run D


def run_agent_first(d: Director) -> None:
    d.set_scenario("D1 agent first", "--agent-first: the agent plays X and opens (center)")
    got = d.wait_msg("I can see the board", timeout=30)
    if got:
        d.say_seen(got[1])
    if not d.check("calibration greets the player", got is not None, "no intro"):
        return
    d.check("agent opens in the center", "I'll take center" in got[1], got[1])
    d.check("agent's X confirmed once drawn", play_agent_ink(d, 4, "D1", symbol="X"))
    d.set_scenario("D2 human as O", "The human replies as O and the game goes on")
    target = play_human_move(d, 0, "D2", symbol="O")
    d.check("game proceeds after the human's O", target is not None)


# ------------------------------------------------------------------ main


RUNS = {
    "happy_path": (run_happy, ["--no-escalation"]),
    "recovery": (run_recovery, ["--no-escalation"]),
    "escalation_live": (run_escalation_live, []),
    "agent_first": (run_agent_first, ["--agent-first", "--no-escalation"]),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", default=None, choices=list(RUNS), help="which runs (default: all)")
    parser.add_argument("--no-screen", action="store_true", help="skip the desktop/audio recording (debugging)")
    args = parser.parse_args()
    names = args.runs or list(RUNS)
    any_fail = False
    for name in names:
        fn, app_args = RUNS[name]
        with Director(name, OUT_ROOT, app_args, record_screen=not args.no_screen) as d:
            try:
                fn(d)
            except Exception as exc:  # a dead app or a script bug — both need reporting
                d.check("scenario run completed", False, f"{type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()
        any_fail = any_fail or any(not v.ok for v in d.verdicts)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
