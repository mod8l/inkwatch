"""Tests for the TTS queue (O1, O2) and cell/line vocabulary (§6.3).

`Speaker` is tested with an injected `speak_fn` so these run without a
real TTS engine or audio hardware — CI and this sandbox have neither.
"""

from __future__ import annotations

import threading
import time

from inkwatch.output import CELL_NAMES, Speaker, cell_name, describe_line
from inkwatch.rules import LINES


def test_cell_names_cover_all_nine_cells_with_center_named_center():
    assert len(CELL_NAMES) == 9
    assert cell_name(4) == "center"
    assert cell_name(0) == "top left"
    assert cell_name(8) == "bottom right"


def test_describe_line_covers_every_winning_line():
    for line in LINES:
        description = describe_line(line)
        assert isinstance(description, str) and description


def test_say_reaches_the_speak_function():
    heard = []
    speaker = Speaker(speak_fn=heard.append)

    speaker.say("hello")
    speaker.close()

    assert heard == ["hello"]


def test_a_newer_utterance_cancels_an_unsaid_older_one():
    release = threading.Event()
    heard = []

    def slow_speak(text: str) -> None:
        # Block the worker on the first utterance long enough that the
        # second and third are both queued before it's picked up.
        if text == "first":
            release.wait(timeout=2)
        heard.append(text)

    speaker = Speaker(speak_fn=slow_speak)
    speaker.say("first")
    time.sleep(0.05)  # let the worker start on "first" and start blocking
    speaker.say("second")
    speaker.say("third")  # cancels "second" (O2): only "third" should follow
    release.set()
    speaker.close()

    assert heard == ["first", "third"]


def test_say_does_not_block_the_caller():
    def slow_speak(text: str) -> None:
        time.sleep(1)

    speaker = Speaker(speak_fn=slow_speak)
    started = time.monotonic()
    speaker.say("anything")
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    speaker.close()


def test_disabled_speaker_never_calls_speak_fn():
    heard = []
    speaker = Speaker(enabled=False, speak_fn=heard.append)

    speaker.say("should not be heard")
    speaker.close()

    assert heard == []
