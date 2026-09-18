"""Tests for the vision-model escalation call (D4, D5). No real network
call ever happens here: `httpx.MockTransport` stands in for OpenRouter,
so these are as fast and offline as every other test while still
exercising the real request-building/parsing code path.
"""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from inkwatch.escalation import Escalator
from inkwatch.rules import EMPTY_BOARD, apply_move

CROP = np.zeros((60, 60, 3), dtype=np.uint8)


def _client_replying(text: str, usage: dict | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": text}}],
                "usage": usage or {"total_tokens": 42},
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def _client_that_times_out() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _client_that_errors(status: int = 500) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "boom"})

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-do-not-print")


def test_disables_itself_with_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    escalator = Escalator(client=_client_replying("0"))

    outcome = escalator.ask(CROP, EMPTY_BOARD, frozenset({0}))

    assert outcome.cell is None
    assert outcome.error == "disabled"
    assert escalator.calls_made == 0


def test_disables_itself_when_explicitly_disabled():
    escalator = Escalator(enabled=False, client=_client_replying("0"))

    outcome = escalator.ask(CROP, EMPTY_BOARD, frozenset({0}))

    assert outcome.cell is None
    assert outcome.error == "disabled"


def test_accepts_a_candidate_cell_the_model_names():
    escalator = Escalator(client=_client_replying("4"))

    outcome = escalator.ask(CROP, EMPTY_BOARD, frozenset({4, 5}))

    assert outcome.cell == 4
    assert outcome.error is None
    assert outcome.latency_s >= 0
    assert outcome.cost == 42


def test_a_reply_of_none_means_no_cell():
    escalator = Escalator(client=_client_replying("none"))

    outcome = escalator.ask(CROP, EMPTY_BOARD, frozenset({0, 1}))

    assert outcome.cell is None
    assert outcome.error is None  # a real, successful answer -- just "no new mark"


def test_a_cell_outside_the_candidates_is_not_accepted():
    """D5: "consistent with the ink data" -- a cell perception never
    flagged isn't a legal answer, even if the model names one."""
    escalator = Escalator(client=_client_replying("7"))

    outcome = escalator.ask(CROP, EMPTY_BOARD, frozenset({0, 1}))

    assert outcome.cell is None
    assert outcome.error is None


def test_timeout_falls_back_cleanly():
    escalator = Escalator(client=_client_that_times_out())

    outcome = escalator.ask(CROP, EMPTY_BOARD, frozenset({0}))

    assert outcome.cell is None
    assert outcome.error == "timeout"


def test_a_server_error_falls_back_cleanly_instead_of_raising():
    escalator = Escalator(client=_client_that_errors(500))

    outcome = escalator.ask(CROP, EMPTY_BOARD, frozenset({0}))

    assert outcome.cell is None
    assert outcome.error is not None


def test_budget_is_exhausted_after_max_calls_per_game():
    escalator = Escalator(client=_client_replying("0"), max_calls_per_game=2)

    escalator.ask(CROP, EMPTY_BOARD, frozenset({0}))
    escalator.ask(CROP, EMPTY_BOARD, frozenset({0}))
    outcome = escalator.ask(CROP, EMPTY_BOARD, frozenset({0}))

    assert outcome.cell is None
    assert outcome.error == "budget_exhausted"
    assert escalator.calls_made == 2  # the third call never actually asks


def test_no_candidate_cells_is_a_no_op():
    escalator = Escalator(client=_client_replying("0"))

    outcome = escalator.ask(CROP, EMPTY_BOARD, frozenset())

    assert outcome.cell is None
    assert outcome.error == "no_candidates"
    assert escalator.calls_made == 0


def test_the_request_carries_the_key_and_board_state_but_never_a_raw_frame():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "0"}}]})

    board = apply_move(EMPTY_BOARD, 4, "X")
    escalator = Escalator(client=httpx.Client(transport=httpx.MockTransport(handler)))

    escalator.ask(CROP, board, frozenset({0}))

    assert captured["auth"] == "Bearer test-key-do-not-print"
    content = captured["body"]["messages"][0]["content"]
    text = next(part["text"] for part in content if part["type"] == "text")
    assert "center is X" in text
    image_url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
    assert image_url.startswith("data:image/png;base64,")
