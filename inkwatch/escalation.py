"""Vision-model read on low confidence (D4, D5): given the rectified
board crop and the cells perception already flagged as candidates, asks
a vision model which single one (if any) actually has new ink, with a
hard timeout and a per-game call budget.

Only ever called by `__main__.py`/`replay.py`, never by `session.py` —
`session.py` never touches the network or a rectified crop; it just gets
handed the resulting `EscalationOutcome` (events.py) via
`Session.apply_escalation`. This keeps session.py's "no raw pixels"
boundary (ARCHITECTURE.md §2) intact even though the actual model call
now exists.

Never prints or logs the API key. Disables itself (no network call,
immediate "disabled" outcome) with no key present, `enabled=False`, or
once `max_calls_per_game` is spent — §9's "vision model timeout or no
API key: skip straight to asking the human" is exactly what an
`EscalationOutcome(cell=None, ...)` produces once `session.py` sees it,
so every one of these failure modes is handled by the same fallback path,
not a special case.
"""

from __future__ import annotations

import base64
import os
import time

import cv2
import httpx
import numpy as np

from inkwatch.events import EscalationOutcome
from inkwatch.output import cell_name
from inkwatch.rules import Board

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.5-flash"
DEFAULT_TIMEOUT_S = 3.0
DEFAULT_MAX_CALLS_PER_GAME = 5


def _board_summary(board: Board) -> str:
    taken = [f"{cell_name(i)} is {symbol}" for i, symbol in enumerate(board) if symbol is not None]
    return "; ".join(taken) if taken else "the board is empty"


def _build_prompt(board: Board, candidate_cells: frozenset[int]) -> str:
    candidates = ", ".join(sorted(cell_name(c) for c in candidate_cells))
    cell_key = ", ".join(f"{i}={cell_name(i)}" for i in range(9))
    return (
        "This is a rectified photo of a tic-tac-toe board drawn on paper. "
        f"Known state before this move: {_board_summary(board)}. "
        f"Cell numbering (row-major, 0-8): {cell_key}. "
        f"Exactly which one of these candidate cells has new, freshly-drawn ink "
        f"that isn't part of an existing mark: {candidates}. "
        "Reply with only the cell number (0-8), or the word none if none of them "
        "actually has a new mark. No other words."
    )


def _encode_crop(crop: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("could not encode the rectified crop")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _parse_cell_reply(text: str, candidate_cells: frozenset[int]) -> int | None:
    digits = "".join(ch for ch in text.strip() if ch.isdigit() or ch == " ")
    for token in digits.split():
        cell = int(token)
        if cell in candidate_cells:
            return cell
    return None


def _estimate_cost(response: dict) -> float:
    """OpenRouter echoes token usage on the response when available;
    without pricing per model wired in, this is a token count, not
    dollars — good enough for the JSONL log's "cost" field until M6's
    metrics pass wants a real number. 0.0, not a crash, if usage is
    missing (some models/providers don't report it)."""
    usage = response.get("usage") or {}
    return float(usage.get("total_tokens", 0.0))


class Escalator:
    """One instance per game: owns the per-game call budget (D4) and the
    API key (OPENROUTER_API_KEY only, per CLAUDE.md — never accepted as
    a constructor literal from a caller that might log it)."""

    def __init__(
        self,
        enabled: bool = True,
        model: str = DEFAULT_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_calls_per_game: int = DEFAULT_MAX_CALLS_PER_GAME,
        client: httpx.Client | None = None,
    ) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        self.enabled = enabled and bool(api_key)
        self.model = model
        self.timeout_s = timeout_s
        self.max_calls_per_game = max_calls_per_game
        self._api_key = api_key
        # `client` is injectable for tests (httpx.MockTransport); otherwise
        # one real Client is opened here and reused for the game's calls,
        # closed via `close()` -- not left to garbage collection.
        self._owns_client = client is None
        self._client = client if client is not None else (httpx.Client() if self.enabled else None)
        self.calls_made = 0

    def ask(self, crop: np.ndarray, board: Board, candidate_cells: frozenset[int]) -> EscalationOutcome:
        """D4: send the crop + known state, ask which candidate cell (if
        any) has new ink. Never raises — a network/parse failure is just
        another `EscalationOutcome(cell=None, ...)`, same as a timeout,
        so a flaky model call can't take down the frame loop."""
        if not self.enabled:
            return EscalationOutcome(cell=None, error="disabled", latency_s=0.0, cost=0.0)
        if self.calls_made >= self.max_calls_per_game:
            return EscalationOutcome(cell=None, error="budget_exhausted", latency_s=0.0, cost=0.0)
        if not candidate_cells:
            return EscalationOutcome(cell=None, error="no_candidates", latency_s=0.0, cost=0.0)

        self.calls_made += 1
        start = time.monotonic()
        try:
            response = self._call_model(crop, board, candidate_cells)
        except httpx.TimeoutException:
            return EscalationOutcome(cell=None, error="timeout", latency_s=time.monotonic() - start, cost=0.0)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure here must degrade to "ask", never crash the loop
            return EscalationOutcome(cell=None, error=str(exc), latency_s=time.monotonic() - start, cost=0.0)

        latency = time.monotonic() - start
        reply = _reply_text(response)
        cell = _parse_cell_reply(reply, candidate_cells) if reply else None
        return EscalationOutcome(cell=cell, error=None, latency_s=latency, cost=_estimate_cost(response))

    def _call_model(self, crop: np.ndarray, board: Board, candidate_cells: frozenset[int]) -> dict:
        image_b64 = _encode_crop(crop)
        payload = {
            "model": self.model,
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_prompt(board, candidate_cells)},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        assert self._client is not None  # only reached when self.enabled, which requires a client
        response = self._client.post(OPENROUTER_URL, json=payload, headers=headers, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()


def _reply_text(response: dict) -> str | None:
    """The model's reply as plain text, or None if there isn't any.
    OpenRouter-compatible APIs usually return a string `content`, but
    some models return a list of typed parts instead — returning it
    unguarded would make `_parse_cell_reply` call `.strip()` on a list
    and raise outside `ask()`'s try/except, crashing the frame loop
    mid-game. Anything that isn't text is "no answer", never an error."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)]
        return " ".join(parts) if parts else None
    return None
