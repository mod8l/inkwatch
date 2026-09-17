"""Event types crossing module boundaries, and the JSONL writer that logs
them (L1).

`Observation` is what perception hands to session every frame: it never
includes raw pixels (ARCHITECTURE.md §2, "raw frames never cross to the
session") or a baseline of its own — perception measures against whatever
baseline session currently owns and hands back, it doesn't keep one
(CLAUDE.md hard rule: perception never mutates game state).

`EscalationOutcome` is the mirror-image boundary type for M5: what
`escalation.py`'s vision-model read hands back to `session.py`. Session
never sees the rectified crop that produced it (same "no raw pixels"
rule) or touches the network itself — `__main__.py`/`replay.py` own the
actual call and just hand the result to `Session.apply_escalation`.

Kept free of cv2/numpy so a future `rules.py`/`decision.py` import of a
type from here (CLAUDE.md's hard rule carves out that exception) can never
drag OpenCV in transitively. `SessionLogger` is plain-stdlib JSON/file I/O
for the same reason — it's still fine for `rules.py`/`decision.py` to
import types from this module without ever touching a filesystem.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

CellMark = Literal["none", "ambiguous", "marked"]


@dataclass(frozen=True)
class Observation:
    """One frame's read of the board (P1-P6, D1).

    `cell_marks` and `ratios` are `None` when the board wasn't found, or
    was found but classification wasn't requested (no baseline yet, e.g.
    during CALIBRATING before the first one is set).
    """

    frame_ts: float
    found: bool
    stable: bool
    occluded: bool
    missing_corners: tuple[str, ...]
    ratios: tuple[float, ...] | None
    cell_marks: tuple[CellMark, ...] | None


@dataclass(frozen=True)
class EscalationOutcome:
    """One vision-model read (D4, D5): which single cell (if any) it
    thinks has new ink, or why there isn't an answer to use.

    `cell=None` covers every non-answer uniformly — disabled (no API
    key), out of budget, timed out, or a network/parse error — `error`
    says which. `session.py` treats all of them the same way (§9: any of
    these "skip straight to asking the human"); the distinction is kept
    only for the log.
    """

    cell: int | None
    error: str | None
    latency_s: float
    cost: float


class SessionLogger:
    """Appends one JSON object per line to `<log_dir>/<session_id>.jsonl`
    (L1). Every write is flushed immediately, so a crash mid-game loses
    at most the in-flight event, not the whole log — the point of the
    log is to reconstruct what happened, including a crash.

    A `SessionLogger(enabled=False)` (or a missing/unwritable `log_dir`)
    is a silent no-op: logging is an evidence-gathering nicety, never a
    reason the game itself can't proceed (same spirit as D5/escalation
    disabling itself offline).
    """

    def __init__(self, log_dir: Path | str | None, session_id: str, enabled: bool = True) -> None:
        self.session_id = session_id
        self._file = None
        if not enabled or log_dir is None:
            return
        try:
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._file = (path / f"{session_id}.jsonl").open("a", encoding="utf-8")
        except OSError:
            self._file = None

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def log(self, event_type: str, **fields: Any) -> None:
        if self._file is None:
            return
        record = {"ts": time.time(), "session_id": self.session_id, "type": event_type}
        record.update({k: _jsonable(v) for k, v in fields.items()})
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of the dataclasses/tuples this module's
    types are built from into plain JSON-able values, so callers can log
    an `Observation`/`SessionResult`/`EscalationOutcome` straight through
    without hand-writing a `dict` at every call site."""
    if is_dataclass_instance(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (frozenset, set)):
        return sorted(_jsonable(v) for v in value)
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return value.value  # Enum, e.g. session.Phase
    return value


def is_dataclass_instance(value: Any) -> bool:
    return hasattr(type(value), "__dataclass_fields__")
