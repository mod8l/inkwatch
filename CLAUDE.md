# CLAUDE.md: instructions for building Inkwatch

You are helping Gad build Inkwatch, a take-home assignment for a VP R&D role. A human plays tic-tac-toe on paper against an agent that watches the page through a camera.

## Source of truth
Read these before writing any code, and re-read the relevant section before each milestone:
- `PRODUCT.md`: requirements (IDs like P1, D2, G4), state machine (§8), edge cases (§9), targets (§10), config (§12), layout (§14), milestones (§15).
- `ARCHITECTURE.md`: component boundaries. One file per component.
- `README.md`: the commands and flags that must actually work.
- `NOTES.md`: decisions log.

If the code needs to differ from these docs, stop and say so. Don't silently diverge. When a change is agreed, update the doc in the same commit.

## How to work
- **One milestone at a time** (M1 → M7 in `PRODUCT.md` §15). At the end of each: run the tests, summarize what was built, list what's untested or uncertain, and **stop for Gad's review**. Don't start the next milestone unprompted.
- M1–M4 are must-have. If time is short, cut M5–M7 before M4.
- Anything that needs the physical camera and paper can't be verified by you. Say exactly what Gad should try and what he should see.
- Prefer small, readable code over clever code. Gad must be able to explain any file.

## Hard rules
- **Never commit secrets.** API key only from `OPENROUTER_API_KEY` in `.env`. Create `.env.example` and `.gitignore` (`.env`, `sessions/`, recordings, `__pycache__`, `.venv`) in the first commit. Never print the key.
- `rules.py` and `decision.py`: no OpenCV, no I/O, no imports from other `inkwatch` modules except `events.py` types.
- Perception never mutates game state. It emits `Observation` events; only `session.py` commits moves.
- The agent never changes committed state on its own. Low confidence → escalate → ask the human.
- The game must be fully playable with no API key and no network.
- TTS must not block the frame loop.
- Only rectified board crops are ever sent to a model. No raw frames.

## Stack
Python 3.11+, `opencv-contrib-python`, `numpy`, `pyyaml`, `pyttsx3` (fall back to macOS `say` if it misbehaves), `python-dotenv`, `httpx`, `pytest`. Ask before adding anything else.

## Testing
- Write tests from the requirement IDs and spec, not from the implementation. Name them after the behavior (`test_two_new_marks_escalates`).
- `test_decision.py` must prove the agent never loses from any reachable position.
- `test_session.py` drives the state machine with synthetic observations covering every row of `PRODUCT.md` §9 that doesn't need a camera.

## NOTES.md
- Add build-time decisions to "Build-time decisions" in the existing format (chosen, considered, why, would change if).
- Add things tried and dropped to §3.
- **Do not fill in §5 "How I used AI tools"** or the time-spent line. Gad writes those himself.
- Never write a measured number anywhere unless it came from an actual run. Leave `TODO` otherwise.

## Commits
Small commits, one per coherent step, message prefixed with the milestone: `M2: per-cell ink ratio with inset`.
