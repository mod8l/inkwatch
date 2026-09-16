# COMPLIANCE.md: SOC 2 / ISO 27001 readiness

## 0. Framing, honestly

SOC 2 (AICPA Trust Services Criteria) and ISO/IEC 27001 are audited
properties of an **organization and a system it operates for customers**
— a report or certificate issued after a third party checks that
documented controls exist *and* are actually followed over time (SOC 2
Type II: over a period; ISO 27001: via a certification body plus annual
surveillance audits). A solo take-home CLI tool that runs on one laptop,
with no customers, no company, and no ongoing operations, is not the kind
of thing either framework certifies, and nothing in this repo can make it
"SOC 2 compliant" by itself.

What *is* useful now, and what this doc is: a gap analysis of Inkwatch's
current design against both frameworks' control domains, assuming it
became the real, hosted, multi-tenant product `ARCHITECTURE.md` sketches
in its dashed ("designed, not built") boxes — session service per
connection, shared model gateway, eval harness, the works. The question
this doc answers is "if this had to pass an audit next year, what design
decisions made now would help or hurt," not "is this compliant today."

## 1. Already aligned

A few things the project already does for other reasons turn out to be
exactly what both frameworks ask for. Worth naming so they don't get
undone by accident later:

| Existing decision | Where | Which control it happens to satisfy |
|---|---|---|
| Only a rectified board crop ever leaves the device, never a raw frame (`CLAUDE.md` hard rule) | `perception.py` / `escalation.py` | Data minimization — SOC 2 Confidentiality criteria; ISO A.8.10, A.5.34 |
| API key only from `.env`, never committed, never printed (`CLAUDE.md` hard rule) | `.env.example`, `.gitignore` | Secrets management — SOC 2 CC6.1; ISO A.8.24, A.5.17 |
| Game fully playable offline, escalation opt-in and self-disabling with no key | `PRODUCT.md` §10, `escalation.py` | Least-privilege-by-default / no forced data egress — SOC 2 Confidentiality; ISO A.8.10 |
| Recording (`--record`) is opt-in, off by default | `README.md`, `config.yaml` | Data minimization / purpose limitation — SOC 2 Privacy; ISO A.5.34 |
| Mandatory PR review + checklist + CI test gate (`AGENTS.md`, `CODE_REVIEW.md`, `.github/workflows/tests.yml`) | repo root | Change management — SOC 2 CC8; ISO A.8.32 |
| `rules.py`/`decision.py` I/O and cross-import ban, enforced by review checklist | `CLAUDE.md`, `CODE_REVIEW.md` §2 | Secure design / least functionality — ISO A.8.27 |
| Event log of every observation, escalation, commit, question (L1) | `PRODUCT.md` §7.5 | System monitoring / audit trail — SOC 2 CC7.2; ISO A.8.15 (see gap below: exists, but not yet audit-grade) |

## 2. Gap analysis by control domain

"SOC 2" below means the Common Criteria (CC1–CC9) unless another Trust
Services Criterion (Availability, Confidentiality, Privacy) is named.
"ISO" means ISO/IEC 27001:2022 Annex A control numbers.

| Domain | SOC 2 | ISO 27001 | Current state | Gap |
|---|---|---|---|---|
| Governance, risk assessment, asset inventory | CC1, CC3 | A.5.1–5.9, A.6.1–6.3 | None — no formal ISMS, no risk register, no documented asset inventory | Org-level artifact, not a code change. First step if this becomes real: a risk register and an "asset" list (camera images, session logs, API key, model provider) with an owner and a classification each. |
| Access control & authN/authZ | CC6.1–6.3 | A.5.15–5.18, A.8.2–8.5 | None — single local user, no login, no per-tenant boundary | `ARCHITECTURE.md`'s future "Session service · per connection" needs an actual auth story before it's multi-tenant: who can start/see/replay whose session. Doesn't exist yet even as a design. |
| Change management | CC8 | A.8.32 | PR + review checklist + CI test gate exist (`AGENTS.md`, `CODE_REVIEW.md`) | Branch protection isn't turned on in GitHub settings (noted in `AGENTS.md` "Not yet set up"); single-maintainer today so "someone else reviews" isn't yet enforceable, only aspirational. |
| Logging, monitoring, audit trail integrity | CC7.2–7.4 | A.8.15, A.8.16 | JSONL event log (L1) exists, written locally to `sessions/` | Not tamper-evident (any process with disk access can edit past entries), no centralized/immutable storage, no retention policy, no alerting on anomalies (e.g. escalation-budget exhaustion, repeated `BOARD_LOST`). Fine for a local debug log; not audit-grade. |
| Encryption at rest / in transit | CC6.1, CC6.7; Confidentiality TSC | A.8.24 | `sessions/` (logs + saved frames, L2) stored as plain files, unencrypted. `httpx` calls to OpenRouter use `https://` (TLS in transit) but this isn't asserted anywhere in code or config — a typo'd `http://` would fail open silently | Add an explicit scheme check/assertion in `escalation.py` when it's built (M5), not just "the URL happens to start with https". At-rest encryption for `sessions/` is a real gap once frames of a real room are stored; not needed for a take-home on a personal machine, but worth flagging before any hosted version. |
| Secrets / key management | CC6.1 | A.8.24, A.5.17 | `.env` file, `.gitignore`'d, read via `python-dotenv`; fine for one developer on one machine | A hosted multi-tenant version needs a managed secret store (e.g. cloud KMS / secrets manager) with rotation, not a `.env` file per deploy. Design note for `ARCHITECTURE.md`'s future "Model gateway" box, not a current-repo change. |
| Vendor / subprocessor management | CC9.2 | A.5.19–5.23 | OpenRouter (and whichever model it routes to) receives board-crop images on escalation; no vendor risk review, no data processing agreement, no subprocessor disclosure exists anywhere | Needs: a documented subprocessor list, confirmation of OpenRouter's/the model vendor's data retention and training-use policy for submitted images, and a DPA if this ever handles a real customer's data. None of this can be satisfied by code — it's a paperwork/vendor-contract gap. |
| Data retention & deletion | Confidentiality/Privacy TSC | A.5.34, A.8.10 | `sessions/` and recordings accumulate indefinitely with no expiry; nothing deletes old sessions | No auto-expiry or deletion tooling exists. A real product needs a stated retention period and a way to actually delete a user's data on request. |
| Privacy notice & consent | Privacy TSC | (pairs with ISO/IEC 27701) | The camera points at a human and their handwriting; `README.md` documents what's sent where, but there's no explicit, shown-to-the-user consent step, and `--record` saves the *raw* stream (i.e., potentially more than the board — hands, background, face if the camera catches it) | Before any real user besides the developer runs this, the "board crop only, recording is opt-in" promise needs to actually be visible in the running app (a startup notice), not just documented in `README.md`. Worth doing regardless of certification, once there's any user other than Gad. |
| Incident response | CC7.3–7.5 | A.5.24–5.28 | None — no IR plan, no contact, no defined severity levels | Org-level artifact. A single line in `README.md`/`COMPLIANCE.md` ("report a security issue to: ...") is a reasonable first step whenever this is shared beyond one machine; a full IR plan is out of scope until there's an operator to run one. |
| Business continuity & backup | A1 (Availability TSC), CC9.1 | A.5.29–5.30, A.8.13 | N/A today — single local process, no uptime commitment, nothing to back up beyond the developer's own `sessions/` | Becomes relevant only once hosted; not a gap for a local CLI tool. |
| Physical & environmental security | — | A.7.1–7.14 | N/A — runs on the user's own laptop | Becomes the cloud provider's shared responsibility once hosted; document the shared-responsibility boundary at that point, nothing to do now. |
| People / HR security | CC1.4 | A.6.1–6.8 | N/A — one contributor, no employees | Org-level; not applicable to a solo take-home. Relevant the moment there's a second contributor or an employee. |
| Secure development lifecycle | CC8 | A.8.25–8.31 | Tests required, review checklist includes a security section (`CODE_REVIEW.md` §3), no secrets in source. CI now runs `pip-audit` (SCA) and `bandit` (SAST) as required checks (`.github/workflows/tests.yml`); a real finding (`PYSEC-2026-3447` in `setuptools`) turned up and was fixed by pinning `setuptools>=83` in `pyproject.toml` before wiring the scanner in, so it started clean | Still no documented threat model. Dependabot *alerts* (as opposed to the version-update PRs already configured in `.github/dependabot.yml`) need someone with repo admin access to flip on under Settings → Security — one toggle, no code change. |

## 3. If this becomes real: prioritized, concrete changes

In the order they'd start mattering, cheapest/most-code-relevant first:

1. ~~Dependency scanning~~ — done: `.github/dependabot.yml` (weekly version-update PRs for `pip` and the GitHub Actions used in CI) plus `pip-audit` and `bandit` as required CI checks (`.github/workflows/tests.yml`). Dependabot *alerts* (distinct from the version-update PRs) still need someone with repo admin access to flip on under Settings → Security, same as branch protection in `AGENTS.md`.
2. **Startup consent notice** — one line added to the CLI's startup output once `output.py` exists (M3), stating what's captured and what leaves the device before the camera starts. Small, and honest regardless of compliance goals.
3. **Explicit TLS assertion in `escalation.py`** (M5) — refuse to send an escalation request if the configured OpenRouter URL isn't `https://`, instead of relying on it happening to be right.
4. **Branch protection turned on** in GitHub settings, closing the gap `AGENTS.md` already flagged.
5. **Retention policy for `sessions/`** — a documented (and eventually enforced) max age or size for logs and recorded frames, once anyone besides the developer runs this regularly.
6. **Auth design for `ARCHITECTURE.md`'s future "Session service · per connection"** — before that box is built for real, it needs an actual per-tenant access-control design, not just the "per user/session" column in the shared-vs-per-tenant table.
7. **Vendor review of the model gateway's downstream provider(s)** — a paperwork step (data retention/training-use policy, DPA), triggered the moment a real (non-developer) user's image data would be sent.
8. **Formal risk register + asset inventory** — the actual first step of an ISO 27001 ISMS; makes sense once there's a team, not for a one-person take-home.
9. **Incident response contact/plan** — a line in `README.md` is enough while this is a demo; a real plan once it's operated for others.

Items 8–9, and the vendor-contract work in item 7, aren't things a future
PR can "finish" — they're ongoing organizational commitments an audit
checks for evidence of over time, not a one-time patch.

## 4. Explicitly out of scope for this take-home

Per `PRODUCT.md`'s stated non-goals (no hosted service, no multiple
simultaneous sessions), none of the multi-tenant, org-level, or
vendor-contract items above should be *built* here — doing so would be
exactly the kind of scope creep `CLAUDE.md` warns against. This doc exists
so the design decisions already being made (data minimization, secrets
handling, change management) are the ones that keep a *future* real
version's compliance story cheap, without spending the take-home's
time-box building infrastructure no one asked for yet.
