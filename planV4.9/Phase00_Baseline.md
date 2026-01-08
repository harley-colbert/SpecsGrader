# Phase 00 — Baseline, Branching, and Golden Smoke

## Goal
Establish a clean baseline from the provided v4.6/v4.5 codebase so every later phase is measurable and reversible.

## Primary agents
- OrchestratorAgent (owner)
- QAAgent
- TestAgent

## Scope
- No feature changes yet.
- Verify current test suite and document current UI behavior with screenshots.

## Implementation steps (do in order)
- Unzip the provided `SpecsGraderv4.6` (or latest available) into a fresh working directory.
- Create a git branch: `upgrade/v4.9`.
- Run tests: `python -m pytest -q` and capture output in `workspace/exports/logs/phase00_pytest.txt`.
- Run app: `python run.py` and verify the window opens (or headless fallback).
- Capture screenshots of Train pane (Path A and Path B) and Classify pane, store under `workspace/exports/screenshots/phase00_*`.
- Record baseline version string displayed in UI (if present). If not present, note that v4.9 will add an About/version label.

## Testing work to CREATE/UPDATE in this phase
- Add/Update `tests/test_smoke_phase00.py` (or extend existing `tests/test_smoke.py`) to assert:
-   - `GET /health` (or existing health/state endpoint) returns 200
-   - `GET /api/state` returns structured state and does not 404
-   - Listing ModelSets endpoint returns 200
- If health endpoint does not exist, create one in `backend/app/main.py` and test it.

## Tests that MUST pass (gate)
- `python -m pytest -q`

## Success checklist (must be YES for every item)
- ✅ Tests pass on baseline.
- ✅ App launches (webview or headless fallback) without exceptions.
- ✅ Baseline screenshots captured and committed to workspace exports (not necessarily git).
- ✅ Working branch `upgrade/v4.9` created.
