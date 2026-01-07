# Phase 0 — Baseline, inventory, and safety rails

## Objective
Establish a clean baseline for v4.2.2 and ensure you can reliably run the app and tests before making changes.

## Scope
- No feature changes.
- Add/adjust only what is necessary to run repeatable tests and capture baseline behavior.

## Implementation steps
1. **Unzip/checkout** the current working code (v4.2.2).
2. Confirm you can start the app:
   - `python run.py`
3. Confirm existing tests run:
   - `pytest -q` (if present)
4. Capture baseline UI behavior:
   - Open Train pane
   - Verify ModelSet section location (currently bottom)
   - Load a modelset version and observe Training/Vector/Rules hydration (v4.2.2 Phase 1 behavior)
5. Add a minimal “smoke test” if none exists:
   - A test that imports the FastAPI app and hits a simple endpoint (e.g., `/api/health` or similar).
   - If no health endpoint exists, add one **only if needed** and document it.

## Tests that must pass
1. App starts:
   - `python run.py`
   - Expected: backend + frontend served, no crash.
2. Unit tests:
   - `pytest -q`
   - Expected: all tests pass (or if no tests, the smoke test passes).
3. Manual sanity:
   - Load Train pane, click around; no console spam.

## Success checklist
- [ ] App starts from repo root with `python run.py`
- [ ] `pip install -r requirements.txt` completes without errors
- [ ] `pytest -q` passes (or smoke test passes if this is the first test)
- [ ] Baseline notes recorded (what works, what is missing)
