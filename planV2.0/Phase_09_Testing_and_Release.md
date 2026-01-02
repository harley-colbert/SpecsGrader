# Phase 09 — Tests, Acceptance, and Release Packaging

## Agents
- Run: **AGENT_06_TEST_ENGINEER**
- Assist: **AGENT_07_RELEASE_MANAGER**

## Goal
Add repeatable tests so “v2.0” is verifiable:
- unit tests for vector DB wrapper (open/upsert/query)
- smoke test for training + inference on minimal dataset
- release checklist so future changes don’t break startup

## Files to Add
- `tests/test_vector_db_unittest.py`
- `tests/test_train_and_infer_unittest.py`
- `tests/data/train_min.csv`
- `tests/data/infer_min.csv`
- Optional: `scripts/smoke_all.py`

## Test Rules
- Tests must run with the standard library:
  - `python -m unittest -v`
- They must not require a GUI to pass.
- They must clean up any temporary vector DB directories they create.

## Must-Pass Tests (no errors)
1. Install deps:
   - `pip install -r requirements.txt`

2. Run:
   - `python -m compileall .`
   - `python -m unittest -v`

3. Optional smoke:
   - `python scripts/smoke_all.py` (if you add it)

## Release Packaging Steps
1. Update `README.md`:
   - new dependency list
   - how to train a model set
   - how vector DB is stored
   - how to migrate older model sets

2. Version tag (lightweight)
   - Add a `__version__ = "2.0.0"` constant (choose location: `app.py` or new `version.py`)
   - Display it in UI footer (optional)

3. Final manual acceptance
   - `python splash.py` opens the UI on a clean machine
   - training minimal dataset works
   - inference works
   - vector DB evidence shows up

## Success Checklist
- [ ] `python -m unittest -v` passes
- [ ] `python -m compileall .` passes
- [ ] Manual `python splash.py` run is stable
- [ ] README documents vector DB usage and migration
