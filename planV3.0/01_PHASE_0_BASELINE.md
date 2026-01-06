    # Phase 0 — Baseline & guardrails

    ## Goal
    Lock down the current behavior of SpecsGraderV2.3 so the web migration doesn’t regress core functionality. Establish repeatable commands and add missing “sanity” checks.

    ## Prerequisites
    - You have `SpecsGraderV2.3` checked out locally.
- Python 3.12 available (or the current repo’s supported version).
- You can run `pip install -r requirements.txt`.

    ## Implementation steps (must follow in order)
    1. **Agent:** RepoAuditAgent
   - Produce a file map of current UI entry points and workflow calls.
   - Identify which `ui_actions.py` functions map to each workflow step.
   - Confirm how Excel/CSV ingestion currently works and where the “Quote #” sheet selection lives.

2. Add a top-level `DEV_COMMANDS.md` containing canonical commands:
   - create venv
   - install deps
   - run tests
   - run desktop UI (legacy)

3. Add a `scripts/smoke_desktop.py` (or equivalent) that:
   - imports core modules (`logic`, `vector_db`)
   - loads model_sets.json
   - prints a short “OK” summary
   - exits 0

4. Add a new test `tests/test_repo_smoke_unittest.py` that validates:
   - model_sets.json exists and is parseable
   - `logic.list_model_sets()` returns a list (can be empty, but must not error)
   - `vector_db` module imports successfully
   Keep it fast and deterministic.

    ## Files to create/change
    - **Create:** `DEV_COMMANDS.md`
- **Create:** `scripts/smoke_desktop.py`
- **Create:** `tests/test_repo_smoke_unittest.py`
- **Change (if needed):** `README.md` to link to DEV_COMMANDS.md

    ## Tests (must run and pass)
    Run from repo root:
1. `python -m pytest -q`
2. `python scripts/smoke_desktop.py`

    ## Success checklist (must be true before moving on)
    - ✅ `pytest` passes with no errors
- ✅ `scripts/smoke_desktop.py` exits with code 0
- ✅ RepoAuditAgent produced a workflow map artifact and it’s committed in `artifacts/` or documented in `DEV_COMMANDS.md`

    ## Notes for agents (assumed available)
    - RepoAuditAgent should output a short “UI→Logic call map” table.
- ParityTestAgent should confirm Phase 0 adds no functional changes.
