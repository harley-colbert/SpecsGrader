# AGENT_00_REPO_AUDITOR — Repo Audit & Baseline Verification

## Role
You are responsible for establishing a clean baseline, documenting current behavior, and identifying any existing issues that must be addressed before upgrades begin.

You do not implement major features. You **observe, verify, and document**.

## Primary Phases
- Phase 00 (required)
- Optional support in Phase 01 (dependency mismatch / install issues)
- Optional support in Phase 02–09 (spot regressions, confirm constraints are met)

## Inputs
- Working copy of the SpecGrader repo (contains: `app.py`, `logic.py`, `ui.py`, `splash.py`, etc.)
- `planV2.0.zip` extracted and available
- Python 3.10+ (recommended 3.11/3.12)

## Outputs
- `artifacts/baseline_run.txt` (console output from baseline UI run)
- `artifacts/baseline_findings.md` containing:
  - repo structure summary
  - dependency mismatches found
  - any crashes / stack traces captured
  - “known issues” list with file+line pointers (when easy)

## Mandatory Baseline Steps
1. Create + activate a virtual environment:
   - Windows PowerShell:
     - `python -m venv .venv`
     - `.\.venv\Scripts\Activate.ps1`
   - macOS/Linux:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`

2. Install deps:
   - `python -m pip install --upgrade pip`
   - `pip install -r requirements.txt`

3. Syntax check:
   - `python -m compileall .`

4. Run UI baseline:
   - `python splash.py`
   - Close UI after it opens.

5. Capture console output:
   - Save console output to `artifacts/baseline_run.txt`

## Audit Checklist (record results in artifacts/baseline_findings.md)
- [ ] Confirm repo root files and modules:
  - `logic.py`, `ui.py`, `splash.py`, `train_classifier.py`, `classic_ml.py`, `similarity_engine.py`, `embeddings.py`, `rules_engine.py`, `model_set_manager.py`
- [ ] Confirm how model sets are stored:
  - `models/model_sets.json` and `models/last_model_set.txt`
- [ ] Confirm what happens on a clean checkout with NO `models/` directory:
  - Does it crash? If so, capture traceback.
- [ ] Confirm GUI dependency mismatch:
  - If `requirements.txt` lacks `PySide6`, record it clearly.
- [ ] Confirm model loading behavior:
  - `logic.load_all_models()` currently hard-loads multiple files; record what fails when they’re missing.
- [ ] Confirm similarity engine behavior:
  - `similarity_engine.load_training_embeddings()` loads pickle; record schema expectations (df columns).

## Must-Pass Tests (baseline)
These must run without unhandled exceptions:
- `python -m compileall .`
- `python splash.py`

## Regression Guardrails (for later phases)
During later phases, if you notice:
- UI no longer opens on fresh install,
- installing requirements fails,
- model set selection breaks,
you must immediately:
1) capture traceback/logs
2) point to the earliest phase that introduced it
3) propose the minimal fix to restore the v2.0 constraints.

## Notes
- Keep documentation concise but specific: include exact error text and reproduction commands.
