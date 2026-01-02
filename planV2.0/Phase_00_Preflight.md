# Phase 00 — Preflight Snapshot & Baseline Verification

## Agents
- Run: **AGENT_00_REPO_AUDITOR**
- Optional assist: **AGENT_01_DEPENDENCY_MANAGER**

## Goal
Establish a clean baseline and prevent “mystery regressions”:
- confirm current app starts
- confirm current training/inference scripts execute (even if they warn about missing models)
- capture baseline behavior and logs

## Inputs
- Working copy of repo (folder containing: `app.py`, `logic.py`, `ui.py`, etc.)
- Python 3.10+ (recommended 3.11/3.12)

## Outputs
- A short baseline log file saved to `artifacts/baseline_run.txt`
- Confirmed commands that run without errors (or a documented baseline error list)

## Implementation Steps
1. Create and activate a virtual environment.
   - Windows (PowerShell):
     - `python -m venv .venv`
     - `.\.venv\Scripts\Activate.ps1`
   - macOS/Linux:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`

2. Install dependencies:
   - `python -m pip install --upgrade pip`
   - `pip install -r requirements.txt`

3. Syntax check (must not raise errors):
   - `python -m compileall .`

4. Baseline run:
   - `python splash.py`
   - Close the UI window after it opens.

5. Create baseline artifacts:
   - Create folder `artifacts/`
   - Save console output (copy/paste) to `artifacts/baseline_run.txt`

## Must-Pass Tests (no errors)
- `python -m compileall .`
- `python splash.py`  (UI must open; no unhandled exceptions)

## Success Checklist
- [ ] Virtual environment created and activated
- [ ] `pip install -r requirements.txt` succeeds (note any missing packages)
- [ ] `python -m compileall .` succeeds
- [ ] `python splash.py` opens the splash + main UI window without crashing
- [ ] Baseline output captured in `artifacts/baseline_run.txt`

## Notes / Known Baseline Issues to Expect
- `requirements.txt` currently mentions `tk` but the UI uses **PySide6**. This will be corrected in Phase 01.
- The app may warn about missing files under `models/` on first run; those warnings are acceptable, crashes are not.
