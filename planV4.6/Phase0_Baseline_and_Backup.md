# Phase 0 — Baseline and Backup

## Objectives

- Establish a clean baseline for SpecsGraderv4.5.
- Confirm that backend and frontend are working as expected.
- Capture reference screenshots of the current Train pane.
- Create a backup branch/tag before making v4.6 changes.

## Required Context

- SpecsGraderv4.5 source code extracted to a working directory.
- Python environment configured and dependencies installed.

## Tasks

1. **Setup and run v4.5**

   - Create and activate a virtual environment.
   - Install dependencies:
     - `pip install -r requirements.txt`
   - Start the app:
     - `python run.py`
   - Confirm the app is accessible in a browser (note the URL/port).

2. **Open the current Train pane**

   - Navigate to the Train pane in the UI.
   - Capture screenshots of:
     - The full Train pane with all sections visible.
     - The state **after loading a training dataset** (if you have a sample).
     - The state **after loading a ModelSet** without training data.

3. **Run the existing test suite**

   - Stop the app server if needed.
   - Run tests:
     - `pytest -q`
   - Save the test output (pass/fail summary).

4. **Create a v4.5 baseline branch/tag**

   - In git:
     - `git checkout -b v4.5-baseline` (or equivalent)
     - Optionally create a tag `git tag specsgrader_v4.5_baseline`

## Tests

- App runs without errors.
- Train pane loads and is interactive in its current (v4.5) form.
- `pytest -q` completes successfully (no new failing tests).

## Success Checklist

- [ ] App is running and reachable in a browser.
- [ ] Screenshots of the current Train pane are captured and stored in a docs or notes folder.
- [ ] `pytest -q` passes on v4.5.
- [ ] A baseline git branch or tag exists for v4.5.
