# Agent — Phase 0: Baseline and Backup

You are the Phase 0 agent for the SpecsGrader v4.6 upgrade.

## Goal

- Confirm SpecsGraderv4.5 currently runs and passes tests.
- Capture the visual state of the Train pane.
- Create a baseline git branch/tag for rollback.

## Inputs

- SpecsGraderv4.5 source code in the working directory.
- `planV4.6/Phase0_Baseline_and_Backup.md` for reference.

## Step-by-Step Instructions

1. **Set up the Python environment**
   - If a virtual environment does not exist, create one (e.g. `.venv`):
     - `python -m venv .venv`
   - Activate it:
     - On Windows: `.venv\Scripts\activate`
     - On macOS/Linux: `source .venv/bin/activate`
   - Install dependencies:
     - `pip install -r requirements.txt`

2. **Run the app**
   - Start the app:
     - `python run.py`
   - Verify that:
     - The server starts without tracebacks.
     - A browser can open the main UI at the configured URL/port.

3. **Capture the current Train pane UI**
   - In the running app, navigate to the Train pane.
   - Capture at least three screenshots (or equivalent documentation):
     - Full Train pane (default state).
     - Train pane after loading a training dataset (if any sample file is available).
     - Train pane after loading a ModelSet (if any existing ModelSet is available).
   - Save them under `docs/train_pane_v4.5/` or a similar folder.

4. **Run backend tests**
   - Stop the app server if needed, or run tests in another terminal.
   - Execute:
     - `pytest -q`
   - Save the test output into a text file, e.g.:
     - `tests/logs/pytest_v4.5_baseline.txt`

5. **Create baseline git branch/tag (if git repo is present)**
   - Check if `.git` directory exists.
   - If yes:
     - Create a branch:
       - `git checkout -b v4.5-baseline`
     - Optionally create a tag:
       - `git tag specsgrader_v4.5_baseline`
     - Do **not** push or expose any credentials in these instructions.

6. **Summarize results**
   - Write a short summary in `docs/upgrade_v4.6/phase0_summary.md` including:
     - Whether the app started successfully.
     - Whether `pytest -q` passed.
     - Locations of screenshots and test logs.

## Tests for This Phase

- `python run.py` starts the app without unhandled exceptions.
- `pytest -q` completes without new failing tests.

## Success Checklist

- [ ] App is running and reachable in a browser.
- [ ] Screenshots of the current Train pane are saved.
- [ ] `pytest -q` passes and log is saved.
- [ ] Baseline branch/tag created (if in git repo).
- [ ] `docs/upgrade_v4.6/phase0_summary.md` created with results.
