# Phase 0 — Baseline + reproduce the issue (v4.3)

## Objective
Establish a clean baseline and reproducibly demonstrate the reported behavior:
- training on ~17 rows
- running classification on those same rows (without labels)
- observing mismatch

This phase also inventories the existing code paths so later phases make correct changes.

## Assigned agents
- OrchestratorAgent (lead)
- QualityAgent (support)

## Implementation steps
1. Unzip `SpecsGraderV4.3.zip` into a working folder.
2. Install deps and start app:
   - `pip install -r requirements.txt`
   - `python run.py`
3. Run unit tests:
   - `pytest -q`
4. Reproduce mismatch:
   - Load training fixtures (or your 17-row sheet)
   - Train
   - Run classify on the same inputs with labels removed
   - Capture results + screen recording or screenshots
5. Confirm current classification pipeline does **not** use the trained joblib models:
   - Identify where classification methods are assembled (backend classify worker).
   - Identify where `level_model.joblib` and `dept_model.joblib` are written (training service).
6. Write baseline notes in a new file `planV4.4/Phase00_Baseline_notes.md` (create if missing):
   - dataset used, row count
   - which methods were enabled
   - what mismatches occurred
   - browser console errors (if any)

## Tests that must pass
- `python run.py` starts without traceback
- `pytest -q` passes
- Manual: reproduce mismatch and record baseline notes

## Success checklist
- [ ] App starts from repo root with `python run.py`
- [ ] `pytest -q` passes
- [ ] Mismatch reproduced and documented
- [ ] Confirmed: trained supervised models are saved but not used in classification (baseline finding)
