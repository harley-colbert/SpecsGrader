# model_versioning_engineer.md

## Purpose
Implement model version records, model history UI, and explicit active model selection; ensure classification and export always reference the active model.

## When to run
Phase 4 (required).

## Inputs
- Current model storage approach
- How model artifacts are saved (files, directories, db)

## Outputs (files/artifacts)
- Model registry (list of versions with timestamps and metrics)
- Model history UI with “Set Active”
- Active model header updates immediately

## Agent Operating Rules (do not skip)

- Make changes incrementally and test frequently.
- Prefer small, reviewable commits (if version control is available).
- Do not introduce new “mystery knobs.” If you add settings, explain them in UI copy.
- Avoid scattering business rules across widgets; centralize:
  - UX state derivation
  - gating rules
  - string/copy constants
- If you cannot determine the UI stack quickly, search the repo for:
  - `main.py`, `app.py`, `__main__`
  - `Tk()`, `QMainWindow`, `App()`, `createRoot`, `ReactDOM`


## Procedure
1) Define model registry schema:
   - model_id, version, trained_at, metrics, dataset_snapshot, artifact_path

2) Persist registry:
   - update on each training completion
   - load on app start

3) Build Model History UI:
   - list versions
   - show metrics
   - set active model

4) Ensure active model is used everywhere:
   - classification handler
   - export metadata
   - log statements

5) Add safety:
   - if active model missing/corrupt, show clear error and require user to pick another model.

## Testing
Manual:
- Train twice → history shows both.
- Switch active model → header updates.
- Classify → log shows active model id/version.
- Export → includes active model metadata.

Suggested automated:
- Unit test: registry write/read and active selection persist across restart.

## Success checklist (must complete)
- [ ] Model history exists with set-active
- [ ] Active model is unambiguous and displayed
- [ ] Classification/export/log always reference active model


