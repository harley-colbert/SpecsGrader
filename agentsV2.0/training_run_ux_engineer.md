# training_run_ux_engineer.md

## Purpose
Improve training UX: staged progress messages, clear completion summary, and integration with model history and active model display.

## When to run
Phase 4 (required).

## Inputs
- Existing training pipeline code
- Where logs/progress currently appear (if at all)

## Outputs (files/artifacts)
- Train panel progress stages
- Post-train summary card (metrics + trained date + dataset size)
- Log linkage (copy/open)

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
1) Add human-readable stages:
   - Preparing data
   - Training model
   - Evaluating
   - Saving model
   - Done

2) Ensure stage updates are visible even if training is fast.

3) On completion, show a summary card:
   - Model name/version
   - Trained timestamp
   - Metrics: F1, Precision, Recall (or your best available)
   - Training dataset labeled count

4) Ensure logs remain available (Log tab) and include model id/version.

5) Trigger “Retrain recommended” indicator if labels change after training (hook provided by state map/guardrails).

## Testing
Manual:
- Start training → stages visible.
- Training completes → summary visible.
- Log shows training run start/end and model version.

Suggested automated:
- Unit test that training run produces a run record with metrics and timestamp.

## Success checklist (must complete)
- [ ] Staged progress messages visible during training
- [ ] Post-train summary displayed
- [ ] Logs include model version and run outcome


