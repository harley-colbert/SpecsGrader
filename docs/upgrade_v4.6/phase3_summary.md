# Phase 3 Summary (Path B stepper)

## Updated files
- `frontend/src/panes/trainPane.js`
- `frontend/styles.css`

## Stepper & gating
- Added a 7-step horizontal stepper shown in `"build-update"` mode.
- Step completion is derived from current app state:
  - Step 1: ModelSet selected.
  - Step 2: Labeled training rows detected.
  - Step 3: Rules optional (marked modified on successful save).
  - Step 4: Training completed.
  - Steps 5–7: placeholders for later phases.

## Path B cards
- Step 1 card provides ModelSet selection and create-new controls.
- Step 2 card loads training data, shows labeled row counts and preview, and flags zero-labeled datasets with guidance.
- Step 3 card offers optional rules editing/testing with JSON validation and revert support.
- Step 4 card gates training, exposes advanced settings, and shows headline macro F1 metrics on completion.
- Steps 5–7 are placeholder cards with disabled actions.
