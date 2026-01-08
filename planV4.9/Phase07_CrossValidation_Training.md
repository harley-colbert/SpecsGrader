# Phase 07 — Cross-Validation Training + Honest Metrics

## Goal
Upgrade training to run k-fold cross-validation (default k=5) and store CV metrics in the ModelSet metadata. This provides stable, best-practice metrics instead of a single split.

## Primary agents
- MLAgent
- BackendAgent
- FrontendAgent
- TestAgent

## Scope
- Backend: implement CV training/evaluation routine and expose results via job status and metadata.
- Frontend: show CV progress and CV metric summary.

## Implementation steps (do in order)
- MLAgent/BackendAgent:
-   1) In `training_service.py`, replace or augment `train_test_split` evaluation with k-fold CV:
-      - use `StratifiedKFold` for each label type (risk and dept).
-      - compute per-fold metrics and averaged metrics (macro/weighted/per-class).
-   2) Decide training flow:
-      - Run CV first for evaluation; then train final models on full dataset.
-   3) Extend `TrainingParams` to include:
-      - `cv_folds` (int, default 5)
-      - `use_class_weight_balanced` (bool, default True)
-   4) Store CV metrics in:
-      - training job status
-      - `bundle_meta.json` / `version.json` metadata.
- 
- FrontendAgent:
-   1) In Train Step 4 (Train Models): add control for CV folds (3/5/10).
-   2) During training poll, show:
-      - current fold / total folds
-      - running averages
-   3) After training, show CV summary metrics alongside final train-on-all metrics.

## Testing work to CREATE/UPDATE in this phase
- Add `tests/test_cv_training.py`:
-   - Use a small synthetic dataset
-   - Assert CV metrics keys exist and are numeric
-   - Assert training produces final model artifacts after CV
- Update `tests/test_training_flow_smoke.py` if training outputs change.

## Tests that MUST pass (gate)
- `python -m pytest -q`

## Success checklist (must be YES for every item)
- ✅ CV runs with deterministic folds (seeded) and returns stable metric schema.
- ✅ Final models are still trained and saved after CV.
- ✅ UI shows CV configuration and results.
- ✅ Tests cover CV schema and artifact outputs.
