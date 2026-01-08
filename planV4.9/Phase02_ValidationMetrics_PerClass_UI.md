# Phase 02 — Per-class Metrics + Confusion Matrices in UI

## Goal
Make validation and evaluation outputs industry-credible by displaying per-class precision/recall/F1 and confusion matrices for both risk and department.

## Primary agents
- BackendAgent
- FrontendAgent
- TestAgent

## Scope
- Backend: extend metrics to include precision + per-class F1 in addition to existing recall/confusion.
- Frontend: render metrics tables and confusion matrices in Train pane validation sections.

## Implementation steps (do in order)
- BackendAgent:
-   1) In `backend/app/services/training_service.py`, metrics currently include macro_f1, balanced_accuracy, per_class_recall, confusion_matrix.
-   2) Extend metrics for both `level` and `dept` to include:
-      - `per_class_precision`
-      - `per_class_f1`
-      - `weighted_f1` (and keep `macro_f1`)
-   3) Ensure class order is deterministic (`pipeline.classes_`).
- 
- FrontendAgent:
-   1) In `frontend/src/panes/trainPane.js`, in the validation/results section(s):
-      - Render summary cards: Macro F1, Weighted F1, Balanced Accuracy
-      - Render per-class table: Precision/Recall/F1 per label
-      - Render confusion matrix grid with labels on axes
-   2) Call out risk recall for `high` and `extreme` (bold or warning if below threshold).
-   3) Ensure empty/missing metrics do not crash rendering.

## Testing work to CREATE/UPDATE in this phase
- Update/extend `tests/test_evaluate_mode.py` and/or `tests/test_training_flow_smoke.py` to assert:
-   - metrics response now contains precision and per-class f1 keys
-   - confusion matrices are square and match number of classes
- Add `tests/test_metrics_schema.py` to validate schema for both label types.

## Tests that MUST pass (gate)
- `python -m pytest -q`

## Success checklist (must be YES for every item)
- ✅ Validation UI shows per-class Precision/Recall/F1 for risk and dept.
- ✅ Confusion matrices render correctly and match backend labels.
- ✅ Backend metrics schema is covered by automated tests.
