# Phase 01 — Dataset Health + Imbalance Warnings (Train Step 2)

## Goal
Expose training dataset distributions (risk + dept) and imbalance warnings in the Train pane so users immediately see whether the dataset is sufficient for reliable training.

## Primary agents
- BackendAgent
- FrontendAgent
- TestAgent

## Scope
- Backend: extend dataset preview/validation responses to include class counts, percentages, and warnings.
- Frontend: render Dataset Health panel in Path B Step 2 and gate later steps if dataset is unusable.

## Implementation steps (do in order)
- BackendAgent:
-   1) Locate dataset preview/validation method in `backend/app/services/training_service.py` (already produces `label_distribution`).
-   2) Extend the returned stats to include:
-      - `label_distribution_pct` (percentages per class for both level and dept)
-      - `warnings`: list of strings (missing classes, rare classes, too few labeled rows)
-      - `blocking_errors`: list of strings (no labeled rows, missing required columns, etc.)
-   3) Ensure the stats are persisted into the training job (`app_state.training_job['stats']`) so frontend can poll.
-   4) If the API currently returns stats only via job polling, add a direct endpoint to fetch dataset health from loaded dataset.
- 
- FrontendAgent:
-   1) In `frontend/src/panes/trainPane.js`, Path B Step 2 UI:
-      - render a table for risk distribution and dept distribution (counts + %).
-      - show warnings and blocking errors distinctly.
-   2) Enforce gating:
-      - If blocking_errors exist, disable Train/Vector steps and show the reason.
-      - If only warnings exist, allow progression but show warning badges.
-   3) Ensure this panel updates both on initial load and during polling.

## Testing work to CREATE/UPDATE in this phase
- Add `tests/test_dataset_health.py` to assert dataset health response includes:
-   - counts and pct fields for both `level` and `dept`
-   - warnings emitted when a class is missing or rare
-   - blocking error emitted when no labeled rows exist
- Add fixtures:
-   - `tests/fixtures/training_balanced.csv`
-   - `tests/fixtures/training_missing_extreme.csv`
-   - `tests/fixtures/training_unlabeled.csv`

## Tests that MUST pass (gate)
- `python -m pytest -q`
- Specifically confirm passing:
- - `tests/test_training_flow_smoke.py`
- - `tests/test_dataset_health.py` (new)

## Success checklist (must be YES for every item)
- ✅ Train Step 2 shows distributions (counts + %).
- ✅ Missing/rare classes produce warnings.
- ✅ No-labeled-data produces blocking error and disables later steps.
- ✅ Tests cover balanced vs missing vs unlabeled datasets.
