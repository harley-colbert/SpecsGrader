# Phase 03 — Model Insights: Top Terms per Class

## Goal
Provide interpretability: show the top TF‑IDF terms that drive each class prediction for risk and department.

## Primary agents
- MLAgent
- BackendAgent
- FrontendAgent
- TestAgent

## Scope
- Backend: add an API to compute and return class-term importances for the supervised model.
- Frontend: add a Model Insights panel to Train pane (available after training).

## Implementation steps (do in order)
- MLAgent/BackendAgent:
-   1) Problem: the current production pipeline uses `CalibratedClassifierCV`, which is awkward to extract stable coefficients from.
-   2) Solution (recommended for v4.9): train and persist a *shadow* uncalibrated LogisticRegression pipeline for insights only:
-      - `level_insights_model.joblib`
-      - `dept_insights_model.joblib`
-      These use the same TF‑IDF config as the production pipeline.
-   3) Add a `ModelInsightsService` (new file) or add methods to `training_service.py` to:
-      - load the insights models
-      - return top N terms per class (positive coefficients) for both label types
-      - include label list and metadata (N used, vectorizer ngram_range, max_features)
-   4) Add a backend route: e.g. `GET /api/modelsets/{id}/insights`.
- 
- FrontendAgent:
-   1) Add a collapsible panel or new tab section in Train pane:
-      - Dropdown: (Risk | Department)
-      - Dropdown: class
-      - Numeric: Top N (default 20)
-      - List results (term + weight)
-   2) Disable the panel until a trained bundle exists.

## Testing work to CREATE/UPDATE in this phase
- Add `tests/test_model_insights.py`:
-   - Train on a tiny synthetic dataset where class terms are obvious
-   - Assert the returned top terms include the expected keywords for each class
- Add fixtures: `tests/fixtures/insights_synthetic.csv`.

## Tests that MUST pass (gate)
- `python -m pytest -q`

## Success checklist (must be YES for every item)
- ✅ Model Insights endpoint returns stable top terms per class.
- ✅ UI can browse top terms for both risk and dept.
- ✅ Automated tests verify insights output on synthetic data.
