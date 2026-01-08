# Phase 04 — Per-row “Why?” Explanations (Trace + Evidence)

## Goal
Every classification result must explain itself: which layer won (rules/model/vector/weighted/abstain/LLM) and what evidence drove the decision.

## Primary agents
- BackendAgent
- FrontendAgent
- TestAgent

## Scope
- Backend: extend method outputs and aggregation trace to include evidence needed by UI.
- Frontend: add a per-row expander in Classify pane to render explanations.

## Implementation steps (do in order)
- BackendAgent:
-   1) In `backend/app/services/aggregate_service.py`, the current `trace` contains steps and winner.
-   2) Extend aggregation to include an `evidence` object in `trace`, populated from available method outputs:
-      - Rules: `matched`, `hard_hits`, `is_hard` from `RulePrediction`
-      - Model: probabilities for the predicted class (already returned as conf); add optional `top_terms` if available
-      - Vector: `top_neighbors` and `neighbors` already exist; include top 3 neighbors and their labels/similarities
-   3) Add helper in model inference path to compute `top_terms_in_text` for the predicted class using the insights model (from Phase 03).
-      - If insights model is missing, return empty list but do not error.
-   4) Ensure `model_inference_service.py` can optionally return `proba` dict for both tasks to show confidence distribution.
-   5) Ensure API response for classification includes `trace` for each row.
- 
- FrontendAgent:
-   1) In `frontend/src/panes/classifyPane.js` and/or `frontend/src/panes/resultsPane.js`:
-      - Add a “Why?” toggle per result row.
-      - Render:
-        - Winner layer + key thresholds (if included)
-        - Rules evidence (keyword hits)
-        - Model evidence (confidence + top terms)
-        - Vector evidence (top neighbors list)
-   2) Keep the row compact; expand/collapse only renders details when opened.

## Testing work to CREATE/UPDATE in this phase
- Add `tests/test_explanations_trace.py`:
-   - Classify a small set and assert every result includes `trace.winner` and `trace.steps`
-   - When rules are enabled and keyword exists, assert trace includes rule evidence
-   - When vector store exists, assert trace includes at least 1 neighbor and similarity values
- Update existing tests if they assert exact response dicts (add backward-compatible fields).

## Tests that MUST pass (gate)
- `python -m pytest -q`
- Confirm passing:
- - `tests/test_production_policy.py`
- - `tests/test_aggregate_service.py`
- - `tests/test_explanations_trace.py` (new)

## Success checklist (must be YES for every item)
- ✅ Every classified row includes `trace` with winner and evidence.
- ✅ UI displays a clear, readable explanation without clutter.
- ✅ Tests prove trace exists and contains evidence when available.
