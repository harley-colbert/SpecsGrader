# Phase 2 — Sanity mode (training sanity report)

## Objective
Implement **Sanity mode** so the user can verify the supervised model “learned what you labeled”.

Sanity mode runs **model-only** predictions on the labeled training dataset and reports:
- train-set accuracy (sanity metric)
- per-row expected vs predicted (dept + level)
- confidence values
- confusion matrix (optional if label set is small)

Important:
- Sanity mode must NOT call rules/vector/LLM.
- Sanity mode does NOT replace real evaluation; it’s a quick correctness check.

## Assigned agents
- BackendAgent (lead)
- FrontendAgent (UI)
- QualityAgent (tests)
- OrchestratorAgent (support)

## Files expected to change
Backend:
- `backend/app/main.py` (new endpoint)
- `backend/app/services/training_service.py` (expose a function to fetch labeled training rows and/or latest training dataset snapshot)
- `backend/app/services/model_inference_service.py` (reuse)
- (optional) `backend/app/state.py` (store latest sanity report)

Frontend:
- `frontend/src/panes/trainPane.js` (add “Run Sanity Check” UI)
- `frontend/src/api/client.js` (new API call)
- (optional) `frontend/src/panes/resultsPane.js` (if you want a richer display)

Tests:
- `tests/test_sanity_mode.py` (NEW)

## Implementation steps
1. Backend: create endpoint
   - `POST /api/train/sanity` (or `/api/evaluate/sanity`)
   - Input:
     - optional `modelset_id/version_id` (if you want to run sanity against loaded modelset)
     - otherwise uses active workspace bundle
   - Output:
     - `available` (false if no trained model or no labeled dataset)
     - `n_rows`
     - `accuracy_level`, `accuracy_dept`
     - `rows`: list of {text_id/row_id, expected_level, pred_level, conf_level, expected_dept, pred_dept, conf_dept, match_level, match_dept}
     - optional `confusion` maps
2. Ensure the labeled dataset source is well-defined
   - Use the same training dataset rows that were used for training (or the same file snapshot)
   - If the app does not persist training rows, at minimum persist a `training_snapshot.json` that includes the labeled rows used (or a hash+file path reference).
3. Frontend: add UI
   - In Train pane, add a “Sanity Check” section:
     - button: Run Sanity Check
     - display summary accuracy + count
     - show a small table of mismatches with predicted/expected and confidences
4. Store last sanity results in memory (optional)
   - So switching panes doesn’t lose results instantly

## Tests that must pass
### Automated
- `pytest -q`

Required cases:
- If no trained model, sanity endpoint returns 200 with `available:false`
- After training on a fixture dataset:
  - sanity returns `available:true`
  - `n_rows` matches dataset
  - accuracies are computed and between 0 and 1
  - for training data, accuracy is “high” (set threshold, e.g., >= 0.8 given tiny datasets)

### Manual
- Train on your 17-row dataset
- Run Sanity Check
- Confirm the report explains mismatches and shows confidence values
- Confirm no rules/vector/llm were used (sanity endpoint must explicitly enforce)

## Success checklist
- [ ] Sanity endpoint exists and returns structured output (no crashes)
- [ ] Sanity uses model-only predictions (no other methods)
- [ ] Train pane displays sanity summary and mismatches clearly
- [ ] Missing-model/data results are 200 with `available:false` (no 404)
- [ ] `pytest -q` passes
