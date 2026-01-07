# Phase 1 — Add Model Inference as a first-class method + mode plumbing

## Objective
Make the supervised model participate in classification as a first-class method:
- Add `ModelInferenceService` that loads cached joblib pipelines and returns `pred/conf` for dept and level
- Include `method_outputs["model"] = {...}` in classification worker
- Add **mode plumbing** so the backend can run:
  - `mode="sanity"` (model-only)
  - `mode="evaluate"` (metrics pipeline)
  - `mode="production"` (decision ladder)
  - plus an optional `enabled_methods` override for advanced users

At the end of Phase 1:
- The app can classify using the supervised model (even if the decision ladder is not final yet).
- `aggregate_service.DEFAULT_WEIGHTS` includes `"model"` and aggregation accepts it.

## Assigned agents
- BackendAgent (lead)
- OrchestratorAgent (support)
- QualityAgent (tests)

## Files expected to change
Backend:
- `backend/app/services/model_inference_service.py` (NEW)
- `backend/app/main.py`
- `backend/app/services/aggregate_service.py`
- (optional) `backend/app/state.py` (store active model availability paths/status)

Frontend (minimal wiring only if needed to pass mode):
- `frontend/src/api/client.js`
- `frontend/src/panes/classifyPane.js`

Tests:
- `tests/test_model_inference_service.py` (NEW)
- extend or add route tests as needed

## Implementation steps
1. Create `ModelInferenceService`
   - Inputs: `app_state` (for current model paths), or explicit paths
   - Behavior:
     - lazy load
     - cache by (path, mtime)
     - thread-safe load using a lock
     - `predict(text) -> {dept_pred, dept_conf, level_pred, level_conf}`
     - if models missing, return `available:false` structure (do not raise)
2. Wire into classification
   - In classify worker (the code that builds `method_outputs`):
     - if mode permits and model available: call model inference and add `"model"`
3. Add mode parameter to classify start payload
   - Add `mode` field with allowed values: `sanity|evaluate|production`
   - Define default as `production`
   - Define that `enabled_methods` can override defaults for power users
4. Update aggregation defaults
   - Update `DEFAULT_WEIGHTS` to include `"model"` with highest weight
   - Ensure aggregator ignores missing methods cleanly
5. Add an API endpoint that reports “active capabilities”
   - Example: `GET /api/state` already exists; ensure it includes:
     - whether model artifacts exist (bool)
     - whether vector store exists (bool)
     - whether rules exist (bool)
     - currently selected mode defaults

## Tests that must pass
### Automated
- `pytest -q`

Required test cases:
- `ModelInferenceService`:
  - loads a saved model and returns valid preds + probabilities
  - gracefully returns not-available when artifacts missing
- Classification integration:
  - calling classify with `mode=production` includes `"model"` in method_outputs when artifacts exist
  - no crash when artifacts missing

### Manual
- Start app: `python run.py`
- Train on a small dataset
- Run classification with model enabled (default)
- Verify results show `"model"` method contribution (in debug output or result JSON if UI doesn’t expose yet)

## Success checklist
- [ ] New `ModelInferenceService` exists and is cache-safe
- [ ] Classify worker includes `"model"` when artifacts exist
- [ ] `DEFAULT_WEIGHTS` includes `"model"`
- [ ] `mode` parameter accepted by classify API and does not break existing UI
- [ ] No new console error spam and no 404 spam for “missing” optional resources
- [ ] `pytest -q` passes
