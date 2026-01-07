# Phase 3 — Evaluate mode (honest metrics per method + ensemble simulation)

## Objective
Implement **Evaluate mode** to compute honest metrics and understand each signal independently.

Evaluate mode must compute metrics for:
- model-only
- rules-only
- vector-only (k=1, and optionally k=5)
- ensemble simulation (using the production policy, but run on held-out data)

After evaluation:
- refit final supervised model on 100% labeled data for production artifacts
- (optional) rebuild vector store from 100% labeled data
- store evaluation report in `training_snapshot.json` (or a dedicated file)

## Assigned agents
- BackendAgent (lead)
- QualityAgent (tests)
- FrontendAgent (UI)
- OrchestratorAgent (support)

## Files expected to change
Backend:
- `backend/app/main.py` (new endpoint)
- `backend/app/services/training_service.py` (add evaluate routine)
- `backend/app/services/vector_service.py` (support evaluation-time vector predictions if needed)
- `backend/app/services/rule_service.py` (rules-only scoring helper)
- `backend/app/services/aggregate_service.py` (ensemble simulation helper, still deterministic)
- (optional) `backend/app/state.py` (store last evaluation report)

Frontend:
- `frontend/src/panes/trainPane.js` (add “Evaluate” UI)
- `frontend/src/api/client.js`

Tests:
- `tests/test_evaluate_mode.py` (NEW)

## Implementation steps
1. Define evaluation strategy for small datasets
   - If dataset >= 30 and each class has >= 5 samples: use stratified k-fold (k=5)
   - If small (like 17 rows): use repeated stratified split (e.g., 5 repeats of 80/20) OR k-fold where k <= min_class_count
   - Always guard against folds with missing classes
2. Backend endpoint
   - `POST /api/train/evaluate`
   - Inputs:
     - evaluation config (optional): `strategy`, `k_folds`, `test_size`, `repeats`, `vector_k_values`
   - Outputs:
     - per-method metrics (accuracy, macro-F1, confusion optional)
     - coverage/abstain rates where applicable
     - ensemble override rates (how often each method won)
     - errors/warnings (e.g., “too few samples for stratified fold; used split strategy”)
3. Implement per-method scoring
   - model-only: predict on holdout with supervised model trained on train fold
   - rules-only: apply rules to holdout; abstain when no hits/ties
   - vector-only: build embeddings/index on train fold then query holdout (for honest retrieval metrics)
   - ensemble: run the production decision policy using only signals built from the train fold (simulate real behavior)
4. Refit final artifacts after evaluation completes
   - Train supervised model on 100% labeled data
   - Save to workspace bundle and/or modelset version
   - Store evaluation report in snapshot with `final_fit_n_rows`
5. UI
   - Add Evaluate button in Train pane
   - Display per-method metric summary + warnings
   - Provide a “download evaluation JSON” button (optional)

## Tests that must pass
### Automated
- `pytest -q`

Required cases:
- No training data → returns 200 with `available:false`
- With fixture labeled dataset:
  - returns per-method metrics objects with expected keys
  - strategy selection is deterministic and does not crash on tiny datasets
  - final refit step produces saved joblib artifacts

### Manual
- Train on 17-row dataset
- Run Evaluate
- Confirm you see:
  - model-only metrics
  - vector-only metrics (k=1 at minimum)
  - rules-only metrics
  - ensemble metrics + override rates
- Confirm evaluate mode does not permanently alter production settings unless explicitly saved

## Success checklist
- [ ] Evaluate endpoint exists and returns per-method metrics
- [ ] Works for small datasets with safe strategy fallback
- [ ] Refit final model on 100% labeled data after evaluation
- [ ] Evaluation report stored in snapshot for traceability
- [ ] UI displays evaluation results clearly
- [ ] `pytest -q` passes
