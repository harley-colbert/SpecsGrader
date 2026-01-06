# Phase 04 — Training: TF-IDF baseline, imbalance handling, and bundle artifacts

## Goal
Implement the best-practice default training approach for severe imbalance:

- TF-IDF + Logistic Regression (class-weighted)
- Calibrated probabilities
- Macro metrics + per-class recall + balanced accuracy
- Optional capped oversampling (text-safe; no SMOTE) as a training toggle
- Produce artifacts saved into a model bundle:
  - `bundle_meta.json`
  - `vector_embedder.json` (placeholder in this phase if embeddings not built yet)
  - `llm_prompt.json` (placeholder)
  - `rules_config.json` (from Phase 03)

This phase trains TWO models:
- Risk Level model (5 classes)
- Department model (4 classes)

## Agents
- Agent_MLTrainer (lead)
- Agent_BackendEngineer
- Agent_QA
- Agent_RepoAuditor
- Agent_FrontendEngineer (training UI)

## Backend: TrainingService
Create `backend/app/services/training_service.py`:

### Inputs
- TrainingDataset (from ingest)
- training params:
  - `oversample_enabled` (bool)
  - `oversample_cap_ratio` (float; e.g., 0.3 meaning minority capped at 0.3× majority)
  - `min_recall_per_class` (float)
  - `calibration_method` (sigmoid/isotonic; default sigmoid)

### Outputs
- trained artifacts stored in a workspace directory (later moved to bundle)
- metrics object:
  - macro_f1
  - balanced_accuracy
  - per_class_recall dict
  - confusion matrix (optional, but recommended for UI)
- class distribution stats

### Mandatory imbalance features
- `class_weight="balanced"` OR explicit inverse frequency weights
- probability calibration using `CalibratedClassifierCV`
- threshold-ready probability outputs (store predict_proba)

### Oversampling (optional)
- Implement simple oversampling by duplicating minority examples up to cap ratio
- NO SMOTE for text

## Bundle structure (created in Phase 07/08 save flow, but define now)
Plan for bundle folder like:
`bundles/<bundle_id>/...`

This phase must produce:
- `bundle_meta.json` (in workspace initially)
  - created_at
  - trained_on_rows
  - label distributions
  - metrics for both tasks
- serialized sklearn pipelines for:
  - level_model.joblib
  - dept_model.joblib

## API endpoints
- `POST /api/train/start`
- `GET /api/train/status`
- `POST /api/train/cancel`
- `GET /api/train/metrics`

Training should run in background thread and stream progress (simple polling is fine).

## Frontend (Train pane)
Add:
- Training parameters UI:
  - oversample toggle + cap
  - min recall per class
- “Train” button
- Training progress bar + cancel
- Metrics panel (macro F1, per-class recall table)
- “Quality gate” warning if min per-class recall below threshold

## Testing (must run and pass)
```bash
python -m pytest -q
```

Required tests:
- training produces pipelines that can predict
- class weights are set (verify in pipeline or via behavior)
- calibration wrapper exists
- metrics computed and returned
- oversampling cap logic works (unit tests)

Manual:
- Train on a small fixture dataset; metrics render in UI

## Success checklist
- [ ] TF-IDF + class-weighted LR models train successfully
- [ ] Probabilities produced (predict_proba) and calibrated
- [ ] Macro metrics computed and shown
- [ ] Optional capped oversampling works and is test-covered
- [ ] Background training job supports progress + cancel
