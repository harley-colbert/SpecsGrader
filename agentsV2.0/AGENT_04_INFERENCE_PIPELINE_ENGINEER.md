# AGENT_04_INFERENCE_PIPELINE_ENGINEER — Vector DB Top‑K Retrieval + Ensemble Scoring

## Role
Upgrade inference to use the vector DB for similarity and improve scoring stability:
- open vector DB during model load when available
- replace single-best match with top‑K query + weighted voting
- add outlier / needs-review logic
- begin using semantic LR probabilities (predict_proba) and fuse scores (Phase 06)

## Primary Phases
- Phase 05 (required)
- Phase 06 (required)

## Inputs
- SpecGrader repo
- `vector_db.py` exists (Phase 02)
- Training writes vector DB keys into model sets (Phase 03/04)
- Phase docs: `Phase_05_Inference_TopK_VectorDB.md`, `Phase_06_Ensemble_Scoring.md`

## Outputs
### Modified
- `logic.py`
  - `load_all_models(...)` opens vector DB when available and stores handle in `models["vector_db"]`
  - `multipass_classify(...)` adds parameters:
    - `sim_checkbox: bool`
    - `top_k: int`
    - `similarity_threshold: float`
  - `multipass_classify(...)` uses vector DB top‑K query + weighted voting
  - `multipass_classify(...)` adds columns:
    - `Top Similarity`
    - `KNN Risk`
    - `KNN Dept`
    - `KNN Confidence`
    - `KNN Evidence` (JSON string or compact text)
    - `Needs Review` (bool)
    - plus semantic proba columns from Phase 06

- `splash.py`
  - preload vector DB softly (warnings ok; crashes not allowed)

- Optional `similarity_engine.py`
  - keep as fallback for legacy model sets (no vector DB keys or DB missing)

## Phase 05: Vector DB Integration Requirements
1) Load path:
- If model set file_dict includes `vector_db_dir` and `vector_collection`, open the DB and store handle.

2) Backward compatibility:
- If vector DB not available but embeddings pickle exists, use current similarity_engine fallback.
- If neither is available, similarity must be skipped gracefully.

3) Weighted voting (mandatory)
Given top‑K matches each with similarity `s`:
- weight `w = max(s, 0.0) ** 2`
- vote risk labels and dept labels separately
- predicted label = argmax(sum weights by label)
- confidence = (max label weight sum) / (total weight sum + 1e-8)

4) Outlier rule (v2.0 baseline)
- if top1_similarity < similarity_threshold:
  - `Needs Review = True` (or `Outlier = True` plus `Needs Review`)

5) Evidence capture
- Include top‑K items with:
  - similarity, label(s), preview text
- Store as JSON string or list, but do not break UI table rendering.

## Phase 06: Ensemble Scoring Requirements
1) Compute semantic probabilities:
- `models['clf_rl'].predict_proba(X_vec)`
- `models['clf_rd'].predict_proba(X_vec)`
Decode using semantic label encoders:
- `models['sem_risklevel_le']`
- `models['sem_reviewdept_le']`

2) Replace winner-take-all trust:
Keep it simple and debuggable:
- compute scores:
  - `rule_trust`
  - `classic_trust`
  - `semantic_trust`
  - `knn_confidence`
- choose labels using max-score source BUT apply a minimum threshold:
  - if max_score < 0.60 -> `Needs Review=True`

Rules may still force labels, but you must:
- allow `Needs Review` to be True if models disagree strongly

## Must-Pass Tests (no errors)
From repo root:

1) Minimal inference dataset:
Create `tests/data/infer_min.csv` with:
Risk Description
"nfpa 79 wiring requirements"
"need emergency stop category 3"
"provide documentation package"

2) Train the minimal model set (Phase 04) if needed:
- `python -c "import logic; print(logic.train_all_models('tests/data/train_min.csv', model_set_name='v2_smoke'))"`

3) Run:
- `python -m compileall .`
- `python -c "import pandas as pd, logic; files=logic.load_model_set('v2_smoke'); models=logic.load_all_models(files); df=pd.read_csv('tests/data/infer_min.csv'); out=logic.multipass_classify(df, models, sim_checkbox=True, top_k=5, similarity_threshold=0.55); print(out.head())"`

Expected:
- no exceptions
- output includes new similarity + needs-review + semantic proba columns

## Success Checklist
- [ ] Vector DB is used for top‑K similarity when available
- [ ] Legacy fallback remains functional
- [ ] Weighted voting works and outputs evidence fields
- [ ] Semantic probabilities are computed and included
- [ ] `Needs Review` is emitted and behaves per thresholds
- [ ] All listed commands run without errors
