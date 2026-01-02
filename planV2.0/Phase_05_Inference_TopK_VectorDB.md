# Phase 05 — Load Vector DB and Use Top‑K Similarity in Inference

## Agents
- Run: **AGENT_04_INFERENCE_PIPELINE_ENGINEER**
- Optional assist: **AGENT_02_VECTOR_DB_ENGINEER**
- Optional assist: **AGENT_06_TEST_ENGINEER**

## Goal
Replace the current single-best match similarity (DataFrame cosine scan) with:
- Vector DB top‑K query
- Weighted voting for risk and dept
- Optional outlier detection based on top-1 similarity threshold

## Files to Modify
- `logic.py`:
  - `load_all_models(...)` should open vector DB when available
  - `multipass_classify(...)` should call vector DB top‑K query when enabled
- `splash.py`:
  - attempt to preload vector DB (soft-fail allowed)
- Optional: `similarity_engine.py`:
  - keep as fallback for legacy model sets only

## Implementation Steps
1. Add vector DB handle to the loaded models dict
   - Example key: `models["vector_db"]`

2. Add inference controls
   - `multipass_classify(df, models, sim_checkbox=True, top_k=5, similarity_threshold=0.55)`
   - If `sim_checkbox` is false → skip similarity
   - If vector DB missing but pickle embeddings exist → fallback to old path (v2.0 compatibility)

3. Implement weighted voting
   - For each query:
     - retrieve top‑K matches with similarity scores
     - vote risk and dept separately using weights:
       - `w = max(similarity, 0.0) ** 2`
     - predicted label = argmax(total weight)
     - confidence = (max weight sum) / (total weight sum + 1e-8)
   - Record evidence:
     - top‑K texts + labels + similarity (store as JSON string or list)

4. Outlier / needs-review rule (v2.0 baseline)
   - If top‑1 similarity < similarity_threshold → mark as `Outlier=True` or `Needs Review=True`

## Must-Pass Tests (no errors)
1. Train the minimal model set from Phase 04 (if not already done).
2. Create a minimal input CSV `tests/data/infer_min.csv`:
Risk Description
"nfpa 79 wiring requirements"
"need emergency stop category 3"
"provide documentation package"

3. Run:
- `python -m compileall .`
- `python -c "import pandas as pd, logic; models=logic.load_all_models('v2_smoke'); df=pd.read_csv('tests/data/infer_min.csv'); out=logic.multipass_classify(df, models, sim_checkbox=True); print(out[['Final Risk Level','Final Review Dept']].head())"`

Expected:
- No exceptions
- Output DataFrame includes similarity evidence fields when enabled

## Success Checklist
- [ ] Vector DB is opened during model loading when available
- [ ] Inference uses top‑K retrieval (not a full scan)
- [ ] Outlier / needs-review flag triggers when similarity is low
- [ ] All commands run without errors on the minimal dataset
