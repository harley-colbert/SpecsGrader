# Phase 05 — Vector embeddings (local default), index, and vector classifier option

## Goal
Implement the vector method used during classification, with local embeddings by default.

Vector method must output:
- dept_pred + dept_conf
- level_pred + level_conf
- similarity evidence (top neighbors)

Internally, implement:
1) kNN similarity voting over embeddings (mandatory)
2) optional supervised classifier on embeddings (recommended), cost-sensitive + calibrated

All vector artifacts live in bundle folder `vector_store/`.

## Agents
- Agent_VectorStore (lead)
- Agent_MLTrainer
- Agent_BackendEngineer
- Agent_QA

## Embedding strategy
Default: local embeddings.
Choose and implement ONE local embedding backend that is reasonable for desktop internal use
(e.g., `sentence-transformers` OR `fastembed`), and keep a clean interface so an OpenRouter
embedding option can be added later.

Create `backend/app/vector/embedder.py`:
- `embed_texts(list[str]) -> np.ndarray`
- Stores config in `vector_embedder.json`

## VectorStore
Create `backend/app/vector/vector_store.py`:
- persists:
  - embeddings matrix
  - training row metadata (risk_text, labels)
  - ANN index (FAISS, Annoy, or sklearn NearestNeighbors; pick based on dependency tolerance)
- provides:
  - `query(text, k) -> neighbors with similarity`

Persist under:
`vector_store/`
- `embeddings.npy` (or parquet)
- `rows.jsonl`
- `index.*`
- optional `embed_level_clf.joblib`, `embed_dept_clf.joblib`, calibration artifacts

## Vector prediction (method output)
Create `backend/app/services/vector_service.py`:
- `predict(risk_text) -> MethodPrediction`
  - compute kNN vote distribution for dept + level
  - compute confidence from:
    - calibrated classifier probability if enabled
    - else normalized neighbor vote share

## API endpoints
- `POST /api/vector/build` (during training or bundle creation)
- `POST /api/vector/test` (ad hoc query to show neighbors + predicted labels)

## UI
Train pane:
- “Build vector store” step after training finishes (or automatic)
- show embedding backend config
- show build progress

Classify pane:
- toggle “Vector”
- set `k` and maybe “use embedding classifier” toggle (advanced)

## Testing (must run and pass)
```bash
python -m pytest -q
```

Required tests:
- embedding output shape correct
- vector store build + reload works
- query returns deterministic neighbors for fixture embeddings
- vector predict returns both dept+level with confidence

Performance smoke:
- 1,000 rows vector inference completes without UI lock (manual acceptable, but add a simple timed test if stable)

## Success checklist
- [ ] Local embeddings work by default
- [ ] Vector store persists in `vector_store/`
- [ ] Vector method produces predictions + confidence + evidence
- [ ] Vector artifacts are bundle-ready (no hidden state outside bundle/workspace)
