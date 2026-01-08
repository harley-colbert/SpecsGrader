# Phase 11 — Optional Transformer Embedder (Local Model Directory)

## Goal
Add a transformer embedding backend as an **optional plugin**: if dependencies + local model files exist, enable it; otherwise report `available:false` and safely skip. (This keeps SpecsGrader local-first and testable offline.)

## Primary agents
- MLAgent
- BackendAgent
- FrontendAgent
- TestAgent

## Scope
- Backend: implement TransformerEmbedder with graceful dependency/model presence checks.
- Frontend: expose selection but clearly label requirements.

## Implementation steps (do in order)
- MLAgent/BackendAgent:
-   1) Implement `TransformerEmbedder` in `backend/app/vector/embedder.py` (or new module).
-   2) Dependency strategy (recommended):
-      - Do NOT add heavy deps to base `requirements.txt`.
-      - Add an optional `requirements-transformer.txt` (documented) with:
-        - `sentence-transformers` (and its deps)
-   3) Model storage:
-      - Expect a local directory like `workspace/models/sentence_transformers/all-MiniLM-L6-v2/`
-      - If missing, return `available:false` when user selects transformer backend.
-   4) Add a helper endpoint `GET /api/embeddings/backends` returning:
-      - which backends are available on this machine
-      - why transformer is unavailable (missing deps vs missing model files).
- 
- FrontendAgent:
-   1) In Train Step “Build Vector Store” controls, add backend dropdown:
-      - TF‑IDF (fast)
-      - LSA (semantic, offline)
-      - Transformer (best semantic; requires extra install + local model folder)
-   2) If transformer selected and unavailable, show blocking message and auto-switch back to LSA or TF‑IDF.

## Testing work to CREATE/UPDATE in this phase
- Add `tests/test_transformer_backend_optional.py` using `pytest.importorskip('sentence_transformers')`:
-   - If dependency absent, test auto-skips (still passes suite).
-   - If present and local model dir exists, run a small smoke embedding and assert shape.
- Add `tests/test_embedding_backends.py` assertion that `/api/embeddings/backends` reports availability correctly.

## Tests that MUST pass (gate)
- `python -m pytest -q`

## Success checklist (must be YES for every item)
- ✅ Transformer backend can be selected when available.
- ✅ When unavailable, system explains why and does not crash.
- ✅ Tests remain green in offline/no-transformer environments.
