# Phase 10 — Embedding Backend Abstraction (TF‑IDF + LSA)

## Goal
Refactor vector embedding into a pluggable backend. Add an offline-friendly dense semantic backend (LSA via TF‑IDF + TruncatedSVD).

## Primary agents
- MLAgent
- BackendAgent
- TestAgent

## Scope
- Backend: refactor `backend/app/vector/embedder.py` into backend selection + persistence.
- Vector store must work with both sparse TF‑IDF and dense LSA embeddings.

## Implementation steps (do in order)
- MLAgent/BackendAgent:
-   1) Refactor `backend/app/vector/embedder.py`:
-      - Rename current `Embedder` to `TfidfEmbedder`
-      - Add `LsaEmbedder` that:
-        - builds TF‑IDF matrix
-        - applies `TruncatedSVD` to get dense vectors (configurable dims, e.g. 256)
-      - Create a factory `create_embedder(config)` returning the right embedder
-   2) Update `EmbedderConfig` to support:
-      - `model: 'tfidf' | 'lsa' | 'transformer'`
-      - for lsa: `svd_components`
-   3) Update persistence:
-      - Save config json
-      - Save vectorizer.joblib
-      - For LSA also save svd.joblib
-   4) Update `VectorStore` and `vector_service.py` to use the selected embedder.
-   5) Ensure existing `vector_embedder.json` load is backward compatible (missing svd -> tfidf).

## Testing work to CREATE/UPDATE in this phase
- Add `tests/test_embedding_backends.py`:
-   - Build a vector store with tfidf embedder and query neighbors (existing behavior)
-   - Build a vector store with lsa embedder and query neighbors
-   - Assert embedding shapes and persistence load/save work
- Update any existing vector store tests if necessary.

## Tests that MUST pass (gate)
- `python -m pytest -q`

## Success checklist (must be YES for every item)
- ✅ Vector store works with `tfidf` and `lsa` backends.
- ✅ Embedder config persists and reloads correctly.
- ✅ Tests cover both backends.
