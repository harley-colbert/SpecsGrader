# AGENT_02_VECTOR_DB_ENGINEER — Implement ChromaDB Wrapper (vector_db.py)

## Role
Implement the v2.0 vector database abstraction layer:
- persistent on disk
- safe on import
- minimal API used by training and inference

You must follow Phase 02 requirements exactly.

## Primary Phases
- Phase 02 (required)
- Supports Phase 03–05 (schema + training + inference integration)

## Inputs
- SpecGrader repo
- Phase instructions: `Phase_02_VectorDB_Module.md`
- `chromadb` dependency already added (Phase 01)

## Outputs
- New file: `vector_db.py`
- Optional helper script for smoke testing:
  - `scripts/smoke_vector_db.py` (allowed; Phase 02 mentions this as acceptable)

## Required API (must implement exactly)
Create a `VectorDB` class with:

1) `@staticmethod open(persist_dir: str, collection_name: str) -> VectorDB`
2) `upsert(ids: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None`
3) `query(embedding: list[float], top_k: int) -> list[dict]`
4) `count() -> int`
5) `reset_collection(confirm: bool) -> None` (optional, but recommended for scripts)

## Mandatory Behaviors
- Importing `vector_db.py` must not crash even if folders are missing.
- `open()` must create directories using `os.makedirs(..., exist_ok=True)`.
- `query()` must return a list of dicts where each dict includes:
  - `id`
  - `text`
  - `risk_level`
  - `review_dept`
  - `similarity` (float in [0,1], higher is closer)
  - `metadata` (dict)

## Similarity Metric Notes
Chroma typically returns `distances`. You must:
- document which metric you use (cosine preferred)
- convert distance -> similarity consistently:
  - if cosine distance in [0,2], similarity = 1 - distance (common assumption in this project plan)
  - clamp similarity into [0,1] to avoid confusing callers

If the returned distance semantics differ, standardize similarity in your wrapper so callers always see [0,1].

## Implementation Details (recommended)
- Use:
  - `chromadb.PersistentClient(path=persist_dir)`
  - `get_or_create_collection(name=collection_name, metadata={...})`
- Store metadata only; store text as metadata too (plan expects `text` in results)
- Keep verbose debug logs consistent with existing code style (`[DEBUG] ...`)

## Must-Pass Tests (no errors)
1) `python -m compileall .`

2) Minimal open test:
- `python -c "from vector_db import VectorDB; db=VectorDB.open('models/vector_db/_smoke/chroma','_smoke'); print('count', db.count())"`

3) Upsert + query smoke:
- Create 2 items with small embeddings (dimension 3 is fine for the smoke test).
- Query with an embedding close to one of them.
- Verify returned results include the required fields.

If you implement `scripts/smoke_vector_db.py`, it must run without errors:
- `python scripts/smoke_vector_db.py`

## Success Checklist
- [ ] `vector_db.py` exists and matches the required API
- [ ] `open()` creates folders and can open an empty collection
- [ ] `upsert()` inserts items without error
- [ ] `query()` returns required schema with numeric similarity
- [ ] No crashes when DB is empty

## Guardrails
- Do not modify training or inference in this phase (only the wrapper + optional smoke script).
- Do not require a running server; must be embedded/persistent client.
