# Phase 02 — Add Vector DB Abstraction Layer (ChromaDB)

## Agents
- Run: **AGENT_02_VECTOR_DB_ENGINEER**
- Optional assist: **AGENT_00_REPO_AUDITOR**

## Goal
Introduce a vector database module that is:
- persistent on disk
- deterministic in behavior
- safe to import even if no DB exists yet
- provides a minimal API: open, upsert, query, stats/health

## New File to Create
- `vector_db.py`

## Design Requirements
### Storage layout (per model set)
- Root: `models/vector_db/`
- Per set: `models/vector_db/<safe_model_set_name>/`
- Chroma persistence directory: `models/vector_db/<safe_model_set_name>/chroma/`
- Collection name: `<safe_model_set_name>` (sanitized)

### Metadata schema per stored row
Store at minimum:
- `text` (string)  — training spec statement
- `risk_level` (string)
- `review_dept` (string)
Recommended:
- `source_file` (string, optional)
- `row_index` (int, optional)
- `model_set` (string)

### Query result schema returned to caller
For each match:
- `id`
- `text`
- `risk_level`
- `review_dept`
- `similarity` (float in [0,1] where higher is closer)
- `metadata` (dict)

## Implementation Steps
1. Implement `VectorDB` class in `vector_db.py`
   Required methods:
   - `@staticmethod open(persist_dir: str, collection_name: str) -> VectorDB`
   - `upsert(ids: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None`
   - `query(embedding: list[float], top_k: int) -> list[dict]`
   - `count() -> int`
   - `reset_collection(confirm: bool) -> None` (optional safety)

2. ChromaDB integration notes
   - Use `chromadb.PersistentClient(path=persist_dir)`
   - Use `get_or_create_collection(name=collection_name, metadata={...})`
   - For similarity: Chroma returns distances; convert to similarity if needed:
     - If distance is cosine distance, similarity = 1 - distance
     - Ensure the plan documents what distance metric is used.

3. Safety requirements
   - Importing `vector_db.py` must not crash if folders are missing.
   - Any filesystem directory required should be created with `os.makedirs(..., exist_ok=True)`.

## Must-Pass Tests (no errors)
Create a new script `scripts/smoke_vector_db.py` OR a unit test (Phase 09 will add formal tests).
Run:
- `python -m compileall .`
- `python -c "from vector_db import VectorDB; db=VectorDB.open('models/vector_db/_smoke/chroma','_smoke'); print('count', db.count())"`

Then do a real insert/query smoke:
- Create a dummy embedding vector with small dimension (e.g., 3 floats) and insert 2 rows, query 1 row.
- The query must return a list of results with `text`, `risk_level`, `review_dept`, and a numeric similarity.

## Success Checklist
- [ ] `vector_db.py` exists with the required API
- [ ] Opening a new DB directory works (it creates folders)
- [ ] Upsert + query works in a smoke run
- [ ] No crashes on import / open / query when the DB is empty
