# BackendScaffoldAgent

## Mission
Create and evolve the **FastAPI backend** for the SpecsGrader web migration.

This agent is used across multiple phases:
- **Phase 1:** FastAPI skeleton + static hosting
- **Phase 2:** upload-based file handling + Excel sheet selection parity
- **Phase 3:** model set APIs (list/load/last-used)
- **Phase 7:** export + download endpoints (backend plumbing)

## Inputs
- Repo working copy (unpacked `SpecsGraderV2.3.zip`)
- The current phase file from `planV3.0/` (the agent must follow the phase gates)

## Required outputs
Depending on the phase, create or update:
- `backend/main.py` (FastAPI app)
- `backend/settings.py` (configuration)
- `backend/api/routes_*.py` (routers)
- `backend/schemas/*.py` (Pydantic models)
- `backend/services/*.py` (storage/service utilities)
- Static hosting for `frontend/`

## Global constraints
- Keep existing test suite passing unless the phase explicitly changes behavior.
- Prefer **explicit imports** and avoid circular dependencies.
- Do not introduce a frontend bundler requirement; serve `frontend/` as static files.
- Use upload-based workflows (browser cannot safely provide local file paths).

## Implementation guidance by phase

### Phase 1: Skeleton + static hosting
1. Create `backend/` as a Python package with `backend/main.py` exporting `app`.
2. Add `/api/health` endpoint returning JSON `{ "status": "ok" }`.
3. Mount `frontend/` files under a stable URL (recommended: mount `/static` to `frontend/`).
4. Serve `frontend/index.html` at `/`.
5. Add dev-only CORS support restricted to localhost origins.
6. Ensure `/docs` loads.

### Phase 2: File upload + Excel parity
1. Add `POST /api/files/upload` (multipart form upload) and store files on disk under `artifacts/uploads/`.
2. Add `GET /api/files/{file_id}/summary` returning:
   - filename
   - file type (csv/xlsx)
   - row count (for chosen sheet)
   - chosen sheet name for Excel
   - column list (for chosen sheet)
3. Preserve the desktop behavior:
   - choose the **first sheet** whose header contains `Quote #` case-insensitively
   - else fall back to the first sheet
4. Ensure the backend uses the same logic as the desktop (`ui_actions._load_table_for_path`) or a faithful port.

### Phase 3: Model set APIs
1. Wrap existing model set operations:
   - list model sets
   - load model set
   - load last-used model set
2. Provide endpoints (names per phase plan):
   - `GET /api/model-sets`
   - `GET /api/model-sets/last`
   - `POST /api/model-sets/{name}/load`
3. Establish a server-side session store for loaded models (single-user local-first is acceptable).

### Phase 7: Export + download
1. Create export endpoint that accepts a `result_id` and export options.
2. Generate file under `artifacts/exports/` and return a download URL.
3. Add `GET /api/artifacts/{artifact_id}` to stream downloads safely.

## Required tests
Run the tests defined in the phase file. At minimum, do not regress:
- `python -m pytest -q`

## Acceptance criteria
- The phase tests pass.
- The phase success checklist is true.
- Static hosting and API routes work as described.

## Output format (agent response)
### Summary
### Files changed
### Tests run
### Notes
