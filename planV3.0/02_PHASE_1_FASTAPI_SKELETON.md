    # Phase 1 — FastAPI skeleton + static hosting

    ## Goal
    Create a FastAPI app that serves both `/api/*` endpoints and the static frontend at `/`.

    ## Prerequisites
    - Phase 0 completed.
- `pytest` is green.

    ## Implementation steps (must follow in order)
    1. **Agent:** BackendScaffoldAgent
   - Create `backend/` package with `backend/main.py` exposing `app`.
   - Add `/api/health` returning `{ status: 'ok' }`.
   - Mount static files from `frontend/` under `/static`.
   - Serve `frontend/index.html` at `/`.
   - Add local-dev CORS support (localhost only).

2. **Agent:** FrontendESMAgent
   - Create `frontend/index.html`, `frontend/styles.css`, `frontend/src/main.js`.
   - Render a shell layout (sidebar + topbar + main area) with placeholder content.
   - Use only ESM imports (relative paths).

3. **Agent:** APIContractAgent
   - Confirm OpenAPI schema is generated and `/docs` works.

    ## Files to create/change
    - **Create:** `backend/main.py`, `backend/settings.py`
- **Create:** `backend/api/__init__.py`
- **Create:** `frontend/index.html`, `frontend/styles.css`, `frontend/src/main.js`
- **Change:** `requirements.txt` (add `fastapi`, `uvicorn[standard]` if missing)

    ## Tests (must run and pass)
    1. `python -m pytest -q`
2. Start server: `uvicorn backend.main:app --reload`
3. Verify in browser:
   - `http://127.0.0.1:8000/` loads UI shell
   - `http://127.0.0.1:8000/api/health` returns OK
   - `http://127.0.0.1:8000/docs` loads OpenAPI UI

    ## Success checklist (must be true before moving on)
    - ✅ `pytest` passes
- ✅ `/api/health` returns 200
- ✅ `/` serves index.html and loads `frontend/src/main.js` successfully (no console errors)
- ✅ `/docs` loads and shows `/api/health`

    ## Notes for agents (assumed available)
    - BackendScaffoldAgent must keep imports explicit and avoid circular dependencies.
- FrontendESMAgent should implement a simple router skeleton (hash router is fine) but can remain placeholder in Phase 1.
