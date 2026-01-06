    # Phase 3 — Model set APIs (list/load/last-used)

    ## Goal
    Expose model set management via API so the web UI can load and use existing model sets.

    ## Prerequisites
    - Phase 2 completed.

    ## Implementation steps (must follow in order)
    1. **Agent:** APIContractAgent + BackendScaffoldAgent
   - Implement:
     - `GET /api/model-sets`
     - `GET /api/model-sets/last`
     - `POST /api/model-sets/{name}/load`
   - Wrap existing `logic.list_model_sets()`, `logic.get_last_used_model_set()`, and model loading.

2. Implement a simple server-side session store:
   - Keep loaded models cached in memory (single-user local app assumption).
   - Provide a small metadata payload on load (e.g., which artifacts were loaded).

3. **Agent:** FrontendESMAgent
   - Add Model Set dropdown to sidebar/topbar.
   - On selection, call `/load` and show a “loaded” badge.
   - Default to last-used when available.

    ## Files to create/change
    - **Create:** `backend/api/routes_model_sets.py`
- **Create:** `backend/services/session_store.py`
- **Create:** `backend/schemas/model_sets.py`
- **Change:** `backend/main.py` add router
- **Change:** `frontend/src/components/sidebar.js` and/or `topbar.js`

    ## Tests (must run and pass)
    1. `python -m pytest -q`
2. Add `tests/test_api_model_sets_unittest.py`:
   - list endpoint returns array
   - load endpoint returns 200 for an existing model set
3. Manual:
   - Start server → open UI → confirm model sets populate

    ## Success checklist (must be true before moving on)
    - ✅ Model set list loads without errors
- ✅ Load model set caches models for subsequent operations
- ✅ Frontend indicates loaded state and uses last-used when present

    ## Notes for agents (assumed available)
    - RepoAuditAgent should confirm model set files are resolved consistently across OS paths.
- PackagingAgent should ensure requirements and imports remain compatible.
