    # Phase 2 — File upload + Excel sheet selection parity

    ## Goal
    Support browser uploads for CSV/XLSX and preserve the existing “Quote # sheet selection” behavior.

    ## Prerequisites
    - Phase 1 completed and server runs.

    ## Implementation steps (must follow in order)
    1. **Agent:** BackendScaffoldAgent + DataFramesAndPagingAgent
   - Implement upload endpoint `POST /api/files/upload` (multipart/form-data).
   - Store uploads under `artifacts/uploads/{file_id}/original.ext`.
   - Return `{ file_id, filename, content_type }`.

2. Implement file inspection endpoints:
   - `GET /api/files/{file_id}/summary`
   - `GET /api/files/{file_id}/sheets` (XLSX only)
   - `GET /api/files/{file_id}/preview?sheet=...&limit=...` (optional but recommended)

3. **Critical parity requirement:**
   - For XLSX, choose the **first worksheet** where any column header contains `Quote #` (case-insensitive).
   - If none found, default to the first sheet.
   - Return the chosen sheet name in `summary`.

4. **Agent:** FrontendESMAgent
   - Implement Import page UI:
     - upload control
     - summary card showing chosen sheet + row count + columns
     - sheets dropdown shown only for XLSX (optional override)
   - Store `file_id` in frontend state.

    ## Files to create/change
    - **Create:** `backend/api/routes_files.py`
- **Create:** `backend/services/file_store.py`, `backend/services/dataframe_store.py`
- **Create:** `backend/schemas/files.py`
- **Change:** `backend/main.py` to include file routes
- **Create:** `frontend/src/pages/import.js`
- **Change:** `frontend/src/main.js` router to include Import page

    ## Tests (must run and pass)
    1. `python -m pytest -q`
2. Add and run new tests:
   - `tests/test_api_files_unittest.py` (FastAPI TestClient):
     - upload CSV → summary has correct row count
     - upload XLSX with multiple sheets where only one has `Quote #` → chosen sheet matches
3. Manual browser test:
   - Upload XLSX, confirm UI displays chosen sheet and preview.

    ## Success checklist (must be true before moving on)
    - ✅ All tests pass (including new API file tests)
- ✅ XLSX “Quote #” sheet selection matches the desktop behavior
- ✅ Upload + summary works end-to-end from browser

    ## Notes for agents (assumed available)
    - APIContractAgent should ensure response schemas are stable and versioned if needed.
- DataFramesAndPagingAgent should store loaded DataFrames keyed by `file_id` (in-memory is OK for now).
