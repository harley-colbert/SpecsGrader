    # Phase 7 — Export + download

    ## Goal
    Enable exporting results (with review overrides) to a downloadable artifact (CSV/XLSX).

    ## Prerequisites
    - Phase 6 completed.

    ## Implementation steps (must follow in order)
    1. **Agent:** BackendScaffoldAgent
   - Implement `POST /api/export`:
     - inputs: `result_id`, preset/options
     - writes export file under `artifacts/exports/{export_id}.xlsx` (or .csv)
     - returns `{ export_id, filename, download_url }`
   - Implement `GET /api/artifacts/{export_id}` to stream download.

2. **Agent:** FrontendESMAgent
   - Build Export page:
     - preset dropdown
     - export button
     - show download link on success

3. **Agent:** APIContractAgent
   - Document presets contract (names, option schema) so it remains stable.

    ## Files to create/change
    - **Create:** `backend/api/routes_export.py`
- **Create:** `backend/services/export_service.py`
- **Create:** `frontend/src/pages/export.js`

    ## Tests (must run and pass)
    1. `python -m pytest -q`
2. Add `tests/test_api_export_download_unittest.py`:
   - create result
   - export
   - download and confirm non-empty bytes
3. Manual:
   - Export from UI and open file locally

    ## Success checklist (must be true before moving on)
    - ✅ Export endpoint produces file and returns a working download URL
- ✅ Download works in browser
- ✅ Export reflects review overrides

    ## Notes for agents (assumed available)
    - PackagingAgent should ensure export dependencies are in requirements (e.g., openpyxl if XLSX).
- ParityTestAgent should verify export columns match the current desktop export presets.
