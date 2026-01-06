    # Phase 5 — Classification job + results paging

    ## Goal
    Run classification asynchronously and expose results via paged endpoints suitable for large tables.

    ## Prerequisites
    - Phase 4 completed (job system exists).

    ## Implementation steps (must follow in order)
    1. **Agent:** BackendScaffoldAgent + DataFramesAndPagingAgent
   - Implement `POST /api/classify` (starts job) with inputs:
     - `model_set_name`
     - `doc_file_id`
     - similarity options (toggle/threshold)
   - Classification job loads doc DataFrame and runs your existing multipass classify pipeline.
   - Store output DataFrame as `result_id`.

2. Implement results APIs:
   - `GET /api/results/{result_id}/columns`
   - `GET /api/results/{result_id}/rows?offset&limit&filter&search`
   - `GET /api/results/{result_id}/row/{row_index}`
   - Filters should include at minimum:
     - `needs_review=true`
     - `risk_level=<value>`

3. **Agent:** FrontendESMAgent
   - Build Classify page:
     - upload doc file
     - run classify job
     - table view with paging controls
     - row click opens a basic inspector (can be enhanced in Phase 6)


    ## Files to create/change
    - **Create:** `backend/api/routes_classify.py`, `backend/api/routes_results.py`
- **Create:** `backend/services/result_store.py`
- **Create:** `backend/schemas/results.py`
- **Create:** `frontend/src/pages/classify.js`
- **Create/Change:** `frontend/src/components/table.js`, `inspector.js`

    ## Tests (must run and pass)
    1. `python -m pytest -q`
2. Add `tests/test_api_classify_results_paging_unittest.py`:
   - start classify job → wait complete
   - fetch columns → non-empty
   - fetch rows page 1 → returns <= limit
   - fetch row detail → contains expected output fields
3. Manual:
   - In UI, classify a file and scroll pages without freezing

    ## Success checklist (must be true before moving on)
    - ✅ Classification runs as a job and returns `result_id`
- ✅ Results paging works and is fast for large tables
- ✅ UI renders results without loading the entire dataset at once

    ## Notes for agents (assumed available)
    - DataFramesAndPagingAgent should implement server-side filtering/search to avoid transferring everything.
- JobsAndLoggingAgent should ensure classify job emits progress markers (rows processed, stage name).
