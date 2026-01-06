    # Phase 6 — Review workflow (queue + inspector + overrides)

    ## Goal
    Replicate the desktop Review step: show Needs Review queue, inspector panel, and allow manual overrides persisted server-side.

    ## Prerequisites
    - Phase 5 completed (results exist and are pageable).

    ## Implementation steps (must follow in order)
    1. **Agent:** DataFramesAndPagingAgent
   - Implement review endpoints:
     - `GET /api/results/{result_id}/review-queue?offset&limit`
     - `POST /api/results/{result_id}/row/{row_index}/label` (override fields)
   - Persist overrides in the stored result DataFrame.
   - Track a `dirty` flag indicating result has manual edits.

2. **Agent:** FrontendESMAgent
   - Build Review page:
     - default queue filter to Needs Review
     - click row → inspector panel with details
     - controls to set label/risk overrides
     - Save + Next buttons


    ## Files to create/change
    - **Change:** `backend/api/routes_results.py` (add review routes) or add `routes_review.py`
- **Change:** `backend/schemas/results.py`
- **Change:** `frontend/src/pages/review.js`
- **Change:** `frontend/src/components/inspector.js`

    ## Tests (must run and pass)
    1. `python -m pytest -q`
2. Add `tests/test_api_review_overrides_unittest.py`:
   - create result
   - apply override to a row
   - refetch row and assert override is present
3. Manual:
   - Review queue decreases as rows are fixed

    ## Success checklist (must be true before moving on)
    - ✅ Overrides persist and reflect immediately in queue and row detail
- ✅ Review queue endpoint is pageable
- ✅ UI supports fast review (save + next) without full reload

    ## Notes for agents (assumed available)
    - ParityTestAgent should compare Review semantics with desktop (field names + meaning) on a known file.
- APIContractAgent should freeze request/response shapes now (avoid churn).
