# DataFramesAndPagingAgent

## Mission
Implement efficient **server-side result storage and paging** for large DataFrames.

This agent is used in:
- Phase 2 (file summary and preview helpers)
- Phase 5 (classification results paging)
- Phase 6 (review queue and row inspector)

## Inputs
- Repo working copy
- Phase file from `planV3.0/`
- Backend skeleton + job manager (if already implemented)

## Required outputs
Create or update:
- `backend/services/dataframe_store.py` (or equivalent)
- `backend/schemas/results.py`
- `backend/api/routes_results.py`
- Any helper functions for filtering/searching

## Functional requirements
- Store a DataFrame server-side keyed by a `result_id`.
- Provide endpoints:
  - `GET /api/results/{result_id}/columns`
  - `GET /api/results/{result_id}/rows?offset=&limit=&filter=&search=`
  - `GET /api/results/{result_id}/row/{row_index}`
- Paging must be the default; never return full tables for big files.
- Filtering must support at least:
  - Needs Review true/false
  - Risk level equals a value (if present)
- Searching should scan a small set of safe columns (configurable) to avoid massive CPU.

## Parity requirements (desktop behavior)
- Classification outputs must include:
  - ALL original columns from the chosen sheet
  - PLUS classification columns

## Procedure
1. Design a `DataFrameStore` abstraction:
   - `put(df) -> result_id`
   - `get(result_id) -> df`
   - optional `delete(result_id)`
2. Implement stable row indexing:
   - ensure `row_index` maps deterministically to the stored DataFrame
3. Implement paging query:
   - apply filter/search
   - compute total rows
   - slice by offset/limit
4. Return JSON with:
   - `rows` (list of dicts)
   - `total` (int)
   - `offset`, `limit`
5. Add unit tests for:
   - paging correctness
   - filter behavior
   - row retrieval

## Acceptance criteria
- Fetching a page is fast and stable.
- Results endpoints never crash when a DataFrame contains mixed types.
- Phase tests and success checklist pass.

## Output format (agent response)
### Summary
### Files changed
### Tests run
### Notes
