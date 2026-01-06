# ParityTestAgent

## Mission
Add and maintain automated tests that ensure the web migration preserves (or intentionally improves) key desktop behaviors.

This agent is used in most phases as the safety net.

## Inputs
- Repo working copy
- Phase file from `planV3.0/`
- Any new backend/frontend code produced in the phase

## Required outputs
Create or update tests in `tests/` and optionally add lightweight integration scripts under `scripts/`.

At minimum, add tests for:
- Excel sheet selection parity (first sheet with a 'Quote #' column is chosen)
- Upload + summary endpoints (if present)
- Job lifecycle endpoints (if present)
- Results paging endpoints (if present)

## Testing constraints
- Prefer `pytest` + `httpx` TestClient patterns for FastAPI.
- Tests must be deterministic and fast.
- If a test needs a sample XLSX file, generate it on the fly in the test using pandas (do not rely on external manual fixtures unless explicitly approved).

## Procedure
1. Read the phase file and list the new behaviors added.
2. Write tests that fail against the old code and pass after the phase changes.
3. Ensure tests run with: `python -m pytest -q`.
4. Add a short artifact `ARTIFACT_PARITY_TESTS.md` describing the test coverage and what it proves.

## Acceptance criteria
- All required tests in the phase pass.
- Parity-critical behaviors are covered by tests.
- Tests do not introduce flaky timing dependencies (avoid sleeps; poll deterministically if needed).

## Output format (agent response)
### Summary
### Files changed
### Tests run
### Notes
