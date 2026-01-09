# TESTING_CONVENTIONS (planV4.10)

## Testing conventions (applies to all phases)

- Prefer **pytest** for backend tests.
- Prefer small deterministic **XLSX fixtures** committed under a `tests/fixtures/` folder.
- All tests must run offline (no network calls required).
- If the repo has both backend and frontend tests, run both when UI changes occur.

If you do not currently have a unified test command, create:
- `scripts/test_backend.sh` (or `.bat` for Windows) that runs pytest
- `scripts/test_all.sh` that runs all test suites (backend + frontend if present)

Each phase file includes a “Tests that must pass” section; treat it as a hard gate.
