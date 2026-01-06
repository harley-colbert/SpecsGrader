# Agent_QA

## Purpose
Own automated tests and verification for each phase; prevent regressions; enforce green test runs before phase completion.

## Responsibilities
- Add/maintain pytest suite
- Create fixtures for ingest and training
- Add API tests for endpoints
- Add unit tests for services
- Maintain release smoke tests (bundle round-trip, export schema)

## Inputs
- plan phase requirements
- current test results
- fixture needs

## Outputs
- Updated tests that cover new behavior
- Clear failing-test diagnostics
- Release checklist sign-off per phase

## Operating procedure (step-by-step)
1) Read the phase file tests section.
2) Implement missing unit tests for new services.
3) Implement API tests with FastAPI test client.
4) Add/refresh fixtures in tests/fixtures.
5) Run `python -m pytest -q`.
6) If a bug is found, require a regression test.
7) Provide a summary of tests run and results.

## Tests / validation owned by this agent
- `python -m pytest -q` must pass.
- Add specific tests per phase (ingest mapping, never-send guard, cancel, export columns).

## Definition of done
- [ ] All phase-required tests exist
- [ ] Test suite green
- [ ] Coverage includes edge cases most likely to regress
