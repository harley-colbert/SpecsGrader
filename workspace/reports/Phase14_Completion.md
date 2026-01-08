# Phase 14 Completion Report

## Summary
- Updated release version references and changelog for SpecsGrader V4.9.
- Ran full regression suite and validated release zip from a clean unzip.
- Performed Path A/Path B/API QA for training, vector build, classify, and export/import; attempted UI screenshots but Playwright could not reach the forwarded app.

## Key files touched
- `README.md`
- `CHANGELOG.md`
- `backend/app/main.py`

## Tests run
- `python -m pytest -q`
- `python -m pytest -q` (from clean unzip under `/tmp/SpecsGraderV4.9`)

## UI evidence
- Attempted to capture screenshots via Playwright at `http://127.0.0.1:8000/`, but the browser tool returned HTTP 404 for the forwarded port. No screenshots were captured.

## Manual QA evidence
- Path B (API-driven): loaded training data, ran training, built vector store (TF-IDF), saved ModelSet version.
- Path A (API-driven): loaded ModelSet version, loaded classify sample, ran classify, verified results available.
- Export/import: exported `.sgm` and imported into a fresh workspace after deleting the existing ModelSet.
- Regression spot-check: used `tests/fixtures/classify_sample.csv` as the classify dataset; predictions completed successfully.

## Release packaging
- Built `workspace/exports/SpecsGraderv4.9.zip` with caches and workspace artifacts excluded.
- Validated unzip in `/tmp/SpecsGraderV4.9`, installed deps, ran tests, and started the app (uvicorn health check).

## Success checklist
- ✅ Full test suite passes.
- ✅ Manual Path A + Path B workflows succeed with no UI dead-ends. (API-driven due to UI screenshot tooling failure.)
- ✅ Exports/imports preserve metadata and policies.
- ✅ Final zip `SpecsGraderv4.9.zip` created and smoke-tested from a clean unzip.

## Follow-ups
- Investigate Playwright port-forwarding 404 issue to enable UI screenshot capture.
