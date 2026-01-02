# AGENT_06_TEST_ENGINEER — Unit Tests + Smoke Tests + Cleanup

## Role
Add repeatable automated tests so SpecGrader v2.0 is verifiable:
- unit tests for vector DB wrapper
- train + infer smoke tests using minimal datasets
- tests must run without GUI and must clean up temporary DB directories

## Primary Phase
- Phase 09 (required)
- Optional support in Phase 02/04/05 for early smoke testing

## Inputs
- SpecGrader repo
- Vector DB wrapper exists
- Training + inference v2.0 code exists
- Phase doc: `Phase_09_Testing_and_Release.md`

## Outputs (files to add)
- `tests/test_vector_db_unittest.py`
- `tests/test_train_and_infer_unittest.py`
- `tests/data/train_min.csv`
- `tests/data/infer_min.csv`
- Optional: `scripts/smoke_all.py`

## Test Constraints (mandatory)
- Use stdlib unittest:
  - tests run with `python -m unittest -v`
- Must not require a GUI to pass
- Must clean up temp vector DB dirs they create
- Must not depend on an already-trained model set in the user’s environment:
  - tests should create a temporary model set name like `_test_v2_smoke`
  - and clean up artifacts where feasible (or isolate under `models/vector_db/_test_*`)

## Recommended Test Design
### test_vector_db_unittest.py
- create a temp persist dir under `models/vector_db/_unittest_tmp/chroma`
- open db, upsert 2 vectors, query, assert schema:
  - list returned
  - each item has `text`, `risk_level`, `review_dept`, numeric `similarity`

### test_train_and_infer_unittest.py
- ensure `tests/data/train_min.csv` and `tests/data/infer_min.csv` exist
- call `logic.train_all_models(..., model_set_name='_test_v2_smoke')`
- load that model set via `logic.load_model_set`
- load all models via `logic.load_all_models(files_dict)`
- run `logic.multipass_classify(..., sim_checkbox=True, top_k=3, similarity_threshold=0.1)`
- assert output DataFrame has:
  - `Final Risk Level`
  - `Final Review Dept`
  - `Needs Review`
  - similarity/evidence columns
- clean up temp vector db directories created for `_test_v2_smoke` if practical

## Must-Pass Tests (no errors)
From repo root:
- `pip install -r requirements.txt`
- `python -m compileall .`
- `python -m unittest -v`

## Success Checklist
- [ ] All unit tests pass locally with `python -m unittest -v`
- [ ] Tests are deterministic and do not require GUI
- [ ] Temp vector DB dirs are cleaned up (or isolated so they don’t pollute real data)
