# planV3.0 — SpecsGrader Web UI Migration Plan (HTML/CSS/ESM-JS + FastAPI)

## Objective
Replace the current **PySide6 desktop UI** in **SpecsGraderV2.3** with a **web UI** built with:
- **Frontend:** HTML + CSS + native ES Modules (ESM) JavaScript (no framework required)
- **Backend:** Python **FastAPI**

While preserving:
- existing business logic (`logic.py`, `rules_engine.py`, `similarity_engine.py`, `vector_db.py`, etc.)
- existing tests (`tests/test_train_and_infer_unittest.py`, `tests/test_vector_db_unittest.py`)
- existing workflow semantics (Import → Train → Classify → Review → Export)
- key behavior: **Excel sheet selection** chooses the first sheet containing a “Quote #” column, else falls back

## Repo baseline (SpecsGraderV2.3)
Key paths:
- Desktop UI: `ui.py`, `ui_main_window.py`, `ui_pages/*`, `ui_components/*`, `ui_actions.py`, `ui_state.py`, `ux_state.py`
- Core logic: `logic.py`, `train_classifier.py`, `classic_ml.py`, `embeddings.py`, `vector_db.py`, `similarity_engine.py`, `rules_engine.py`
- Model sets: `models/model_sets.json`, `model_set_manager.py`
- Tests: `tests/test_train_and_infer_unittest.py`, `tests/test_vector_db_unittest.py`
- Scripts: `scripts/*`

## Target architecture (added folders)
- `backend/` FastAPI app:
  - `backend/main.py`
  - `backend/api/routes_*.py`
  - `backend/services/*`
  - `backend/schemas/*`
  - `backend/settings.py`
- `frontend/` static site:
  - `frontend/index.html`
  - `frontend/styles.css`
  - `frontend/src/*.js` (ESM)
  - `frontend/src/pages/*.js`
  - `frontend/src/components/*.js`

## General implementation rules
1. **Do not break existing tests** during early phases; add new tests as needed.
2. Use **upload-based** file workflows (browser cannot reliably provide local file paths).
3. All long-running work (train/classify/export) must be **job-based** and non-blocking.
4. Large result sets must be **paged**; never return whole 8k×N tables in one JSON response.
5. Preserve (or improve) current debuggability: logs, progress, and deterministic outputs.

## Definition of “Done”
A user can:
1) choose/load a model set
2) upload training data and train
3) upload a doc and classify
4) review “Needs Review” rows and apply overrides
5) export results to a downloadable file

All phase gates must pass: **tests + success checklist**.

## Phases
- Phase 0: Baseline & guardrails
- Phase 1: FastAPI skeleton + static hosting
- Phase 2: File upload + Excel sheet selection parity
- Phase 3: Model set APIs (list/load/last-used)
- Phase 4: Job system + training job
- Phase 5: Classification job + results paging
- Phase 6: Review workflow (queue + inspector + overrides)
- Phase 7: Export + download
- Phase 8: Parity verification + cleanup + packaging

See individual phase files for full instructions, required tests, and success checklists.
