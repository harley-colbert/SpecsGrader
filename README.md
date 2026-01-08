# SpecsGrader

This repository follows the SpecsGrader plan V4.6. Run the application from the
project root; it serves the backend and frontend on the same port and opens a
PyWebView window to the local URL. The UI currently exposes a three-pane shell
(Train, Classify, Results) driven by backend state, with the v4.6 Train pane UX
upgrade (Quick Start mode selector, Path A load & classify flow, and Path B
build/update stepper).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python run.py
```

## Development
- Backend: FastAPI app defined in `backend/app/main.py`.
- Frontend: Static assets under `frontend/` with ESM modules in `frontend/src`.
- Data ingest: CSV/XLSX handled in `backend/app/services/ingest_service.py` with
  `/api/data/load` and `/api/data/preview` endpoints.
- Rules: keyword rules handled in `backend/app/services/rule_service.py` with
  `/api/rules/get`, `/api/rules/set`, and `/api/rules/test`.
- Training: TF-IDF + calibrated logistic regression implemented in
  `backend/app/services/training_service.py` with `/api/train/start`,
  `/api/train/status`, `/api/train/cancel`, `/api/train/metrics`,
  `/api/train/sanity`, and `/api/train/evaluate`.
- Vector: local TF-IDF embeddings and ANN queries via `backend/app/services/vector_service.py`
  with `/api/vector/build`, `/api/vector/test`, and `/api/vector/status`.
- ModelSets: versioned, on-disk bundles of (trained model artifacts + vector store + rules)
  with import/export to a single `.sgm` archive via `backend/app/services/modelset_service.py`.
  API endpoints: `/api/modelsets`, `/api/modelsets/{id}/versions`, `/api/modelsets/{id}/load`,
  `/api/modelsets/{id}/export`, `/api/modelsets/import`, plus `PATCH /api/modelsets/{id}` and
  `DELETE /api/modelsets/{id}/versions/{version_id}` for metadata updates and guarded deletes.
- LLM: OpenRouter-backed service with never-send mode in
  `backend/app/services/llm_service.py` and settings via `/api/settings`.
- Tests: pytest suite under `tests/`.
- Modes:
  - Sanity (model-only, quick training check)
  - Evaluate (holdout metrics per method)
  - Production (decision ladder with trace output)

Run tests with:

```bash
python -m pytest -q
```

## Environment and security
- Python: 3.12
- Optional: set `OPENROUTER_API_KEY` in your environment to enable LLM calls.
- To block all external LLM calls, set `never_send_externally` via `/api/settings`
  (UI toggle in Classify pane). Server-side checks enforce this flag.

## File format rules
- Inputs: CSV or XLSX.
- XLSX sheets: first sheet whose name contains “Standards Risk Matrix”.
- Data starts on row 5 (1-indexed). Column E = risk text, Column F = risk level
  (train), Column G = department (train).

## Export
`/api/export/csv` should emit a CSV preserving original columns and appending:
id, risk_text, pred_level, pred_dept, confidence, methods_used, model_bundle_id,
user_override_level, user_override_dept.

## ModelSets and .sgm
- A ModelSet is a **family** (stable ID, name, description, tags) with **immutable versions**.
- Saving a snapshot creates a new version (no in-place updates).
- Loading a version updates the active modelset/version in app state.
- `.sgm` exports include `manifest.json` and `checksums.sha256`; imports validate checksums
  and block unsafe zip contents or version collisions.
