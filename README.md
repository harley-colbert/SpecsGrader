# SpecsGrader

This repository follows the SpecsGrader plan V4.0. Run the application from the
project root; it serves the backend and frontend on the same port and opens a
PyWebView window to the local URL. The UI currently exposes a three-pane shell
(Train, Classify, Results) driven by backend state.

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
  `/api/train/status`, `/api/train/cancel`, and `/api/train/metrics`.
- Vector: local TF-IDF embeddings and ANN queries via `backend/app/services/vector_service.py`
  with `/api/vector/build` and `/api/vector/test`.
- LLM: OpenRouter-backed service with never-send mode in
  `backend/app/services/llm_service.py` and settings via `/api/settings`.
- Tests: pytest suite under `tests/`.

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
