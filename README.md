# SpecsGrader

This repository follows the SpecsGrader plan V4.0. Run the application from the
project root; it serves the backend and frontend on the same port and opens a
PyWebView window to the local URL.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
python run.py
```

## Development
- Backend: FastAPI app defined in `backend/app/main.py`.
- Frontend: Static assets under `frontend/` with ESM modules in `frontend/src`.
- Tests: pytest suite under `tests/`.

Run tests with:

```bash
python -m pytest -q
```
