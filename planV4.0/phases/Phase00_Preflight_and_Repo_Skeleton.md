# Phase 00 — Preflight and repo skeleton

## Goal
Create a clean, reproducible repository skeleton that runs via:

```bash
python run.py
```

…and starts backend + frontend on a single local port, opening a PyWebView window to that URL.

This phase establishes:
- root-level `run.py`
- root-level `requirements.txt`
- initial Python package layout
- minimal FastAPI server that serves a static frontend placeholder
- minimal PyWebView bootstrap
- pytest wiring

## Agents
- Agent_RepoAuditor (lead)
- Agent_BackendEngineer
- Agent_QA

## Repo layout (required)
At project root:

- `run.py`
- `requirements.txt`
- `backend/`
  - `app/`
    - `__init__.py`
    - `main.py` (FastAPI app factory)
    - `settings.py`
    - `state.py`
- `frontend/`
  - `index.html`
  - `styles.css`
  - `src/`
    - `main.js`
- `tests/`
  - `test_smoke.py`
- `README.md` (repo readme; separate from this plan readme)

**Important:** The backend must serve the built/static frontend directly from `frontend/` (no Node build required).
ESM imports must work via static file serving.

## Implementation instructions
1) Create `requirements.txt` at root with minimum dependencies:
   - pywebview
   - fastapi
   - uvicorn
   - pydantic (or pydantic-settings if used)
   - pandas
   - openpyxl
   - numpy
   - scikit-learn
   - joblib
   - requests (for OpenRouter later)
   - pytest

2) Implement FastAPI app in `backend/app/main.py`:
   - `create_app()` returns a FastAPI app
   - Mount static files:
     - `/` serves `frontend/index.html`
     - `/styles.css` serves the stylesheet
     - `/src/*` serves ESM modules
   - Add a simple `/api/health` returning JSON `{ "ok": true }`

3) Implement `run.py` at root:
   - Choose a port (default `7860` or dynamic free port).
   - Start uvicorn server in a background thread (or separate process) so PyWebView can open.
   - Create a PyWebView window to `http://127.0.0.1:{PORT}/`
   - Ensure clean shutdown when the window closes.
   - Prefer single port for both UI and API.
   - Do NOT start a separate frontend dev server.

4) Add `backend/app/settings.py`:
   - Contains PORT default and “never send externally” flag placeholder.

5) Add initial `backend/app/state.py` with `AppState` dataclass:
   - just placeholders for now (no business logic)

6) Add minimal `frontend/index.html` + `frontend/src/main.js`:
   - render a placeholder shell and “Health check” button that calls `/api/health` and displays the result.

7) Add pytest skeleton:
   - `tests/test_smoke.py`:
     - import `create_app()`
     - use FastAPI test client
     - assert `/api/health` returns ok

## Testing (must run and pass)
From project root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python -m pytest -q
python run.py
```

Manual:
- App window opens and renders placeholder UI
- Health check button shows `{ "ok": true }`

## Success checklist
- [ ] `python run.py` works from project root
- [ ] Backend and frontend served on same port
- [ ] PyWebView opens `http://127.0.0.1:{PORT}/`
- [ ] `/api/health` passes via pytest
- [ ] No Node tooling required to run
- [ ] Static ESM module imports load without 404s
