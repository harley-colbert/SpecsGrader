# Phase 01 — App shell and pane contracts

## Goal
Implement the UI shell + three sequestered panes with strict contracts:
- Train pane
- Classify pane
- Results pane

Panes do not directly reference each other. They request shared data via main app APIs.

## Agents
- Agent_FrontendEngineer (lead)
- Agent_BackendEngineer
- Agent_RepoAuditor
- Agent_QA

## Backend contracts (JS bridge)
Implement FastAPI endpoints (preferred) or PyWebView JS bridge methods that front-end uses.
Recommendation: standardize on REST endpoints under `/api/*`.

Required endpoints (minimum for this phase):
- `GET /api/state`
  - returns current pane, loaded dataset summaries, active bundle id (if any)
- `POST /api/ui/set_active_pane`
  - body: `{ "pane": "train"|"classify"|"results" }`
- `GET /api/health` (already exists)

State rules:
- Main app owns state
- Panes only render what `/api/state` returns + their local ephemeral UI state

## Frontend structure (required)
In `frontend/src/`:
- `main.js` (bootstrap + router)
- `api/client.js` (fetch wrapper)
- `ui/shell.js` (left nav + right content host)
- `panes/trainPane.js`
- `panes/classifyPane.js`
- `panes/resultsPane.js`
- `state/store.js` (client-side ephemeral state only; does NOT become source of truth)

Pane interface:
- `mount(containerEl, deps)`
- `unmount()`
- `refresh(state)`

## Implementation instructions
1) Build shell layout:
   - Left nav (Train / Classify / Results)
   - Right pane region
   - Light theme default

2) Implement routing:
   - Clicking nav triggers:
     - `POST /api/ui/set_active_pane`
     - then `GET /api/state` and re-render current pane

3) Implement placeholder content for each pane:
   - Train: “Load training file”, “Train model”, “Save bundle” (disabled placeholders)
   - Classify: “Load classify file”, toggles (disabled placeholders), “Run job” (disabled)
   - Results: empty table placeholder

4) Implement `GET /api/state` in backend:
   - return at least:
     - `active_pane`
     - `data_loaded`: { train: false, classify: false }
     - `active_bundle_id`: null

5) Keep pane isolation:
   - No direct imports between panes except shared `api/client.js` and shared UI components

## Testing (must run and pass)
Automated:
```bash
python -m pytest -q
```

Add tests:
- `tests/test_state_api.py`:
  - `GET /api/state` returns required fields
  - `POST /api/ui/set_active_pane` updates `active_pane`

Manual:
- `python run.py`
- Click each nav item; pane changes and does not error
- Refresh app window; app loads default pane (Train)

## Success checklist
- [ ] 3 panes exist as separate modules and mount/unmount cleanly
- [ ] Panes do not call each other directly (only API)
- [ ] `/api/state` is the single source of truth for current view
- [ ] Navigation works reliably without page reload
- [ ] All pytest tests pass
