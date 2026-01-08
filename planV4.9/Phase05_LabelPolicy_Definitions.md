# Phase 05 — Label Policy (Definitions + Consistency)

## Goal
Add a single source-of-truth Label Policy defining each risk level and department; expose it in UI and bundle it with ModelSets.

## Primary agents
- BackendAgent
- FrontendAgent
- TestAgent

## Scope
- Backend: add label policy file + endpoints; ensure ModelSets carry it.
- Frontend: display label definitions in Train/Classify panes.

## Implementation steps (do in order)
- BackendAgent:
-   1) Add `backend/app/label_policy.py` (or `backend/app/policy/label_policy.py`) that loads a default policy dict.
-   2) Allow override per ModelSet version (store under `workspace/modelsets/<id>/label_policy.json`).
-   3) Add API endpoints:
-      - `GET /api/label-policy` (effective policy)
-      - optionally `PUT /api/modelsets/{id}/label-policy` (if you want edits now; otherwise read-only).
-   4) Ensure ModelSet export (`.sgm`) contains the label policy in its bundle meta.
- 
- FrontendAgent:
-   1) Add a “Label definitions” block in Train pane (near Dataset Health) and in Classify pane.
-   2) Keep it compact: tooltip or collapsible panel.

## Testing work to CREATE/UPDATE in this phase
- Add `tests/test_label_policy.py`:
-   - `GET /api/label-policy` returns all expected keys
-   - ModelSet export/import preserves policy (extend `tests/test_sgm_io.py`).

## Tests that MUST pass (gate)
- `python -m pytest -q`

## Success checklist (must be YES for every item)
- ✅ Label definitions visible in UI.
- ✅ Label policy included in exports/imports.
- ✅ Tests confirm policy endpoint and persistence.
