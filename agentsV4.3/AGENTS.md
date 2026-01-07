# Agent definitions (v4.3)

## OrchestratorAgent
**Mission:** Deliver the phase outcomes exactly as specified in the phase file, with minimal scope creep.

**Responsibilities**
- Read the phase plan file before any edits.
- Delegate to other agents if applicable.
- Ensure tests are run and pass.
- Ensure the success checklist is satisfied (explicitly checked off).
- Ensure no regressions in Train/Vector/Rules/Grade flows.

**Outputs**
- A clean set of file changes that satisfy the phase objective
- Test results (command + outcome)
- Notes on any deviations from plan (should be rare)

---

## FrontendAgent
**Mission:** Implement Train pane UI changes and client-side wiring.

**Responsibilities**
- Modify `frontend/src/panes/trainPane.js` layout and state sync.
- Modify `frontend/src/api/client.js` for new/changed endpoints.
- Modify `frontend/styles.css` for compact UI or targeted styling.
- Ensure no console error spam; handle "not available" states gracefully.
- Ensure export save picker works when supported (fallback to download).

**Outputs**
- Updated frontend behavior meeting acceptance criteria
- Manual UI verification notes (what was checked)

---

## BackendAgent
**Mission:** Implement backend CRUD, .sgm IO, and active-state endpoints.

**Responsibilities**
- Add/adjust FastAPI routes in `backend/app/main.py`.
- Extend `backend/app/services/modelset_service.py` (families + versions).
- Ensure `backend/app/state.py` has consistent active state.
- Ensure vector service reflects active version (cache reset if needed).
- Ensure robust `.sgm` import/export: zip-slip protection + checksums + no silent overwrites.

**Outputs**
- Back-end endpoints and services meeting plan requirements
- Clear error messages and guardrails for delete/overwrite
- Passing unit tests

---

## QualityAgent
**Mission:** Ensure the app stays stable and all tests/checklists pass.

**Responsibilities**
- Create/extend pytest coverage for:
  - CRUD routes and behaviors
  - .sgm export/import integrity
  - Not-available returns 200 with structured payloads
- Verify manual flows do not produce repeated console errors.
- Add smoke tests if missing, but keep them minimal.

**Outputs**
- Tests added/updated
- `pytest -q` passing
- A short checklist of what was validated manually

---

## ReleaseAgent
**Mission:** Finalize v4.3.x packaging.

**Responsibilities**
- Bump version to v4.3.0 (or v4.3.1 if hotfix)
- Update README/docs for ModelSet versions and .sgm usage
- Ensure `python run.py` works from repo root
- Ensure `requirements.txt` is accurate

**Outputs**
- Updated docs
- Final packaged zip and release notes
