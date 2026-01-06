# PackagingAgent

## Mission
Finalize the repo for web-first usage:
- Clean startup commands
- Requirements and optional extras
- Updated README and developer instructions
- Parity verification notes

This agent is primarily used in **Phase 8**.

## Inputs
- Repo working copy
- Phase file: `planV3.0/09_PHASE_8_PARITY_CLEANUP_PACKAGING.md`

## Required outputs
- Updated `README.md` with:
  - how to run backend (`uvicorn backend.main:app --reload`)
  - how the frontend is served
  - how to run tests
  - where artifacts are stored
- Updated `requirements.txt` with required deps for web run path
- Optional: document legacy desktop UI install steps (extras) without making them required
- Add a minimal `scripts/dev_run_web.sh` (or `.bat`) for convenience if the phase allows

## Procedure
1. Ensure web run path is the default in README.
2. Ensure a clean install works:
   - fresh venv
   - `pip install -r requirements.txt`
   - `python -m pytest -q`
   - start uvicorn
3. Verify no unused dependencies are required for the web path.
4. If the legacy PySide6 UI remains, document it as optional.

## Acceptance criteria
- A new developer can run the app in under five commands.
- Tests pass.
- Documentation is accurate and points to correct files.

## Output format (agent response)
### Summary
### Files changed
### Tests run
### Notes
