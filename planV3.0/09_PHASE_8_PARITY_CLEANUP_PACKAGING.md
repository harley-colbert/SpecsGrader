    # Phase 8 — Parity verification + cleanup + packaging

    ## Goal
    Confirm end-to-end parity with the desktop workflow and package web UI as the default interface.

    ## Prerequisites
    - Phase 7 completed.

    ## Implementation steps (must follow in order)
    1. **Agent:** ParityTestAgent
   - Create an end-to-end smoke test script `scripts/smoke_web_e2e.py` that:
     - starts TestClient
     - uploads training + doc fixtures
     - runs train and classify jobs
     - applies one override
     - exports
     - asserts outputs exist

2. **Agent:** PackagingAgent
   - Update `README.md` to make web UI the default.
   - Keep desktop UI optional (document as legacy) rather than removing immediately.
   - Ensure requirements are clean and minimal.
   - Add `make_web.sh` / `make_web.bat` (optional) or update `DEV_COMMANDS.md`.

3. **Agent:** RepoAuditAgent
   - Confirm there is no dead code or circular dependency introduced.
   - Confirm folder layout matches plan.

4. Final UX pass:
   - Sidebar stepper is clear
   - Errors are user-readable
   - Job failures show stack traces in logs but friendly messages in UI

    ## Files to create/change
    - **Create:** `scripts/smoke_web_e2e.py`
- **Change:** `README.md`, `DEV_COMMANDS.md`
- **Optional:** `requirements-legacy-ui.txt` for PySide6

    ## Tests (must run and pass)
    1. `python -m pytest -q`
2. `python scripts/smoke_web_e2e.py`
3. Manual full workflow in browser:
   - Import → Train → Classify → Review → Export

    ## Success checklist (must be true before moving on)
    - ✅ All unit tests pass
- ✅ `scripts/smoke_web_e2e.py` passes
- ✅ Web UI completes full workflow end-to-end
- ✅ Documentation updated and accurate
- ✅ Desktop UI remains runnable (optional) OR clearly marked as legacy with install instructions

    ## Notes for agents (assumed available)
    - ParityTestAgent should define at least 3 parity checkpoints:
  1) Excel sheet selection
  2) Classification output column set
  3) Review override persistence + export
- PackagingAgent should ensure Windows and macOS instructions are both present if you support both.
