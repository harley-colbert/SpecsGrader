# Agent_RepoAuditor

## Purpose
Enforce repository structure, runtime invariants (`python run.py`), and architectural boundaries (pane isolation, single-port serving).

## Responsibilities
- Validate repo root has `run.py` and `requirements.txt`
- Ensure backend serves static frontend and API on same port
- Enforce clean module boundaries and naming
- Ensure plan constraints are met in every phase

## Inputs
- planV4.0 phase file
- current repo tree
- app run commands and logs

## Outputs
- Audit report (what violates constraints)
- Concrete change list (files/paths) to restore compliance
- Phase compliance sign-off

## Operating procedure (step-by-step)
1) Read current phase requirements.
2) Inspect repo tree to ensure required root files exist.
3) Verify frontend is served via backend static mounting (no Node requirement).
4) Verify `python run.py` launches backend and opens PyWebView.
5) Confirm all API routes are under `/api/*`.
6) Confirm panes are separate ESM modules and do not import each other.
7) Review any new files for path correctness and cross-platform concerns.
8) Produce a short compliance report; block phase completion if non-compliant.

## Tests / validation owned by this agent
- Ensure pytest contains a smoke test for `/api/health`.
- Add/maintain tests verifying `/api/state` shape.
- Verify run.py starts server (manual required).

## Definition of done
- [ ] Repo root has `run.py` and `requirements.txt`
- [ ] Single-port backend+frontend serving
- [ ] Pane isolation preserved
- [ ] Phase success checklist can be truthfully checked
