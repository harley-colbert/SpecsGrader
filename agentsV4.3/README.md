# SpecsGrader agentsV4.3

This folder defines the **agent roles** and **phase execution prompts** intended for ChatGPT Codex / agent-mode style execution of `planV4.3.zip`.

## How to use
1. Unzip `planV4.3.zip` and `agentsV4.3.zip` into the repo root (or keep them nearby).
2. For each phase:
   - Open the corresponding phase file in `planV4.3/`
   - Use the matching prompt in `agentsV4.3/prompts/`
3. Follow the "Tests that must pass" and "Success checklist" in the phase file. If anything fails, fix and re-run.

## Agent roster
- **OrchestratorAgent**: coordinates phases, enforces checklists, merges work, keeps scope tight.
- **FrontendAgent**: Train pane UI layout/styling + client API wiring.
- **BackendAgent**: FastAPI routes, app_state, modelset_service, vector_service behaviors.
- **QualityAgent**: pytest work, test additions, regression checks, console error checks.
- **ReleaseAgent**: version bump, docs, final packaging.

## Artifacts and conventions
- Never introduce 404 spam for "not ready" state; return 200 with structured payloads.
- Treat ModelSet versions as **immutable snapshots**; "update" is done by creating a new version.
- Keep changes scoped to the current phase only.
