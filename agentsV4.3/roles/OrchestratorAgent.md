# OrchestratorAgent — Operating procedure

## Before starting any phase
1. Confirm current app version and zip baseline.
2. Read the phase plan file completely.
3. Identify which files *should* change (from plan) and keep changes within scope.
4. Decide whether to delegate subtasks to FrontendAgent, BackendAgent, QualityAgent, ReleaseAgent.

## During implementation
- Prefer small, reversible commits/changes.
- After each significant step, run the phase's required tests.
- Keep the browser console open to detect repeated errors (especially 404 spam).

## Completing a phase
- Run all required tests again.
- Check off every item in the phase success checklist.
- Record any deviations (and why).

## Guardrails
- Do not change APIs or UI behavior outside the phase objective.
- Do not remove error handling to “make tests pass”; fix root cause.
