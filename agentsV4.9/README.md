# agentsV4.9 (SpecsGrader)

These agent prompts are designed to execute **phaseV4.9.zip** (planV4.9/) in order, upgrading:

**SpecsGraderV4.6 → SpecsGraderV4.9**

## How to use

1. Unzip **phaseV4.9.zip** and **agentsV4.9.zip** into the same working directory.
2. Start with `Orchestrator_AgentsV4.9.md`.
3. Execute phases in order: Phase00 → Phase14.
4. Each phase prompt:
   - tells you which plan file to follow
   - what artifacts to produce
   - what tests must pass
   - what to record as completion evidence

## Assumptions

- You have the starting codebase available (SpecsGraderV4.6 or later).
- You can run Python tests locally (pytest) and any frontend dev/build checks if present.

## Agent Roles (assumed to exist)

- **OrchestratorAgent**: runs the whole phase, delegates to other agents, enforces gates.
- **BackendAgent**: implements backend changes (services, persistence, APIs).
- **FrontendAgent**: implements UI changes (Train pane stepper, metrics panels, etc.).
- **MLAgent**: implements modeling, embeddings, CV, policy logic, evaluation metrics.
- **TestAgent**: creates/updates automated tests; enforces “must pass” list.
- **QAAgent**: manual smoke tests + usability checks; captures screenshots if needed.
- **ReleaseAgent**: versioning, changelog, packaging `SpecsGraderv4.9.zip`.

## Output Convention (required)

After each phase completes, create:

`workspace/reports/PhaseXX_Completion.md`

Include:
- Summary of what changed
- Files touched (high level)
- Test commands run + results
- Screenshot notes (if UI changed)
- Any follow-ups / tech debt

Date: 2026-01-08
