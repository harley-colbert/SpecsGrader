# Runbook: executing planV3.0 with agentsV3.0

This runbook explains how to use the agents in this archive to complete the phases in `planV3.0.zip`.

## Setup
1. Unpack `SpecsGraderV2.3.zip` into a working directory.
2. Create and activate a virtual environment.
3. Install requirements:
   - `pip install -r requirements.txt`
4. Run baseline tests:
   - `python -m pytest -q`

## Phase execution pattern
For each phase file `planV3.0/NN_PHASE_X_*.md`:

1. Read the entire phase file.
2. Invoke each agent listed in its **Implementation steps** section, in the order listed.
3. After all agent work is done, run the **Tests (must run and pass)** from the phase.
4. Confirm every item in the **Success checklist** is true.
5. Only then proceed to the next phase.

## Recommended agent mapping by phase
- **Phase 0**: RepoAuditAgent
- **Phase 1**: BackendScaffoldAgent, FrontendESMAgent, APIContractAgent
- **Phase 2**: BackendScaffoldAgent, DataFramesAndPagingAgent, APIContractAgent, ParityTestAgent
- **Phase 3**: BackendScaffoldAgent, APIContractAgent, ParityTestAgent
- **Phase 4**: JobsAndLoggingAgent, APIContractAgent, ParityTestAgent
- **Phase 5**: JobsAndLoggingAgent, DataFramesAndPagingAgent, APIContractAgent, ParityTestAgent
- **Phase 6**: FrontendESMAgent, DataFramesAndPagingAgent, APIContractAgent, ParityTestAgent
- **Phase 7**: BackendScaffoldAgent, FrontendESMAgent, APIContractAgent, ParityTestAgent
- **Phase 8**: PackagingAgent, ParityTestAgent

## Testing discipline
- Treat each phase as a gated PR.
- If a required test fails, fix the failure before moving forward.
- Add tests only when they are stable and clearly tied to the phase acceptance criteria.
