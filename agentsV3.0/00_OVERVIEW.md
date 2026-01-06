# agentsV3.0 — SpecsGrader Web UI Migration Agents

This archive contains the **matching agent prompts** referenced by `planV3.0.zip`.

## Intended use
Use these agents to execute **planV3.0 phases 0–8** against **SpecsGraderV2.3**.

Each plan phase file calls out an agent by name. Invoke that agent with the repo + phase file instructions.

## Operating rules (apply to all agents)
1. **Follow the phase file order.** Do not skip steps.
2. **No placeholders.** Do not leave `TODO`, `pass`, stubs, or fake returns in production code.
3. **No omitted code.** When you output code files, output complete files (no ellipses).
4. **Keep existing behavior and tests working** unless the phase explicitly changes behavior.
5. **Every phase has gates:** required tests must pass and the success checklist must be true.
6. **Prefer small, reviewable commits/diffs** that match the phase scope.
7. **Be explicit.** If you rename, move, or delete anything, explain why and update references.

## Deliverables expected from an agent run
For each invocation, the agent must produce:
- A short summary of what changed (bullet list)
- A complete list of created/modified/deleted files
- Commands to run tests and the expected outcome
- Any follow-up notes needed by the next phase

## Agent roster
- RepoAuditAgent
- BackendScaffoldAgent
- FrontendESMAgent
- APIContractAgent
- JobsAndLoggingAgent
- DataFramesAndPagingAgent
- ParityTestAgent
- PackagingAgent

See `01_AGENT_INTERFACE.md` for the standard input/output contract.
