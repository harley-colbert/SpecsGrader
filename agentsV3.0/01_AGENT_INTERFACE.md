# Standard agent interface contract

All agents in this archive follow a consistent invocation and output format.

## What you pass to the agent
Provide:

1) **Repo context**
   - Path to the working copy of the repo (unpacked `SpecsGraderV2.3.zip`)
   - Any relevant environment notes (OS, Python version)

2) **The exact phase instructions**
   - The full text of the relevant phase file from `planV3.0/`
   - Any extra constraints (for example: no frontend bundler, pure ESM, upload-based workflows)

3) **Inputs / artifacts**
   - Any sample CSV or XLSX used for testing
   - Any existing config (for example: `models/model_sets.json`) if tests rely on it

## What the agent must do
- Read and follow the phase instructions in-order.
- Identify which files must be created or changed to satisfy the phase.
- Implement the changes with production-ready code (no placeholders).
- Run the required tests from the phase file.
- Verify every item in the phase success checklist is true.

## Required output format (agent response)
### Summary
- Bullet list of completed work

### Files changed
- Created:
- Modified:
- Deleted:

### Tests run
- Exact commands executed
- Expected outcome (for example: `pytest` passes)

### Notes
- Risks, tradeoffs, and anything the next phase needs to know

## Non-negotiables
- Do not output partial files or omit code. If a file is shown, it must be complete.
- Do not weaken acceptance criteria from the phase file.
- Keep existing tests passing unless the phase explicitly changes behavior.
- If you must rename or relocate files, update all imports and references in the repo.
