# Orchestrator (single prompt to run planV2.0 with agentsV2.0)

Use this as a master instruction block for an agentic runner.

## Inputs you must have
- Repo checked out locally
- `planV2.0.zip` extracted or accessible
- This agents pack available in the workspace

## Execution rules
- Run phases in order: 0 → 7
- Do not proceed to the next phase until:
  1) phase-specific tests pass
  2) phase success checklist is fully satisfied
  3) regression checklist passes
- If a phase introduces regressions, fix them before continuing.

## Phase loop template
For each phase file in `planV2.0/00_PHASES/`:
1) Read the phase instructions.
2) Invoke the referenced agent(s) from this pack.
3) Implement required changes.
4) Run phase manual tests (and any automated tests available).
5) Update required artifacts and screenshots.
6) Run `ui_regression_tester.md` checklist.
7) Mark phase checklist items as completed in a short phase report:
   - `02_ASSETS/phaseX_report.md`

## Output expectation
At the end you should have:
- A UI that clearly guides a first-time user through:
  Import → Review/Label → Train → Classify → Export
- Updated docs:
  - `01_SHARED/UX_STATE_MAP.md`
  - `01_SHARED/LABELING_POLICY.md`
  - `01_SHARED/EXPORT_PRESETS.md`
- Proof artifacts:
  - screenshots
  - usability test results
  - regression results
