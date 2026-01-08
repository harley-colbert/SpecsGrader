# SpecsGrader v4.6 — Train Pane UX Upgrade Plan

Base: SpecsGraderv4.5
Target: SpecsGraderv4.6
Scope: Train pane UI / UX, zero-logic-change to training backend except for wiring new UI controls and help text.

## Goals

- Make the Train pane **guided and understandable** for two main workflows:
  1. "Use an existing ModelSet and classify" (no training data required).
  2. "Build / update a ModelSet" (full training workflow with dataset, training, validation, vector store, snapshot).

- Reduce confusion around:
  - `Rows: 0` after loading a ModelSet.
  - "No labeled training data loaded" errors.
  - The purpose of Sanity/Evaluate buttons when no dataset is loaded.

- Introduce:
  - **Quick Start** workflow chooser at the top of the Train pane.
  - **Path A** UI for "Load ModelSet → Classify" as a primary/common workflow.
  - **Path B** stepper-style UI for "Build / update ModelSet".
  - An optional **Validation** section that clearly states training data is needed.
  - A **readiness strip** showing system status at a glance.
  - Better microcopy and inline explanations.

## Constraints

- Do not change the underlying training logic in the backend (already updated in v4.5).
- Only adjust backend APIs if strictly necessary for the new UI state (e.g., exposing readiness info).
- Keep classification behavior unchanged when using an existing ModelSet.
- All existing tests from v4.5 must pass; add new tests as needed for UI wiring, not for visual layout.

## Phases

- Phase 0: Baseline and backup
- Phase 1: Quick Start mode selector
- Phase 2: Path A — Load ModelSet & classify flow
- Phase 3: Path B — Build/Update ModelSet stepper
- Phase 4: Validation section refactor
- Phase 5: Readiness strip and microcopy
- Phase 6: QA, usability checks, and regression
- Phase 7: Release and versioning

Each phase file contains:
- Objectives
- Required context
- Step-by-step tasks for agents
- Tests
- Success checklist

Follow phases **in order**. Do not start a later phase until all tests in the prior phase pass.
