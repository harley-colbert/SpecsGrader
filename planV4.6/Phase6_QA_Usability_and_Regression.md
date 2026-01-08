# Phase 6 — QA, Usability, and Regression

## Objectives

- Verify that the new Train pane UX behaves correctly in both workflows.
- Ensure the changes did not introduce regressions in backend behavior.
- Validate that typical user flows are intuitive:
  - Load & classify.
  - Build/update ModelSet.

## Required Context

- All previous phases (0–5) complete.
- The app builds and runs without errors.

## Tasks

1. **Test Path A: Use existing ModelSet and classify**

   - Start in Path A mode.
   - With no ModelSet loaded:
     - Verify readiness strip shows "No active ModelSet".
     - Verify the Quick Start suggests loading a ModelSet.
   - Load a ModelSet with trained models:
     - Verify "Active ModelSet" updates.
     - Verify "Go to Classify" appears.
     - Switch to Classify tab and confirm classification UI behaves as in v4.5.
   - Optional: use the optional validation accordion to run validation with a labeled dataset.

2. **Test Path B: Build / update ModelSet**

   - Switch to Path B mode.
   - Walk through all steps:
     - Select/Create ModelSet.
     - Load training data (valid file).
     - Edit and save rules (optional).
     - Train models.
     - Run validation.
     - Build vector store.
     - Save snapshot.
   - Confirm:
     - Stepper states reflect step completion.
     - No steps are accessible out of order (buttons disabled when they should be).
     - The saved snapshot can be loaded in Path A and used for classification.

3. **Edge cases**

   - Training data with no labeled rows:
     - Verify error message and manual guidance in Step 2.
   - ModelSet loaded with no models:
     - Verify Path A shows appropriate guidance.
   - Vector store build with training data loaded but without models:
     - Confirm behavior is consistent with v4.5 (this should be primarily a UI change, not logic change).

4. **Regression test: existing tests**

   - Run backend tests:
     - `pytest -q`
   - Run frontend tests (if configured):
     - `npm test` or equivalent.
   - Confirm no new failing tests.

## Success Checklist

- [ ] Path A flow works end-to-end (load ModelSet → classify).
- [ ] Path B flow works end-to-end (create ModelSet → train → validate → vector → save → load & classify).
- [ ] No unexpected errors appear in the console/logs.
- [ ] All existing tests still pass.
- [ ] Any new tests added in previous phases pass.
