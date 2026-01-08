# Agent — Phase 6: QA, Usability, and Regression

You are the Phase 6 agent for the SpecsGrader v4.6 upgrade.

## Goal

- Verify that both primary workflows behave correctly:
  - Path A: "Use existing ModelSet and classify".
  - Path B: "Build / update ModelSet".
- Confirm the new Train pane UX is intuitive and that no significant regressions were introduced.

## Inputs

- Completed implementations from Phases 0–5.
- `planV4.6/Phase6_QA_Usability_and_Regression.md`.

## High-Level Steps

1. Run through the full Path A flow and capture observations.
2. Run through the full Path B flow and capture observations.
3. Test key edge cases.
4. Run automated tests and confirm no regressions.

## Detailed Instructions

1. **Test Path A: Use existing ModelSet and classify**
   - Start the app: `python run.py`.
   - Navigate to Train pane:
     - Confirm readiness strip appears and shows `Active ModelSet: none` initially.
     - Confirm Quick Start defaults to `"Use an existing ModelSet and classify"`.
   - Load a ModelSet with trained models (from earlier work or sample data):
     - Verify readiness strip updates to show:
       - Active ModelSet ✅.
       - Models ✅ (if both level and dept).
       - Rules and vector store statuses (as applicable).
     - Verify Path A shows:
       - `Step 1 — Load a ModelSet` card with active ModelSet.
       - A "Next step" card with a `Go to Classify` button.
     - Click `Go to Classify` and ensure the Classify tab works as in v4.5:
       - You can paste or load text.
       - Predictions appear in expected format.
   - Optionally, test Validation in Path A:
     - Expand the "Optional — Validate this ModelSet" accordion.
     - Load a labeled validation dataset.
     - Run Sanity and Holdout and ensure metrics are displayed.

2. **Test Path B: Build / update ModelSet**
   - Switch Quick Start to `"Build / update a ModelSet"`.
   - Verify Path B appears with the stepper and Step 1 active.
   - Walk through the steps end-to-end:
     1. Select/Create ModelSet (Step 1):
        - Choose a ModelSet or create a new one.
     2. Load training data (Step 2):
        - Use a labeled dataset.
        - Confirm summary and preview appear.
     3. Rules (Step 3):
        - Optionally edit rules, save, and ensure JSON validation.
     4. Train models (Step 4):
        - Run training; confirm metrics appear and Step 4 is marked complete.
     5. Validate (Step 5):
        - Run Sanity and Holdout; confirm metrics and completion state.
     6. Vector store (Step 6):
        - Build the vector store if the UI supports it now or as placeholder for Phase 5 behavior.
     7. Save snapshot (Step 7):
        - Save a new version; ensure it shows up in ModelSet/versions list.
   - After saving:
     - Switch back to Path A.
     - Load the newly created ModelSet version.
     - Confirm `Go to Classify` appears and classification works.

3. **Edge cases**
   - Training data with **no labeled rows**:
     - Verify Step 2 shows clear guidance and does not allow training.
   - ModelSet with no models:
     - Load such a ModelSet and verify Path A displays guidance about rules-only or needing to build/update.
   - Missing vector store:
     - Ensure readiness strip shows Vector store ⭕ and any related copy is accurate.

4. **Automated tests**
   - Stop the app server if necessary.
   - Run backend tests:
     - `pytest -q`
   - Run frontend tests (if configured):
     - `npm test` or similar.
   - Confirm all tests pass.

5. **Record findings**
   - Create `docs/upgrade_v4.6/phase6_summary.md` including:
     - Path A behavior, notes on usability.
     - Path B behavior, notes on usability.
     - Any small issues found and how they were resolved (or flagged for later).

## Success Checklist

- [ ] Path A flow (load ModelSet → classify) works as expected and feels straightforward.
- [ ] Path B flow (ModelSet → data → rules → train → validate → vector → save) works end-to-end.
- [ ] Edge cases behave with clear, non-confusing messages.
- [ ] All automated tests pass.
- [ ] Phase 6 summary describes usability and any remaining minor issues.
