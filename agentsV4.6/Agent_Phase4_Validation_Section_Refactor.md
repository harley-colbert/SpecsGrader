# Agent — Phase 4: Validation Section Refactor

You are the Phase 4 agent for the SpecsGrader v4.6 upgrade.

## Goal

- Move all Validation functionality (Sanity check & Holdout evaluation) into:
  - A fully implemented Step 5 card in Path B.
  - An optional, collapsed Validation section in Path A.
- Prevent confusing "Rows: 0" states for users who only want to load a ModelSet and classify.

## Inputs

- Stepper and Path B from Phase 3.
- Path A from Phase 2.
- Existing validation logic from v4.5 (sanity and evaluate endpoints).
- `planV4.6/Phase4_Validation_Section_Refactor.md`.

## High-Level Steps

1. Implement Step 5 — Validate in Path B.
2. Add a gated validation accordion to Path A.
3. Ensure no training dataset stats are shown in Path A unless the user explicitly loads validation data.
4. Update stepper state transitions.

## Detailed Instructions

1. **Implement Step 5 — Validate in Path B**
   - In `"build-update"` mode, locate the Step 5 placeholder card and replace it with a full card titled:
     - `Step 5 — Validate`
   - Inside the card, add:
     - Two main buttons:
       - `Run sanity check (fast)`
       - `Run holdout evaluation (slower)`
   - Gate these actions such that they are disabled unless:
     - A ModelSet is selected.
     - Training data with labeled rows is loaded (via Step 2).
     - Models have been trained (from Step 4).
   - When buttons are clicked:
     - Call the existing backend endpoints:
       - Sanity check.
       - Evaluate/holdout.
     - Show results inline:
       - For sanity:
         - Simple counts of correct vs incorrect predictions.
         - Optional top few mismatches.
       - For holdout:
         - Overall macro F1.
         - Per-label or per-class metrics if available.
   - After at least one successful validation run:
     - Mark Step 5 as completed in the stepper.

2. **Handle missing training data in Path B**
   - When no training data is loaded or `labeled_rows == 0`:
     - Keep Step 5 buttons disabled.
     - Inside the card, show helper text:
       - `Load labeled training data in Step 2 to enable validation.`
   - Do **not** duplicate the row count here; keep that in Step 2.

3. **Add optional Validation section for Path A**
   - In `"use-existing"` mode (Path A), after the ModelSet card and "Next step" card, add a collapsed accordion titled:
     - `Optional — Validate this ModelSet`
   - When the accordion is expanded:
     - Show text explaining:
       - `To validate this ModelSet, load a labeled dataset. Validation does not change the model; it only computes metrics.`
     - Provide:
       - A dedicated validation dataset loader:
         - File picker and `Load validation dataset` button.
       - Two buttons:
         - `Run sanity check (fast)`
         - `Run holdout evaluation`
     - Use the same endpoints and display logic as Step 5 in Path B, but treat the dataset as **validation-only** (does not retrain models).
   - If no validation dataset is loaded:
     - Disable the buttons and show text:
       - `No validation dataset loaded (optional). Load one to compute metrics for this ModelSet.`

4. **Avoid "Rows: 0" confusion in Path A**
   - Ensure that in Path A:
     - You do not show any training/validation dataset row counts **unless** a validation dataset was explicitly loaded in the Path A accordion.
     - If no validation data is loaded, only the explanatory text appears.

5. **Update stepper state for Step 5**
   - In Path B:
     - Mark the Step 5 node as:
       - `active` when the user is viewing/editing the card but has not yet run a validation.
       - `completed` when at least one validation run succeeded.
       - `error` if the validation call fails for reasons other than gated conditions (e.g. server error).

6. **Run tests**
   - Run backend tests: `pytest -q`, paying attention to any validation-related tests.
   - Run frontend tests (if available).
   - Perform manual exploration of:
     - Path B: run Sanity and Holdout with training data loaded and models trained.
     - Path A: open Validation accordion, load a dataset, run metrics.

7. **Document changes**
   - Write `docs/upgrade_v4.6/phase4_summary.md` describing:
     - Validation UX in Path B and Path A.
     - Gating rules.
     - How you avoided confusing "Rows: 0" messages.

## Success Checklist

- [ ] Step 5 in Path B provides working Sanity and Holdout actions.
- [ ] Step 5 is gated by ModelSet selection, training data, and trained models.
- [ ] Validation is optional but available in Path A via a collapsed accordion.
- [ ] No confusing "Rows: 0" states appear for Path A users who haven’t loaded validation data.
- [ ] Tests pass and Phase 4 summary describes validation behavior.
