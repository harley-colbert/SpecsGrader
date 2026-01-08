# Phase 4 — Validation Section Refactor

## Objectives

- Refactor the **Validation** features (Sanity check and Holdout evaluation) into a clear, gated Step 5 in Path B.
- Add an optional Validation section in Path A that:
  - Is collapsed by default.
  - Clearly states that labeled training data is required.
- Avoid showing "Rows: 0" or similar as a failed state when user is only in Path A.

## Required Context

- Stepper and Step 5 placeholder exist in Path B (Phase 3).
- Path A is implemented (Phase 2).

## Tasks

1. **Move Validation controls into a dedicated Step 5 card for Path B**

   - Step 5 card title: `Step 5 — Validate`
   - Inside, provide two primary actions:
     - `Run sanity check (fast)`
     - `Run holdout evaluation (slower)`
   - Gate these actions:
     - Require:
       - A ModelSet to be selected.
       - Labeled training data to be loaded.
       - Models to be trained (for model-based metrics).
   - Use existing backend endpoints:
     - Sanity check
     - Evaluate (holdout metrics)
   - Display results inline under each button:
     - For sanity: summary of matches/mismatches.
     - For holdout: key metrics (macro F1, recall, per-class highlights).

2. **Prevent confusion when no training data is loaded**

   - If the user has not loaded training data in Path B:
     - Disable the validation buttons.
     - Show helper text (in the card body):
       - `Load labeled training data in Step 2 to enable validation.`
   - Do not show dataset row counts in the Validation card; keep that in Step 2.

3. **Add an optional validation section for Path A**

   - In `mode === "use-existing"` (Path A), add a collapsed accordion:
     - Title: `Optional — Validate this ModelSet`
   - When expanded:
     - Explain:
       - `To validate this ModelSet, load a labeled dataset. Validation does not change the existing model; it only computes metrics.`
     - Provide:
       - File picker and `Load validation dataset` button.
       - Two buttons:
         - `Run sanity check (fast)`
         - `Run holdout evaluation`
       - Display metrics inline, similar to Step 5 in Path B.
   - Clearly label this as **optional** and separate from the main classification workflow.

4. **Avoid "Rows: 0" confusion in Path A**

   - When in Path A:
     - Do not display any training dataset stats unless the user explicitly loads a validation dataset.
     - If no validation dataset is loaded, show simple text like:
       - `No validation dataset loaded (optional). Load one to compute metrics for this ModelSet.`

5. **Update stepper state for Step 5**

   - In Path B:
     - Mark Step 5 as completed when at least one validation run successfully completes.
     - If validation fails due to missing data, keep Step 5 in an "available but incomplete" state.

## Tests

- Manual:
  - Path B:
    - With ModelSet selected, training data loaded, and models trained:
      - Run sanity check and holdout evaluation.
      - Confirm metrics are displayed and Step 5 is marked as completed.
    - Without training data:
      - Confirm validation buttons are disabled and helper text points back to Step 2.
  - Path A:
    - Confirm the optional validation accordion exists but is collapsed by default.
    - Confirm validation only becomes available after a dataset is explicitly loaded.
    - Confirm no confusing "Rows: 0" messages appear when user has not loaded validation data.

## Success Checklist

- [ ] Validation UI is fully contained in Step 5 for Path B.
- [ ] Validation is optional and clearly marked in Path A.
- [ ] Validation actions are correctly gated by the presence of labeled data and trained models.
- [ ] "Rows: 0" style messages do not appear in Path A unless explicitly meaningful.
- [ ] Metrics are displayed clearly and inline for both sanity and holdout runs.
