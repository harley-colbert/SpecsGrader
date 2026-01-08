# Phase 3 — Path B: Build / Update ModelSet Stepper

## Objectives

- Implement a stepper-style **Path B** for the "Build / update ModelSet" workflow.
- Split the Train pane advanced workflow into clear steps:
  1. Select/Create ModelSet
  2. Load Training Data
  3. Rules (optional)
  4. Train Models
  5. Validate (ties into Phase 4)
  6. Vector Store
  7. Save Snapshot/Export
- Gate later steps based on completion of earlier ones.

## Required Context

- Train pane supports workflow mode state (Phase 1).
- Path A has been implemented (Phase 2) and can be hidden when `mode === "build-update"`.

## Tasks

1. **Add a horizontal stepper at the top of Path B**

   - When `mode === "build-update"`, render a stepper row with 7 steps:
     1. ModelSet
     2. Training data
     3. Rules
     4. Train
     5. Validate
     6. Vector store
     7. Save
   - Each step should display a small status indicator:
     - e.g., default, active, completed, or error.
   - Clicking a step may scroll to the relevant card, but cannot skip gating (i.e., buttons inside later cards remain disabled until prerequisites are met).

2. **Step 1 — Select/Create ModelSet**

   - Create a card similar to Path A’s ModelSet loader, but tailored to building/updating:
     - Title: `Step 1 — Select or create ModelSet`
   - Provide:
     - Ability to select an existing ModelSet + version as the target for new snapshots.
     - Ability to create a new ModelSet name.
   - Completion criteria:
     - Some ModelSet is selected as the active target.
   - Once complete, update the stepper to mark Step 1 as completed and Step 2 as the current step.

3. **Step 2 — Load Training Data**

   - Create a card:
     - Title: `Step 2 — Load labeled training data`
   - Include:
     - File picker and `Load training data` button.
   - After load, show:
     - Summary: total rows, labeled rows, missing fields.
     - A small preview of the first 10 labeled rows.
   - Check label viability:
     - If labeled_rows > 0 → mark as completed.
     - If labeled_rows == 0 → mark as error; show explicit guidance on expected columns (E/F/G) and valid label values.
   - Disable Train, Validation, Vector Store, and Save steps if labeled_rows == 0.

4. **Step 3 — Rules (optional)**

   - Create a card:
     - Title: `Step 3 — Rules (optional)`
   - Show a simple view by default:
     - "Rules status: Default rules loaded" or "Rules modified".
     - Buttons:
       - `Edit rules` (opens existing JSON editor in a collapsible section).
       - `Test a phrase` (optional).
   - Provide a `Save rules` button within the editor view that:
     - Validates JSON and updates app state.
   - Completion criteria:
     - If the user edits rules, they must save successfully (JSON valid).
     - User may skip this step; stepper may show "Skipped" state.

5. **Step 4 — Train Models**

   - Create a card:
     - Title: `Step 4 — Train models`
   - Include:
     - A Train button (`Train`).
     - Training parameters, but with advanced options behind a collapsible "Advanced settings".
   - Gate:
     - Disable Train button if:
       - No active ModelSet selected (Step 1 incomplete).
       - No labeled training data loaded (Step 2 incomplete).
   - On training start:
     - Show progress and recent events.
   - On success:
     - Mark Step 4 as completed.
     - Display headline metrics (e.g., Level macro F1, Dept macro F1).
     - Suggest: "Next: Validate".

6. **Step 5 — Validate (preview) and Step 6/7 placeholders**

   - For this phase, you can prepare placeholder cards for:
     - Step 5 — Validate
     - Step 6 — Vector store
     - Step 7 — Save snapshot/export
   - Do not implement full logic yet; this will be completed in Phase 4 and Phase 5.
   - Ensure their buttons are disabled until prior steps are complete.

## Tests

- Manual:
  - Switch to "Build / update ModelSet" mode.
  - Confirm the stepper appears with 7 steps.
  - Verify:
    - User cannot trigger Train until ModelSet is selected and training data with labeled rows is loaded.
    - Rules editing works and invalid JSON is caught.
    - Stepper states update as steps are completed (ModelSet selected, training data loaded, models trained).
  - Confirm Path A is hidden or visually secondary when in Path B mode.

- Automated:
  - Add tests (if applicable) to verify that:
    - The Train button is disabled when prerequisites are not met.
    - The stepper renders in the correct mode and steps reflect the app state.

## Success Checklist

- [ ] Stepper is visible in "build-update" mode with correct step labels.
- [ ] Step 1 allows selecting/creating a ModelSet and marks completion.
- [ ] Step 2 loads training data, shows summary and preview, and supports labeled/zero-labeled states with proper messaging.
- [ ] Step 3 allows editing and saving rules with JSON validation.
- [ ] Step 4 gates training based on previous steps and marks completion with headline metrics.
- [ ] Placeholder cards exist for Steps 5–7 for later phases.
- [ ] Tests and manual checks confirm behavior.
