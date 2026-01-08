# Agent — Phase 3: Path B Stepper (Build / Update ModelSet)

You are the Phase 3 agent for the SpecsGrader v4.6 upgrade.

## Goal

- Implement **Path B** for `"build-update"` mode as a stepper-based workflow:
  1. Select/Create ModelSet
  2. Load Training Data
  3. Rules (optional)
  4. Train Models
  5. Validate (placeholder)
  6. Vector Store (placeholder)
  7. Save Snapshot/Export (placeholder)
- Gate later steps based on completion of earlier ones.

## Inputs

- Train pane with Quick Start + Path A from previous phases.
- `planV4.6/Phase3_PathB_Build_Update_ModelSet_Stepper.md`.
- Existing Train pane controls from v4.5 (data loader, train button, rules editor, etc.).

## High-Level Steps

1. Add a horizontal stepper visible in `"build-update"` mode.
2. Create Step cards 1–4 with fully implemented behavior.
3. Create placeholder cards for Steps 5–7 (to be completed later).
4. Relocate and reorganize v4.5 controls into appropriate steps.

## Detailed Instructions

1. **Add the stepper UI**
   - In `"build-update"` mode, at the top of Path B, render a horizontal stepper with labels:
     - `1. ModelSet`
     - `2. Training data`
     - `3. Rules`
     - `4. Train`
     - `5. Validate`
     - `6. Vector store`
     - `7. Save`
   - For each step, support a basic status:
     - `inactive` (default)
     - `active` (current editing step)
     - `completed`
     - `error`
   - Clicking a step may scroll to the corresponding card, but **does not** allow users to bypass gating.

2. **Step 1 — Select/Create ModelSet**
   - Create a card with title:
     - `Step 1 — Select or create ModelSet`
   - Inside, reuse or adapt ModelSet selection/creation controls:
     - Ability to choose an existing ModelSet + version as the "target" for training results.
     - Ability to create a new ModelSet name if supported.
   - On success:
     - Mark Step 1 as completed.
     - Set Step 2 as active.
   - Gate logic:
     - Training and later steps must be disabled if no target ModelSet is selected.

3. **Step 2 — Load Training Data**
   - Create a card with title:
     - `Step 2 — Load labeled training data`
   - Move the v4.5 training data loader controls here:
     - File picker.
     - `Load training data` button.
   - After loading:
     - Compute and display:
       - Total rows.
       - Number of labeled rows.
       - Any missing label fields.
     - Show a small table preview (first ~10 labeled rows) if feasible.
   - Determine labeled viability:
     - If `labeled_rows > 0`:
       - Mark Step 2 as completed.
       - Advance Step 3 as active.
     - If `labeled_rows == 0`:
       - Mark Step 2 as error.
       - Show explicit guidance text referencing:
         - Columns `E`/`F`/`G`.
         - Expected labels for level/dept.
   - Gate logic:
     - Keep Step 4 Train controls disabled while `labeled_rows == 0`.

4. **Step 3 — Rules (optional)**
   - Create a card with title:
     - `Step 3 — Rules (optional)`
   - Add a simple status line:
     - `Rules status: Default rules loaded` or `Rules status: Modified`.
   - Provide buttons:
     - `Edit rules` — opens the existing JSON rules editor within a collapsible panel.
     - `Test a phrase` — optional test UI for rules.
   - In the editor view:
     - Include `Save rules` and `Revert` buttons.
     - Validate JSON when saving; show error messages inline if invalid.
   - Completion:
     - If user edits and saves valid rules, mark Step 3 as completed.
     - If user skips, Step 3 can be marked as "optional" or "skipped" but should not block later steps.

5. **Step 4 — Train Models**
   - Create a card titled:
     - `Step 4 — Train models`
   - Move existing training controls here:
     - `Train` button.
     - Training parameters (calibration, class weights, etc.).
   - UI organization:
     - Show only essential parameters by default.
     - Place advanced options (e.g., calibration method, cost grid, oversampling) inside a collapsible `Advanced settings` section.
   - Gate logic:
     - Disable `Train` button while:
       - No target ModelSet selected (Step 1 incomplete).
       - No viable training data (`labeled_rows == 0` or Step 2 error).
   - Behavior:
     - On click `Train`, call the same backend training endpoint used in v4.5.
     - Show progress and recent events log within the card.
   - On successful training:
     - Display headline metrics:
       - Level macro F1.
       - Dept macro F1.
     - Mark Step 4 as completed.
     - Set Step 5 (Validate) as active.

6. **Steps 5–7 placeholders**
   - Implement placeholder cards with titles:
     - `Step 5 — Validate`
     - `Step 6 — Vector store`
     - `Step 7 — Save snapshot / Export`
   - For now:
     - Add short descriptive text.
     - Disable their buttons (if any) until future phases.
     - Wire them into the stepper statuses as `inactive` or `pending`.

7. **Run tests**
   - Run frontend tests if configured.
   - Run backend tests: `pytest -q`.
   - Ensure training still functions and no major regressions occur.

8. **Document changes**
   - Add `docs/upgrade_v4.6/phase3_summary.md` describing:
     - Component files modified.
     - Stepper structure and gating conditions.
     - Any UX decisions for optional rules step.

## Success Checklist

- [ ] Stepper appears in `"build-update"` mode with 7 steps.
- [ ] Step 1 lets you select/create a ModelSet and marks completion.
- [ ] Step 2 loads training data, shows summary and preview, and handles zero-labeled data with clear error messaging.
- [ ] Step 3 allows editing and saving rules, with JSON validation and optional skip.
- [ ] Step 4 trains models only when Steps 1–2 are satisfied; shows metrics and marks completion.
- [ ] Steps 5–7 exist as placeholders wired into the stepper.
- [ ] Tests pass and Phase 3 summary describes the implementation.
