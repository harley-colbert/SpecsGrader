# Agent — Phase 2: Path A (Load ModelSet and Classify)

You are the Phase 2 agent for the SpecsGrader v4.6 upgrade.

## Goal

- Implement **Path A** UI and behavior for the Train pane in `"use-existing"` mode:
  - Load or import a ModelSet.
  - Show a readiness summary (models, rules, vector store).
  - Provide a clear "Go to Classify" call-to-action when ready.
  - Hide training/validation/vector controls in this mode.

## Inputs

- Updated Train pane from Phase 1 (with `mode` and Quick Start).
- `planV4.6/Phase2_PathA_Load_ModelSet_and_Classify.md`.
- Existing ModelSet load/import logic from v4.5.

## High-Level Steps

1. Refactor existing ModelSet loader into a `Step 1 — Load a ModelSet` card within Path A.
2. Add a "what's inside this ModelSet" mini-summary (models/rules/vector store).
3. Add a "Next step" card with "Go to Classify" logic.
4. Ensure that all training/validation/vector UI is hidden in Path A.
5. Run tests and document changes.

## Detailed Instructions

1. **Refactor ModelSet loader into a Path A card**
   - In `"use-existing"` mode, render a card titled:
     - `Step 1 — Load a ModelSet`
   - Move or reuse the existing ModelSet selection/import controls from v4.5 into this card:
     - Dropdown for existing ModelSets + versions (if available).
     - File-picker for importing `.sgm` bundles.
     - Load button and Import & Load button.
   - Ensure that loading or importing sets the active ModelSet in the same way as before.

2. **Add tabs if both local listing and .sgm import are supported**
   - If previously supported:
     - Add two tabs (or segmented controls) inside the card:
       - `Local ModelSets` — list existing definitions and versions.
       - `Import .sgm` — file upload and import.
   - Keep behavior identical to v4.5, but now visually organized under this card.

3. **Show ModelSet readiness summary**
   - After a ModelSet is successfully loaded:
     - Show a banner in the card, e.g.:
       - `Active ModelSet: Model_NAME / Version X` with a ✅ icon.
     - Immediately underneath, show a short readiness summary:
       - `Models (level/dept):` ✅ if both present, otherwise ⭕.
       - `Rules:` ✅ if rules object is present, otherwise ⭕.
       - `Vector store:` ✅ if vector index is available, otherwise ⭕.
   - Use existing app state or API responses to derive these flags.

4. **Add Path A "Next step" card**
   - Below the ModelSet card, create a small card titled `Next step`:
     - If models are present:
       - Show a primary button: **Go to Classify**
       - Helper text: `You’re ready to classify using the active ModelSet. Training data is not required for this workflow.`
       - Clicking the button should switch to the Classify tab (e.g., via router or tab context).
     - If no models but rules exist:
       - Show a button: **Go to Classify (rules-only)** and explain that ML models are missing.
     - If neither models nor rules exist:
       - Show a message:
         - `This ModelSet has no usable models or rules. Import another version or switch to the Build/Update workflow to train models.`
       - Provide a button: **Switch to Build / update ModelSet** that sets `mode = "build-update"`.

5. **Hide training/validation/vector controls in Path A**
   - When `mode === "use-existing"`:
     - Do not render:
       - Training data loader.
       - Training parameter controls.
       - Train button.
       - Sanity check / holdout evaluation buttons.
       - Vector store build button.
   - If necessary, move these controls into the Path B container where they will be reorganized in later phases.

6. **Copy / messaging**
   - Ensure Path A text explicitly states:
     - `Note: Training data is only required for training and validation, not for classification with an existing ModelSet.`

7. **Run tests**
   - Run frontend tests (if any): `npm test` or equivalent.
   - Run backend tests: `pytest -q`.
   - Verify that ModelSet loading still functions as in v4.5 (no regression).

8. **Document changes**
   - Update `docs/upgrade_v4.6/phase2_summary.md` including:
     - Which components were modified.
     - How readiness flags are computed.
     - How navigation to the Classify tab is implemented.

## Success Checklist

- [ ] Path A shows `Step 1 — Load a ModelSet` card in `"use-existing"` mode.
- [ ] After loading a ModelSet, readiness summary appears (models/rules/vector store).
- [ ] A prominent "Go to Classify" (or rules-only variant) button appears when appropriate.
- [ ] No training/validation/vector controls are visible in Path A.
- [ ] All tests pass and Phase 2 summary is written.
