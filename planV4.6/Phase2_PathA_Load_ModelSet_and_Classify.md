# Phase 2 — Path A: Load ModelSet and Classify

## Objectives

- Implement a **Path A** UI that focuses on:
  - Loading or importing a ModelSet.
  - Showing what components exist in the loaded ModelSet (models, rules, vector store).
  - Clearly directing the user to the Classify tab when ready.
- Make it clear that **training data is not required** for this workflow.

## Required Context

- Train pane now has workflow mode state (Phase 1).
- Backend already supports:
  - Listing ModelSets and versions.
  - Loading a ModelSet.
  - Importing a `.sgm` bundle.

## Tasks

1. **Create the Path A container UI**

   - Under the Quick Start card, in `mode === "use-existing"`, render a card titled:
     - `Step 1 — Load a ModelSet`
   - Inside, provide:
     - A dropdown or selector for existing ModelSets and versions **or**
     - A clear import `.sgm` path.
   - Reuse or refactor existing ModelSet UI from v4.5 into this card.

2. **Two tabs for ModelSet loading (if applicable)**

   - If the app already supports both local listing and `.sgm` import, structure them as:
     - Tab 1: "Local ModelSets" — for listing existing ModelSets and versions.
     - Tab 2: "Import .sgm" — for uploading/importing a `.sgm` file.
   - Ensure the "Import & load" path sets the active ModelSet just as loading a local version would.

3. **Display active ModelSet and its contents**

   - After a ModelSet is loaded successfully:
     - Show a banner like:
       - `Active ModelSet: Model_XXXX / Version YYYY` (with a ✅ icon).
     - Below, show a small "what's inside" summary:
       - `Models (level/dept):` ✅/⭕
       - `Rules:` ✅/⭕
       - `Vector store:` ✅/⭕
   - This summary can reuse the data already available in app state or returned from load endpoints.

4. **Add a Path A "Next step" CTA**

   - Below the ModelSet summary, render a "Next step" card that is only visible when a ModelSet is active.
   - Behavior:
     - If level/dept models exist:
       - Show a primary button: `Go to Classify`
       - Helper text: `You’re ready to classify using the active ModelSet. Training data is not required for this workflow.`
     - If no models exist but rules exist:
       - Show: `Go to Classify (rules-only)` and explain that no ML models are available.
     - If neither models nor rules exist:
       - Show a message like:
         - `This ModelSet has no usable models or rules. Import another version or build a new ModelSet in the advanced workflow.`
       - Provide a button: `Switch to Build / update ModelSet` that toggles mode to `"build-update"`.

5. **Ensure training sections are hidden in Path A**

   - When `mode === "use-existing"`, do **not** show:
     - Training data loader.
     - Training parameters.
     - Train button.
     - Sanity or holdout evaluation buttons.
     - Vector store build step.
   - This avoids the current confusion of seeing "Rows: 0" and training controls when user only wants to classify.

6. **Copy and messaging**

   - Add explicit text that training data is optional in this mode:
     - Example: `Note: Training data is only required for training and validation, not for classification with an existing ModelSet.`

## Tests

- Manual:
  - Switch to "Use existing ModelSet and classify" mode.
  - Load a ModelSet with models:
    - Confirm the active ModelSet is shown.
    - Confirm the "what's inside" summary (models/rules/vector store).
    - Confirm a prominent "Go to Classify" button appears.
  - Load a ModelSet without models but with rules:
    - Confirm the button text indicates rules-only classification.
  - Load a ModelSet with nothing useful:
    - Confirm the UI suggests switching to the Build/Update workflow.
  - Verify that no training-data-related sections are visible in Path A.

- Automated:
  - If frontend tests exist, add a test that:
    - Renders Train pane in "use-existing" mode with a mocked active ModelSet.
    - Verifies that training UI elements are not present and the "Go to Classify" button is present.

## Success Checklist

- [ ] Train pane Path A offers a clear "Step 1 — Load a ModelSet" card.
- [ ] After loading a ModelSet, the UI shows its readiness (models/rules/vector store).
- [ ] Path A presents a single, obvious "Go to Classify" action when possible.
- [ ] No training/validation/vectors UI is shown in Path A.
- [ ] Tests pass and manual verification confirms the expected experience.
