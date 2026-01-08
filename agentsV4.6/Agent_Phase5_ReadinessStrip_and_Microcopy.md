# Agent — Phase 5: Readiness Strip and Microcopy

You are the Phase 5 agent for the SpecsGrader v4.6 upgrade.

## Goal

- Add a compact **readiness strip** at the top of the Train pane showing global status.
- Align microcopy across Path A and Path B to clarify workflows, requirements, and error states.

## Inputs

- Train pane with Path A, Path B, and Validation refactor from earlier phases.
- `planV4.6/Phase5_ReadinessStrip_and_Microcopy.md`.

## High-Level Steps

1. Implement readiness strip that reflects:
   - Active ModelSet.
   - Models.
   - Rules.
   - Vector store.
   - Training data loaded.
2. Update copy across the Train pane to:
   - Clarify training vs classification.
   - Explain when data is required.
   - Provide actionable error/empty-state messages.

## Detailed Instructions

1. **Implement readiness strip**
   - At the top of the Train pane, above the Quick Start card, add a slim horizontal bar or small card containing:
     - `Active ModelSet: (none / name)` with ✅ if present, ⭕ if none.
     - `Models:` ✅ if both level and dept models exist, ⭕ otherwise.
     - `Rules:` ✅ if a rules config exists, ⭕ otherwise.
     - `Vector store:` ✅ if a vector index is present, ⭕ otherwise.
     - `Training data loaded:` ✅ if current Train pane session has a loaded labeled dataset, ⭕ otherwise.
   - Use existing app state and selectors to derive these booleans; do **not** perform heavy operations on render.
   - Prefer a compact visual style so it does not overshadow main content.

2. **Align Path A copy**
   - In Path A (use existing ModelSet):
     - Ensure that after loading a ModelSet, the "Next step" card clearly states:
       - `You’re ready to classify using the active ModelSet. Training data is not required for this workflow.`
     - In the Path A Validation accordion:
       - Explain that validation is **optional**, and that it will not modify the ModelSet.
       - Clarify that labeled data is required **only** to compute metrics (not to classify).

3. **Align Path B copy**
   - For each step card in Path B, add 1–2 lines of top-of-card description:
     - Step 1 (ModelSet):
       - Example: `Choose which ModelSet you want to train and save new versions into.`
     - Step 2 (Training data):
       - Example: `Load labeled examples so the system can learn to predict risk level and department.`
     - Step 3 (Rules):
       - Example: `Define rule-based overrides (e.g., keywords) that complement the machine learning model.`
     - Step 4 (Train):
       - Example: `Train ML models for risk level and department using the loaded dataset.`
     - Step 5 (Validate):
       - Example: `Check how the models perform on your labeled data.`
     - Step 6 (Vector store):
       - Example: `Build a similarity index so new text can be compared to known examples.`
     - Step 7 (Save):
       - Example: `Save everything to a new ModelSet version you can reuse in the Classify tab.`

4. **Clarify common error / empty states**
   - When training data is present but has no labeled rows:
     - Use a message like:
       - `We found data rows but no labels in columns F and G starting at row 5. Please ensure your file includes labeled rows with valid risk level and department values.`
   - When a ModelSet is loaded but has no models:
     - Path A should say something like:
       - `This ModelSet has no trained models. You can still use rules (if present), or switch to the Build/Update workflow to train models.`
   - When no ModelSet is active:
     - The readiness strip should show `Active ModelSet: none` with ⭕.
     - At least one message on the Train pane should say:
       - `No active ModelSet. Load one to classify, or switch to the Build/Update workflow to create one.`

5. **Normalize button labels**
   - Verify that buttons use consistent terminology:
     - `Load training data` (for Step 2 / Path B).
     - `Load validation dataset` (for Validation in Path A).
     - `Import & load .sgm` (for ModelSet imports).
     - `Train models`, `Build vector store`, `Save snapshot`, etc.
   - Update labels and tooltips where necessary to avoid ambiguity.

6. **Tooltips/help icons (if supported)**
   - For advanced concepts (e.g., vector store), add info icons with one-sentence explanations:
     - Example: `Vector store enables nearest-neighbor style similarity, comparing new text to similar past examples.`

7. **Run tests**
   - Ensure no new runtime or test errors:
     - `pytest -q`
     - Frontend tests (`npm test`) if configured.

8. **Document changes**
   - Create `docs/upgrade_v4.6/phase5_summary.md` summarizing:
     - What the readiness strip shows.
     - Key copy improvements in Path A and Path B.
     - Any changed labels or tooltips.

## Success Checklist

- [ ] Readiness strip appears and accurately reflects app state.
- [ ] Path A clearly communicates that classification does not require training data.
- [ ] Path B steps have clear descriptions of their purpose.
- [ ] Error and empty states give actionable guidance.
- [ ] Button labels are consistent and unambiguous.
- [ ] Tests pass and Phase 5 summary is documented.
