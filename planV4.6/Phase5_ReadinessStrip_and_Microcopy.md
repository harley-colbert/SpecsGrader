# Phase 5 — Readiness Strip and Microcopy

## Objectives

- Add a compact **readiness strip** at the top of the Train pane that shows global system status:
  - Active ModelSet
  - Models
  - Rules
  - Vector store
  - Training data loaded (optional)
- Improve microcopy across the Train pane to clarify:
  - Training vs classification workflows.
  - When training data is required vs optional.
  - How to fix common issues (e.g., no labeled rows).

## Required Context

- Path A and Path B flows are implemented (Phases 2–4).
- App state exposes enough information to derive readiness (ModelSet, data, etc.).

## Tasks

1. **Implement the readiness strip**

   - At the very top of the Train pane (above Quick Start), add a slim horizontal strip showing:
     - `Active ModelSet: (none / name)` — with ✅ or ⭕.
     - `Models:` ✅/⭕
     - `Rules:` ✅/⭕
     - `Vector store:` ✅/⭕
     - `Training data loaded:` ✅/⭕ (and optionally the count of labeled rows).
   - Use existing app state / selectors to compute these flags.
   - Keep the strip compact and unobtrusive but visible.

2. **Align copy for Path A and Path B**

   - Path A:
     - Ensure text explicitly says that training data is not required to classify once a ModelSet is loaded.
     - Example:
       - `Training data is only needed if you want to validate models, not to classify with an existing ModelSet.`
   - Path B:
     - Ensure each step card has a one-line description at the top explaining its role.
     - Example: For Step 2:
       - `Step 2 loads labeled examples so the system can learn how to predict risk level and department.`

3. **Clarify common error / empty states**

   - When training data is loaded but no labeled rows:
     - Provide actionable instructions:
       - `We found data rows but no labels in columns F and G starting at row 5. Please ensure the file has labeled rows with valid values.`
   - When a ModelSet has no models:
     - Path A text:
       - `This ModelSet has no trained models. You can still use rules (if present) or switch to the Build/Update workflow to train models.`
   - When nothing is active:
     - Top-level message:
       - `No active ModelSet. Load one to classify, or switch to the Build/Update workflow to create one.`

4. **Review and normalize button labels**

   - Ensure consistent naming:
     - "Load training data" vs "Load validation dataset" vs "Import & load .sgm"
   - Ensure actions clearly state what they operate on:
     - `Train models` vs `Build vector store` vs `Save snapshot`.

5. **Tooltips or help icons (optional)**

   - For more complex concepts like "Vector store", add an info icon or tooltip describing:
     - `Vector store enables nearest-neighbor style similarity and can help improve classification by comparing new text to known examples.`

## Tests

- Manual:
  - Confirm readiness strip updates when:
    - ModelSet is loaded.
    - Models are trained.
    - Rules are changed/saved.
    - Vector store is built.
    - Training data is loaded/unloaded.
  - Read through all Train pane text and confirm it matches the intended workflows, without implying training is required for classification with existing ModelSets.

## Success Checklist

- [ ] Readiness strip is visible and accurately reflects app state.
- [ ] Path A copy clearly communicates classification without training data.
- [ ] Path B step descriptions are present and helpful.
- [ ] Error and empty states provide concrete next steps to fix problems.
- [ ] Button labels are consistent and unambiguous.
