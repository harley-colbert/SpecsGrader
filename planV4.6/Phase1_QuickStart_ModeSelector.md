# Phase 1 — Quick Start Mode Selector

## Objectives

- Introduce a **Quick Start** section at the top of the Train pane.
- Allow the user to choose between:
  - "Use an existing ModelSet and classify" (Path A).
  - "Build / update a ModelSet" (Path B).
- Wire this selection into component state so that:
  - Path A UI is shown by default.
  - Path B UI is collapsed or visually secondary until selected.

## Required Context

- Locate the Train pane component in the frontend codebase.
  - Likely under `frontend/src/ui/...` (e.g., `TrainPane.tsx` or similar).
  - Confirm the root component that renders all Train-related sections.

## Tasks

1. **Locate the Train pane component**

   - Use project search to find the existing Train UI:
     - Search for strings like `"Train"` (pane title), `"Sanity check"`, or existing Train-specific labels.
   - Identify the single React component (or equivalent) that serves as the main container for the Train tab.

2. **Define an internal mode state**

   - Add a state variable for the current workflow mode (e.g., `"use-existing"` vs `"build-update"`).
   - Set the **default** mode to `"use-existing"` (Path A).
   - Expose setter functions or callbacks as needed if children components require mode awareness.

3. **Introduce the Quick Start section at the top**

   - At the top of the Train pane, render a card-like component titled:
     - `What do you want to do?`
   - Inside, render two selectable options (radio-card style):
     - Option A: "Use an existing ModelSet and classify"
       - Description: "Load a saved .sgm (or pick a local ModelSet version) and go straight to Classify."
     - Option B: "Build / update a ModelSet (advanced)"
       - Description: "Load labeled training data, train models, validate, build vector store, then save a new version."
   - Each option should:
     - Be visually obvious when selected (highlighted card or radio).
     - Update the mode state when clicked.
     - Optionally have a small "Continue" button inside to confirm selection.

4. **Hook mode state into layout**

   - Conditions:
     - When `mode === "use-existing"`:
       - Path A sections (Load ModelSet & classify) should be visible.
       - Path B sections (training stepper) should be hidden or visually collapsed.
     - When `mode === "build-update"`:
       - Path B sections should be visible.
       - Path A content should be collapsed or minimized.
   - This phase does not yet change what Path A or Path B contains; it only adds the mode switch and basic conditional rendering.

5. **Styling and UX**

   - Ensure Quick Start is clearly separate from other content (card or panel with padding).
   - Ensure the Quick Start section remains at the top of the Train pane, above all other sections.

## Tests

- Manual:
  - Open Train pane.
  - Confirm a Quick Start card appears at the top with the two options.
  - Verify that:
    - By default, "Use an existing ModelSet and classify" is selected.
    - Selecting "Build / update a ModelSet" toggles the mode state.
  - Confirm that when modes are toggled:
    - Only the corresponding sections of the page are visible or emphasized (even if they still look like v4.5 content for now).

- Automated:
  - Existing tests should still pass (`npm test` if present, and `pytest -q`).
  - No new frontend test is strictly required in this phase, but if the project has a convention, add a simple component rendering test that verifies the Quick Start section renders without throwing.

## Success Checklist

- [ ] Train pane shows a Quick Start card at the top.
- [ ] User can select between "Use existing ModelSet" and "Build / update ModelSet".
- [ ] Mode selection affects which high-level sections are visible/emphasized.
- [ ] Existing tests still pass (backend + any frontend tests).
