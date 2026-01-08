# Agent — Phase 1: Quick Start Mode Selector

You are the Phase 1 agent for the SpecsGrader v4.6 upgrade.

## Goal

- Implement a **Quick Start** mode selector at the top of the Train pane.
- Introduce internal state for:
  - `mode = "use-existing"` (Path A) — default.
  - `mode = "build-update"` (Path B)`.
- Use that mode to conditionally render high-level sections.

## Inputs

- SpecsGraderv4.5 codebase (post-Phase 0 baseline).
- `planV4.6/Phase1_QuickStart_ModeSelector.md` for detailed requirements.

## High-Level Steps

1. Locate the Train pane root component.
2. Add a mode state variable and setter (default `"use-existing"`).
3. Create a Quick Start card UI with two selectable options.
4. Wire the mode state to show/hide Path A vs Path B containers (even if they still look like v4.5).
5. Verify that tests still pass.

## Detailed Instructions

1. **Locate the Train pane root component**
   - Search in `frontend/src` for the current Train UI.
     - Look for strings like `"Train"`, `"Sanity check"`, `"Evaluate"`, `"Vector store"`, etc.
   - Identify the top-level component for the Train tab (e.g. `TrainPane.tsx` or similar).

2. **Add mode state**
   - In the Train pane root component:
     - Introduce a mode state (React example):
       - `const [mode, setMode] = useState<"use-existing" | "build-update">("use-existing");`
     - Ensure this state is available where layout decisions are made (pass as props if necessary).

3. **Render Quick Start section at the top**
   - At the very top of the Train pane JSX, before other content, add a card/panel with:
     - Title: `What do you want to do?`
     - Two “radio-card” options:
       - Option A: **Use an existing ModelSet and classify**
         - Description: `Load a saved .sgm (or pick a local ModelSet version) and go straight to Classify.`
       - Option B: **Build / update a ModelSet (advanced)**
         - Description: `Load labeled training data, train models, validate, build vector store, then save a new version.`
   - When the user selects an option:
     - Update `mode` to `"use-existing"` or `"build-update"`.
   - Make it visually obvious which option is active:
     - e.g., selected card is highlighted or has a checkmark.

4. **Conditionally render Path A / Path B containers**
   - Wrap existing Train content into two logical segments:
     - A container for Path A (existing ModelSet flow).
     - A container for Path B (training flow).
   - For this phase:
     - Do not deeply restructure the internals yet.
     - Only ensure that when `mode === "use-existing"`:
       - Path A container is visible.
       - Path B container is hidden or de-emphasized.
     - When `mode === "build-update"`:
       - Path B container is visible.
       - Path A container is hidden or de-emphasized.

5. **Styling and UX**
   - Ensure the Quick Start card is visually separated (padding/margin).
   - It must remain the **first thing** the user sees on the Train pane.

6. **Run tests**
   - Run frontend tests if they exist (e.g. `npm test` or `npm run test`).
   - Run backend tests to confirm no regressions: `pytest -q`.

7. **Document changes**
   - In `docs/upgrade_v4.6/phase1_summary.md`, describe:
     - The entry point file and component for the Train pane.
     - The new `mode` state and where it is defined.
     - How Quick Start affects which sections are visible.

## Success Checklist

- [ ] Quick Start card appears at the top of the Train pane.
- [ ] `"use-existing"` is the default mode.
- [ ] Selecting the advanced option switches mode to `"build-update"`.
- [ ] Path A vs Path B containers show/hide based on mode.
- [ ] All existing tests (frontend + backend) still pass.
- [ ] `docs/upgrade_v4.6/phase1_summary.md` exists and explains the implementation.
