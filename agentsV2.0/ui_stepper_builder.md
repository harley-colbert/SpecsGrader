# ui_stepper_builder.md

## Purpose
Implement the 5-step workflow stepper and wire it to derived UX state so the UI always shows 'what to do next.'

## When to run
Phase 1 (required). Re-run if stepper becomes inconsistent with UX state map.

## Inputs
- `01_SHARED/UX_STATE_MAP.md`
- Current UI layout code for right-side Results pane

## Outputs (files/artifacts)
- A persistent stepper UI in the Results header
- Central functions:
  - `deriveUxState(appState)`
  - `deriveStepperStatus(uxState, appState)`

## Agent Operating Rules (do not skip)

- Make changes incrementally and test frequently.
- Prefer small, reviewable commits (if version control is available).
- Do not introduce new “mystery knobs.” If you add settings, explain them in UI copy.
- Avoid scattering business rules across widgets; centralize:
  - UX state derivation
  - gating rules
  - string/copy constants
- If you cannot determine the UI stack quickly, search the repo for:
  - `main.py`, `app.py`, `__main__`
  - `Tk()`, `QMainWindow`, `App()`, `createRoot`, `ReactDOM`


## Procedure
1) Implement `deriveUxState(appState)`
   - Use the entry conditions from `UX_STATE_MAP.md`.
   - Keep it in a central place (not inside widgets).

2) Define step list and statuses
   - Import Data, Label/Review, Train Model, Classify Document, Export
   - Each step has a status: not_started / in_progress / done / needs_attention

3) Render stepper at top of Results pane
   - Always visible across tabs.
   - Include a dynamic helper line beneath it (1 sentence).

4) Wire stepper to state changes
   - When project/model/labels/results change, stepper updates.

5) Add lightweight interaction (optional but useful)
   - Clicking a step may navigate to relevant tab/panel, but do not rely on that for clarity.

## Testing
Manual:
- Verify stepper appears on all tabs.
- Verify statuses change correctly in:
  - fresh launch (no project)
  - project loaded (no labels)
  - labels exist (no model)
  - model active
  - classification done with uncertain > 0

Suggested automated:
- Unit test `deriveUxState` and `deriveStepperStatus` for representative states.

## Success checklist (must complete)
- [ ] Stepper visible across Results tabs
- [ ] Statuses match `UX_STATE_MAP.md`
- [ ] Helper line changes based on state
- [ ] State derivation centralized (not scattered)

## Implementation notes by stack
- Tkinter: stepper can be a Frame with Labels; icons via Unicode or small images.
- Qt: use a horizontal layout with QLabels/QToolButtons.
- React: a Stepper component with derived props from global state.
