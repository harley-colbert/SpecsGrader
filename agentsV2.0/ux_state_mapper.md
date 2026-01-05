# ux_state_mapper.md

## Purpose
Define the finite UX workflow states and transitions; specify UI behavior (primary CTA, stepper status, empty state copy, disabled actions) per state.

## When to run
Phase 0 (required). Update if new workflow states are added later.

## Inputs
- `planV2.0/01_SHARED/UX_STATE_MAP_TEMPLATE.md`
- UI inventory report from `02_ASSETS/phase0_ui_inventory.md`
- Understanding of what data is available in app state (project loaded? labels count? model loaded? results?)

## Outputs (files/artifacts)
- `01_SHARED/UX_STATE_MAP.md` (filled, complete)
- Optional: `02_ASSETS/phase0_state_diagram.md`

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
1) Copy the template into `01_SHARED/UX_STATE_MAP.md` if not already present.

2) Enumerate states (minimum set)
   - NO_PROJECT
   - PROJECT_LOADED_NO_LABELS
   - LABELS_EXIST_NO_MODEL
   - MODEL_LOADED_READY_TO_CLASSIFY
   - CLASSIFICATION_DONE_NO_UNCERTAIN
   - CLASSIFICATION_DONE_HAS_UNCERTAIN
   - TRAINING_RUNNING
   - CLASSIFY_RUNNING
   - ERROR_STATE

3) For each state, write:
   - Entry conditions (boolean rules over app state)
   - Primary CTA (single main action)
   - Secondary actions (optional)
   - Disabled actions and user-facing explanation (tooltip/hint)
   - Stepper status (which step is highlighted; which are done/blocked)
   - Results empty-state copy (headline + 1–2 sentences + primary button)
   - Exit conditions (what transitions to other states)

4) Create a transition table
   - From state, trigger, to state, notes

5) Validate against reality
   - Confirm every visible UI situation maps to exactly one state.
   - If ambiguity exists, split states or tighten entry conditions.

## Testing
Manual:
- Walk through typical flows and ensure a single state always applies.
- Spot-check edge cases:
  - project loaded + model loaded but no labels
  - results exist but model missing (should not happen; define behavior)
  - errors during training/classify

## Success checklist (must complete)
- [ ] `01_SHARED/UX_STATE_MAP.md` filled and complete
- [ ] Every state defines: entry conditions, primary CTA, stepper status, empty-state copy
- [ ] Transition table exists
- [ ] No ambiguous “half-states” remain

## Tip
If you struggle to define clean entry conditions, add a small derived summary object like `UxFacts` (labelsCount, hasModel, hasResults, uncertainCount, hasProject) and derive states from that.
