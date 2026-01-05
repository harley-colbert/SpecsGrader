# ui_inventory_auditor.md

## Purpose
Inventory the current UI (components, labels, empty states, disabled states, and user-visible flows) and write a baseline report.

## When to run
Phase 0 (before any UI changes). Re-run whenever the UI is substantially refactored.

## Inputs
- Current repository codebase
- Ability to run the app (preferred) or at least inspect UI code

## Outputs (files/artifacts)
- `02_ASSETS/phase0_ui_inventory.md`
- Optional screenshots: `02_ASSETS/baseline_*.png`

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
1) Locate UI entry point(s)
   - Find main window/root component and layout file(s).
   - Identify left panel code and right results pane code.

2) Run the app (preferred) and list:
   - all visible panels/sections
   - all buttons and their labels
   - all tabs
   - counters/metrics shown

3) Identify empty states
   - What does the right pane show on first launch?
   - What does Train show without data?
   - What does Classify show without file/model?

4) Identify disabled states (if any)
   - Which controls disable? Under what conditions?

5) Write `02_ASSETS/phase0_ui_inventory.md` with:
   - Component inventory
   - Label inventory (exact strings)
   - Current workflow description (“what the user can do now”)
   - Confusing areas / dead ends observed

## Testing
Manual:
- Launch app and verify inventory is complete vs what’s visible.

Optional:
- If UI has snapshot/export capability, capture baseline screenshot set.

## Success checklist (must complete)
- [ ] `02_ASSETS/phase0_ui_inventory.md` created
- [ ] All current UI labels captured (exact strings)
- [ ] All empty states documented
- [ ] Any “dead-end” interactions noted


