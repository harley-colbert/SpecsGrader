# ui_keyboard_shortcuts.md

## Purpose
Add optional keyboard shortcuts to speed up Review Queue labeling (high UX ROI).

## When to run
Phase 3 (recommended).

## Inputs
- Review Queue UI implementation
- UI stack keyboard event handling

## Outputs (files/artifacts)
- Keyboard shortcuts in Review tab (and optionally Details)
- Visible hint text in Review UI footer

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
1) Implement shortcuts:
   - 1 = Mark as Spec
   - 2 = Mark as Risk
   - 3 = Mark as Not Relevant
   - Enter = Save & Next
   - S = Skip

2) Ensure shortcuts only fire when Review tab is focused.

3) Add a small on-screen hint:
   “Tips: 1=Spec, 2=Risk, 3=Not Relevant, Enter=Save & Next”

4) Add a setting toggle if you expect conflicts (optional).

## Testing
Manual:
- Press keys while Review tab focused → correct action triggers.
- Press keys outside Review tab → no unintended actions.

Suggested automated:
- UI test if your stack supports input simulation.

## Success checklist (must complete)
- [ ] Shortcuts implemented and scoped to Review
- [ ] On-screen hint present
- [ ] No conflicts with global shortcuts


