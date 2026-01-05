# ui_accessibility_checker.md

## Purpose
Check basic accessibility: focus order, keyboard navigation, readable contrast, and tooltips/labels for icons.

## When to run
Phase 7 (required).

## Inputs
- Current UI
- Ability to navigate with keyboard only

## Outputs (files/artifacts)
- `02_ASSETS/phase7_accessibility_report.md`

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
1) Keyboard-only pass:
   - Can you reach key controls (Import/Review/Train/Classify/Export)?
   - Is focus visible?
   - Does tab order make sense?

2) Review queue pass:
   - Can you complete 10 review actions without mouse? (if shortcuts exist)

3) Contrast and readability:
   - ensure text is readable on light theme
   - icons have tooltips/labels

4) Write findings and fixes in:
   - `02_ASSETS/phase7_accessibility_report.md`

5) Fix issues immediately if they block core workflow.

## Testing
Manual:
- Full keyboard walkthrough (focus order).
- Shortcut validation (if implemented).

Suggested automated:
- None required; manual is acceptable here.

## Success checklist (must complete)
- [ ] Accessibility report written
- [ ] Blocking issues fixed
- [ ] Focus order supports core workflow


