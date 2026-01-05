# ui_navigation_designer.md

## Purpose
Make key navigation points discoverable (Review tab, header showing active project/model) and align the page structure with the workflow.

## When to run
Phase 1 (alongside stepper).

## Inputs
- Current UI layout
- `UX_STATE_MAP.md`

## Outputs (files/artifacts)
- Review tab present and discoverable
- Results header shows Active Project/Model/Trained date

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
1) Add/Expose a Review tab in the Results area.
   - If Review is not implemented yet, show an informative placeholder and a CTA.

2) Add a small Results header area (above tabs or integrated with stepper) showing:
   - Active Project (or 'None')
   - Active Model (or 'None')
   - Trained date (if model exists)

3) Make Specs/Risks/Uncertain counters clickable and consistent:
   - Specs/Risks filter table
   - Uncertain opens Review (or prompts)

4) Keep navigation lightweight:
   - Avoid creating multiple competing navigation systems.
   - Stepper + tabs + left panel is enough.

## Testing
Manual:
- Review tab is visible.
- Active Model header changes when switching models.
- Clicking Uncertain reliably opens Review.

Suggested automated:
- UI snapshot test for header content under different states.

## Success checklist (must complete)
- [ ] Review tab exists and is reachable in one click
- [ ] Active Project/Model header is always visible and correct
- [ ] Counters are clickable and consistent


