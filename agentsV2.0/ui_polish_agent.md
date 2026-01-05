# ui_polish_agent.md

## Purpose
Final visual and interaction polish: spacing, hierarchy, empty state aesthetics, button consistency, and small usability improvements.

## When to run
Phase 7 (required).

## Inputs
- Current UI
- Screenshots from baseline and current

## Outputs (files/artifacts)
- Updated UI polish
- Final screenshots in `02_ASSETS/`

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
1) Audit spacing and alignment:
   - consistent padding/margins across panels
   - consistent heading sizes

2) Audit button hierarchy:
   - one primary CTA per panel/empty state
   - secondary actions styled secondary

3) Improve empty state visuals:
   - clear headline
   - short explanation
   - single primary button

4) Ensure tooltips aren’t clipped and text wraps properly.

5) Capture final screenshots:
   - fresh launch
   - no labels
   - ready to train
   - classification results
   - review queue

## Testing
Manual:
- Visual scan for misalignment and clipped text.
- Keyboard focus order sanity check (basic).

Suggested automated:
- Screenshot diff tests if you have infra.

## Success checklist (must complete)
- [ ] Visual hierarchy clear (primary CTA stands out)
- [ ] No clipped text/tooltips
- [ ] Final screenshot set captured


