# ui_copy_writer.md

## Purpose
Create and apply a consistent UI copy pack (labels, tooltips, empty states, warnings, toasts) that makes the workflow obvious to new users.

## When to run
Phase 2 (required). Re-run whenever new features add new terminology.

## Inputs
- Approved microcopy direction (from chat)
- `UX_STATE_MAP.md`
- Current UI string locations

## Outputs (files/artifacts)
- Centralized UI string dictionary/constants
- Updated labels and tooltips across the UI
- Adaptive empty states copy implemented

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
1) Create a UI string dictionary/constants module (or equivalent).
   - Define labels, tooltips, empty state content, warnings, toasts.

2) Apply key renames:
   - Project / Model Set → Project & Active Model
   - Train → Train Model
   - Classify (Multipass) → Classify Document (+ Advanced)
   - Save Results to CSV → Export…

3) Implement adaptive empty states:
   - Headline + 1–2 lines + primary CTA
   - Must be driven by `deriveUxState`

4) Add tooltips for ambiguous concepts:
   - Spec, Risk, Uncertain
   - Active model
   - Why a button is disabled

5) Add toasts/snackbars for key completions:
   - training completed
   - classification completed
   - export completed

## Testing
Manual:
- Scan UI for inconsistent terminology.
- Trigger each empty state and confirm it shows the correct primary CTA.
- Hover disabled actions and confirm tooltip explains why.

Suggested automated:
- Lint/test ensuring no raw strings remain in UI (optional).

## Success checklist (must complete)
- [ ] Strings centralized
- [ ] Key renames applied consistently
- [ ] Empty states are adaptive and match state map
- [ ] Tooltips explain Uncertain and disabled actions

## Practical tip
If full localization is overkill, still centralize strings in one file to prevent drift.
