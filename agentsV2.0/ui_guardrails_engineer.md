# ui_guardrails_engineer.md

## Purpose
Implement predictable gating/guardrails so users can’t click into dead ends, and always understand why an action is blocked.

## When to run
Phase 2 (required).

## Inputs
- `UX_STATE_MAP.md`
- Current Train/Classify/Export handlers

## Outputs (files/artifacts)
- Central gating functions (e.g., `canTrain`, `canClassify`, `canExport`)
- Disabled states + tooltips and/or inline hints
- Optional Advanced 'Train anyway' with warning

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
1) Implement gating functions:
   - `canTrain(appState)` (labels threshold; training not already running)
   - `canClassify(appState)` (active model exists; file selected)
   - `canExport(appState)` (results exist)

2) Apply gating consistently:
   - Disable button when `canX` is false
   - Provide tooltip with exact reason
   - Provide inline hint with link/CTA to next step when appropriate

3) Avoid over-blocking:
   - If label threshold blocks training, optionally add:
     Advanced → 'Train anyway' requiring confirm + warning.

4) Ensure gating is derived from centralized facts/state.
   - Do not duplicate gating logic across multiple widgets.

## Testing
Manual:
- Confirm Train disabled when labels=0 and tooltip explains.
- Confirm Classify disabled when no model and tooltip explains.
- Confirm Export disabled when no results.

Suggested automated:
- Unit tests for gating functions across representative app states.

## Success checklist (must complete)
- [ ] Central gating functions implemented
- [ ] Buttons correctly disabled/enabled across states
- [ ] Tooltips/hints explain blocked actions
- [ ] No dead-end button clicks remain


