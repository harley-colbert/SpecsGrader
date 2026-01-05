# results_table_designer.md

## Purpose
Redesign the Results Table to be actionable (type/confidence/snippet/source/actions) and consistent with review/persistence policy.

## When to run
Phase 5 (required).

## Inputs
- Current results table implementation
- Classification output schema
- Label persistence hooks

## Outputs (files/artifacts)
- Updated Results Table columns and actions
- Filters hooked to Specs/Risks/Uncertain chips

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
1) Implement columns:
   - Type (Spec/Risk/Uncertain)
   - Confidence
   - Snippet
   - Source
   - Action

2) Implement actions:
   - Accept (writes label if policy allows)
   - Correct… (choose label and persist)
   - Send to Review
   - Ignore

3) Ensure table filtering works:
   - clicking chips filters visible items

4) Ensure counts update after user actions.

5) Avoid clutter:
   - default view shows essentials; advanced fields in Details.

## Testing
Manual:
- Table shows new columns.
- Actions work and persist.
- Filtering works via chips.

Suggested automated:
- Unit tests for action handlers and filter state updates.

## Success checklist (must complete)
- [ ] New table columns present
- [ ] Row actions implemented and persistent
- [ ] Chips filter table correctly


