# details_inspector_builder.md

## Purpose
Build the Details Inspector so selecting an item shows context and provides label decisions; integrate with persistence.

## When to run
Phase 5 (required).

## Inputs
- Details tab implementation
- Ability to fetch context around a snippet (if available)

## Outputs (files/artifacts)
- Details panel that shows:
  predicted label, confidence, snippet, context, user decision actions

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
1) Define selection model:
   - When user selects a table row or review item, store selected item id.

2) Populate Details with:
   - predicted label + confidence
   - full snippet text
   - source fields
   - context (before/after text, page excerpt, etc.)

3) Add decision actions:
   - Mark as Spec
   - Mark as Risk
   - Not Relevant
   - optional note field

4) Persist decisions per `LABELING_POLICY.md`.

5) Ensure Details edits reflect immediately in Table/Review status.

## Testing
Manual:
- Select a row → details shows.
- Change label in details → persists and updates counts.

Suggested automated:
- UI test: selection changes details; action updates model.

## Success checklist (must complete)
- [ ] Details populated when selecting items
- [ ] User decisions persist
- [ ] Table/Review reflect details changes immediately


