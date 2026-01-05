# label_persistence_engineer.md

## Purpose
Implement persistence for labels created via Review Queue and Table/Details actions, following a documented policy (immediate or staged).

## When to run
Phase 3 and Phase 5 (required).

## Inputs
- Project storage mechanism (files, sqlite, jsonl, etc.)
- `01_SHARED/LABELING_POLICY.md`

## Outputs (files/artifacts)
- Updated `01_SHARED/LABELING_POLICY.md` (filled, matches implementation)
- Persistence code that writes/reads labels reliably
- Optional: migration script if schema changes

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
1) Decide storage schema
   - Minimum fields:
     id, text, source_doc, source_section/page, label, confidence_at_time, created_at, updated_at, user_note(optional)

2) Implement write path(s)
   - from Review Queue actions
   - from Table actions (Accept/Correct)
   - from Details edits

3) Implement read path(s)
   - load label counts for Train panel
   - load labels for model training input

4) Handle duplicates
   - define whether duplicates overwrite, merge, or create new revision
   - document in `LABELING_POLICY.md`

5) Add durability checks
   - labels persist across restart
   - partial writes do not corrupt dataset (atomic write pattern)

6) Update label counts in UI immediately after save (or after staging commit).

## Testing
Manual:
- Create labels via Review and via Table.
- Restart app and confirm labels remain.
- Confirm label counts update in Train panel.

Suggested automated:
- Unit test: write then read returns same labels.
- If sqlite: test transaction behavior and schema versioning.

## Success checklist (must complete)
- [ ] `LABELING_POLICY.md` filled and accurate
- [ ] Labels persist across restart
- [ ] Duplicate handling defined and implemented
- [ ] Label counts reliably update UI

## Atomic write guidance
If using JSON/CSV, write to a temp file then rename to avoid corruption on crash.
