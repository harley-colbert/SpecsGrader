# Agent_ResultsUX

## Purpose
Implement results table UX: editable predictions, overrides persistence, corrections dataset auto-save, and add-to-training workflow.

## Responsibilities
- Results table with filters and paging
- Inline edit of dept and level
- Persist overrides in backend + corrections dataset (SQLite preferred)
- Add-to-training button exports training-ready data

## Inputs
- results row schema
- corrections storage requirements
- export schema

## Outputs
- Results endpoints
- Frontend Results pane
- CorrectionsService + tests

## Operating procedure (step-by-step)
1) Implement results row retrieval endpoint with paging.
2) Implement apply_edits endpoint; validate enums.
3) Write edits to corrections dataset immediately (append-only).
4) Update AppState overrides so UI reflects saved edits.
5) Implement add-to-training export routine (documented format).
6) Implement Results pane UI: table, edit dropdowns, save indicator.
7) Add tests for persistence across restarts.

## Tests / validation owned by this agent
- API tests for apply_edits and persistence
- Manual UX: edit 10 rows, refresh app, verify persisted

## Definition of done
- [ ] Edits persist reliably
- [ ] Corrections dataset updated automatically
- [ ] Add-to-training output usable by Train pane
