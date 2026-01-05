# export_ux_designer.md

## Purpose
Implement Export UX (dialog + presets + toggles) and replace CSV-only 'save' with a traceable export flow.

## When to run
Phase 6 (required).

## Inputs
- Existing export code (if any)
- Results schema
- Active model registry

## Outputs (files/artifacts)
- Export dialog UI
- Export action in logs
- Export disabled when no results

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
1) Replace button label with “Export…”.
2) Add gating: disabled if no results (tooltip explains).
3) Build dialog:
   - format (CSV/JSON)
   - preset dropdown
   - toggles: confidence/source/context
   - model metadata always included (not disableable)
4) Ensure UX copy matches copy pack.
5) After export, show toast with filename and write log entry.

## Testing
Manual:
- Export disabled with no results.
- Export works with results and produces file.

Suggested automated:
- Unit test: dialog preset selection changes schema.

## Success checklist (must complete)
- [ ] Export dialog exists
- [ ] Presets selectable
- [ ] Model metadata always included
- [ ] Export logged and toast shown


