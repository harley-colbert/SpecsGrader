# export_schema_engineer.md

## Purpose
Define and implement export schemas for each preset, ensuring model metadata and required fields are present and stable.

## When to run
Phase 6 (required).

## Inputs
- `01_SHARED/EXPORT_PRESETS.md` (to be finalized)
- Results + model registry schema

## Outputs (files/artifacts)
- `01_SHARED/EXPORT_PRESETS.md` completed
- Export schema code for presets
- Validation tests (recommended)

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
1) Define presets:
   - SpecGrader Standard
   - Customer Review
   - Internal Engineering

2) For each preset define columns/fields:
   - required: label/type, snippet/text, source, confidence (optional per preset)
   - required metadata: active_model_name/version, trained_at, project id/name

3) Implement schema mapping in code:
   - stable ordering of fields
   - consistent naming

4) Validate output:
   - CSV columns match preset spec
   - JSON keys match spec

5) Update `EXPORT_PRESETS.md` to match actual behavior.

## Testing
Manual:
- Export using each preset; verify columns/fields.

Suggested automated:
- Unit test per preset verifying required keys exist and parse succeeds.

## Success checklist (must complete)
- [ ] `EXPORT_PRESETS.md` completed and accurate
- [ ] Each preset produces expected fields
- [ ] Metadata always present
- [ ] Outputs parse correctly (CSV/JSON)


