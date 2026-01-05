# ui_regression_tester.md

## Purpose
Prevent regressions by running a repeatable checklist after each phase. Produces a short regression report with PASS/FAIL and notes.

## When to run
Run after each phase (0–7) before proceeding.

## Inputs
- Ability to run the app
- Any existing automated tests
- A sample dataset/project if available

## Outputs (files/artifacts)
- `02_ASSETS/phaseX_regression_results.md` (where X is current phase)

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
1) Identify what “core loop” means in this codebase:
   - Train
   - Classify
   - View logs
   - Export/save (if present)

2) Run the regression checklist:
   - App launches without errors
   - Left panel renders correctly
   - Results tabs render (Table/Details/Log/Stats + Review when added)
   - Training workflow still runs (or at least starts and logs correctly)
   - Classification workflow still runs and produces output
   - Logs tab shows meaningful content
   - Export still works (or remains correctly disabled when no results)

3) Record results as:
   - PASS / FAIL per item
   - Steps to reproduce for any failure
   - Screenshots for failures (optional)

4) If any FAIL occurs:
   - stop and fix regression
   - re-run checklist until all PASS

## Testing
Manual (minimum):
- Launch app
- Train (or confirm Train is gated with correct message)
- Classify a known doc
- Open Log tab
- Export (or confirm it’s correctly gated)

Suggested automated:
- Add a smoke test script that can run headlessly if your stack supports it.

## Success checklist (must complete)
- [ ] Regression report created for current phase
- [ ] No FAIL items remain
- [ ] Any new UI states are covered by tests/checklist

## Recommendation
Keep a tiny 'sample project' checked into a `tests/fixtures/` folder so regressions are easy to reproduce.
