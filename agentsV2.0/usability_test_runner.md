# usability_test_runner.md

## Purpose
Run the 10-minute usability script and record PASS/FAIL plus observations; ensures the workflow is obvious to non-experts.

## When to run
Phase 7 release gate (required).

## Inputs
- `planV2.0/01_SHARED/USABILITY_TEST_SCRIPT.md`
- A fresh user data state or test project

## Outputs (files/artifacts)
- `02_ASSETS/phase7_usability_results.md`

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
1) Reset to a clean starting state (new user).
2) Run each task in the usability script without using internal knowledge.
3) Record:
   - time to first click
   - where confusion occurred
   - any dead ends encountered
4) Assign PASS/SOFT PASS/FAIL.
5) If FAIL:
   - stop and fix the issue
   - rerun the relevant task(s) until PASS/SOFT PASS.

## Testing
Manual:
- Execute the full usability script.
- Confirm the user can answer:
  1) What model is active?
  2) What to do with many Uncertain items?

## Success checklist (must complete)
- [ ] Usability results recorded
- [ ] No FAIL outcomes remain
- [ ] Core loop achievable without guidance


