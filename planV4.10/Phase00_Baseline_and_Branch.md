# Phase00 – Baseline and Branch

Date: 2026-01-09

## Goal

Establish a clean baseline from the current codebase and create the v4.10 working branch.

## Overview

This phase prevents regressions by capturing baseline behavior and ensuring tests are green before changes.



## Implementation steps (follow exactly)

1. **OrchestratorAgent**
   - Unzip the starting codebase (SpecsGrader v4.9 or current working branch).
   - Create a new branch: `upgrade/v4.10-xlsx-contract`.
   - Run the app to confirm baseline behavior.
2. **QAAgent**
   - Capture baseline screenshots for:
     - Train pane (Path A + Path B if present)
     - Classify pane (Excel upload → classify → download)
   - Record baseline behavior of how the app currently reads/writes Excel columns.
3. **TestAgent**
   - Run all existing tests; record the command list and results in a Phase00 completion report.
4. Create `workspace/reports/Phase00_Completion.md` including:
   - baseline test run output (summary)
   - baseline screenshots list (filenames/paths)
   - any known issues encountered


## Tests to create or update in this phase

- No new tests required unless tests do not exist.
- If there are no XLSX tests currently, create a placeholder test module:
  - `tests/test_xlsx_contract_placeholder.py`
  - It should `assert True` and include a TODO comment referencing Phase02.

## Tests that must pass (gate)

- Existing backend test suite (typically `pytest`)
- Any existing frontend test/build checks (if present)
- App starts without crashing

## Success checklist (must be ✅ before moving on)

- ✅ Branch created and baseline confirmed
- ✅ Baseline screenshots captured
- ✅ Existing tests executed and results recorded
- ✅ Phase00 completion report created
