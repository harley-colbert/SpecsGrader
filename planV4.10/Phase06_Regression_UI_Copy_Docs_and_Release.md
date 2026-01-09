# Phase06 – Regression, UI Copy, Docs, and Release

Date: 2026-01-09

## Goal

Finalize v4.10 by running regression, updating UI/docs, and producing the release zip.

## Overview

This phase ensures the new column contract is end-to-end correct, documented, and shippable.



## Implementation steps (follow exactly)

1. **OrchestratorAgent**
   - Run full regression test suite.
   - Confirm no existing workflows are broken (train, load modelset, classify).
2. **FrontendAgent**
   - Update all remaining labels/tooltips/help text referencing old columns.
   - Add a single “Excel Mapping” help section in Train and Classify:
     - D=input spec, E=specific risk (medium+ only), F=risk level, G=department.
3. **BackendAgent**
   - Confirm exporter preserves workbook structure:
     - Existing sheets remain.
     - Only D/E/F/G cells are updated per rules.
     - No accidental data loss in other columns.
4. **TestAgent**
   - Add best-effort end-to-end test:
     - create/load workbook → classify → re-open workbook → validate D/E/F/G
5. **ReleaseAgent**
   - Bump version to **4.10** wherever version is defined.
   - Update CHANGELOG / release notes:
     - Document new Excel contract and overwrite rules.
   - Build:
     - `SpecsGraderv4.10.zip`
   - Smoke test from a clean unzip:
     - app starts
     - import xlsx works
     - classify updates F/G and E per rules


## Tests to create or update in this phase

Create/Update:
- `tests/test_end_to_end_xlsx_contract_v410.py` (best-effort)
- Update README / docs for new column mapping
- Update any golden fixtures if your project uses them

## Tests that must pass (gate)

- `pytest`
- Any frontend build/test commands (if present)
- Manual smoke from clean unzip

## Success checklist (must be ✅ before moving on)

- ✅ All tests pass
- ✅ UI text and docs consistently describe the D/E/F/G contract
- ✅ `SpecsGraderv4.10.zip` created and smoke-tested from clean unzip
- ✅ No regressions in loading modelsets and classifying
