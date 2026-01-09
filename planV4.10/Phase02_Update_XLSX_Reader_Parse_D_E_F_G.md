# Phase02 – Update XLSX Reader (Parse D/E/F/G)

Date: 2026-01-09

## Goal

Update Excel ingestion to read specification text from D and (optionally) existing risk/labels from E/F/G.

## Overview

This phase aligns import behavior with the new workbook contract without changing training or classification yet.



## Implementation steps (follow exactly)

1. **BackendAgent**
   - Update the XLSX ingestion/reader so it parses the new columns per row:
     - `spec_text = cell(D)`
     - `specific_risk_existing = cell(E)` (optional; may be blank)
     - `risk_level_existing = cell(F)` (optional; may be blank)
     - `dept_existing = cell(G)` (optional; may be blank)
   - Row skipping rule:
     - Skip rows where Column D is empty or whitespace-only.
   - Normalization:
     - Trim string fields.
     - Normalize risk level and dept to canonical values if present.
   - Robust handling:
     - If a workbook is missing columns beyond D, treat E/F/G as absent (no crash).
     - If cells contain formulas, read computed values if supported by your loader.
2. **FrontendAgent**
   - If the UI previews imported rows, ensure it displays:
     - Spec text from D
     - Existing E/F/G values if present
3. **TestAgent**
   - Create a small XLSX fixture with at least 6 rows:
     - Some valid specs in D
     - Some blank D rows
     - Some rows with existing F/G filled
     - Some with E filled
   - Add tests verifying the parsed output matches expected values and skip logic works.
4. **QAAgent**
   - Manual import test with a real-ish workbook to confirm no crashes and correct parsing.


## Tests to create or update in this phase

Create/Update:
- `tests/fixtures/xlsx/contract_v410_input.xlsx`
- `tests/test_xlsx_reader_contract_v410.py`
  - test reads D/E/F/G correctly
  - test skips blank D rows
  - test normalization (e.g., " Medium " → "medium") if applicable

## Tests that must pass (gate)

- `pytest`
- Any existing CLI or smoke test scripts (if present)

## Success checklist (must be ✅ before moving on)

- ✅ XLSX reader uses the new D/E/F/G mapping
- ✅ Blank D rows are skipped
- ✅ Reader does not crash on missing optional columns or empty cells
- ✅ Automated tests validate parsing behavior with an XLSX fixture
