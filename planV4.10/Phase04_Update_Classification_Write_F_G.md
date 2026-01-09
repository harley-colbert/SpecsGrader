# Phase04 – Update Classification Output (Write F/G)

Date: 2026-01-09

## Goal

Write predicted risk level to Column F and predicted department to Column G for each spec in Column D.

## Overview

This phase updates the output contract so downstream processes can rely on F/G as the model outputs.



## Implementation steps (follow exactly)

1. **BackendAgent + MLAgent**
   - Update classification so it is performed from Column D only.
   - For each row with valid D text:
     - Run ensemble classification.
     - Write outputs:
       - Column F = predicted risk level (string)
       - Column G = predicted department (string)
2. Overwrite policy for F/G (choose and implement; must be documented and tested):
   - Recommended default:
     - Always overwrite F/G with model outputs.
   - Alternative:
     - Add option flag `overwrite_predictions: bool` (default True).
3. **FrontendAgent**
   - Update UI labels to reflect:
     - F/G are model outputs
   - If you implement `overwrite_predictions`, expose as an “Advanced” checkbox.
4. **TestAgent**
   - XLSX integration tests:
     - Case A: D filled, F/G blank → after classify, F/G populated.
     - Case B: D filled, F/G prefilled → confirm overwrite policy.
5. **QAAgent**
   - Manual classify test:
     - upload workbook → classify → download → verify F/G updated.


## Tests to create or update in this phase

Create/Update:
- `tests/fixtures/xlsx/contract_v410_classify_in.xlsx`
- `tests/test_xlsx_classify_writes_FG.py`
  - test F/G writes for rows with D text
  - test overwrite policy

## Tests that must pass (gate)

- `pytest`
- Manual UI smoke: classify workbook works end-to-end

## Success checklist (must be ✅ before moving on)

- ✅ Classification uses Column D only
- ✅ Outputs are written to F and G for each classified row
- ✅ Overwrite rules are implemented, documented, and tested
- ✅ Tests confirm workbook round-trip correctness
