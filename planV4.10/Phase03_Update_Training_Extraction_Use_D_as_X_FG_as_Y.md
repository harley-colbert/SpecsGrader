# Phase03 – Update Training Extraction (D→F/G)

Date: 2026-01-09

## Goal

Train models using Column D as the input text and Columns F/G as the supervised labels.

## Overview

This phase ensures your training pipeline matches the new workbook schema and supports best-practice local supervised learning.



## Implementation steps (follow exactly)

1. **MLAgent + BackendAgent**
   - Update the training-data extraction pipeline:
     - **X (input text)** must come from Column D only.
     - **y_level (risk label)** must come from Column F.
     - **y_dept (department label)** must come from Column G.
   - Column E is **not** used as a training label; treat it as optional human risk notes.
2. Add explicit training row validity rules:
   - A row is usable for training risk model only if:
     - D has text AND F is a valid risk label.
   - A row is usable for training dept model only if:
     - D has text AND G is a valid department label.
   - A row can train both models if both labels are valid.
3. **FrontendAgent**
   - Update any dataset health panels to compute distributions based on:
     - F for risk labels
     - G for dept labels
4. **TestAgent**
   - Add tests that:
     - Only valid labeled rows are included in training sets.
     - Invalid labels are excluded (and reported if you have reporting).
   - Prefer deterministic synthetic dataframes created in-test.


## Tests to create or update in this phase

Create/Update:
- `tests/test_training_extraction_contract_v410.py`
  - test training X comes from D
  - test y_level comes from F
  - test y_dept comes from G
  - test invalid labels are filtered out
- Optional: add a small CSV fixture under `tests/fixtures/` if your pipeline expects file IO

## Tests that must pass (gate)

- `pytest`

## Success checklist (must be ✅ before moving on)

- ✅ Training extraction uses D as X and F/G as labels
- ✅ Invalid labels do not silently enter training
- ✅ Dataset health/validation views are consistent with the new mapping
- ✅ Tests cover label validity filtering
