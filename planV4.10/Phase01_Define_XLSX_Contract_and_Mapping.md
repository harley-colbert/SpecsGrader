# Phase01 – Define XLSX Contract and Mapping

Date: 2026-01-09

## Goal

Centralize the new column contract and risk-level gating logic so all features use the same mapping.

## Overview

This phase prevents drift by making column letters and risk gating a single source of truth.

> This phase should not change runtime behavior yet; it only introduces the new mapping module and supporting helpers.

## Implementation steps (follow exactly)

1. **BackendAgent + MLAgent**
   - Identify the current XLSX column mapping location(s) (reader, writer/exporter, training loader).
   - Introduce a single source-of-truth mapping module used everywhere XLSX columns are referenced.
     - Suggested file: `backend/specsgrader/config/xlsx_contract.py` (adjust to your repo structure)
   - Define:
     - `SPEC_TEXT_COL = "D"`
     - `SPECIFIC_RISK_COL = "E"`
     - `RISK_LEVEL_COL = "F"`
     - `DEPT_COL = "G"`
   - Define risk ordering and helpers:
     - `RISK_ORDER = ["none","low","medium","high","extreme"]`
     - `normalize_risk_level(s: str) -> str|None`
     - `is_medium_plus(level: str) -> bool`
2. **FrontendAgent**
   - Add a small read-only “Excel mapping” help snippet in the Train/Classify UI where Excel import is used.
   - The snippet must match the new contract exactly.
3. **TestAgent**
   - Add unit tests for:
     - risk ordering validity
     - `is_medium_plus()` for each enum value
     - normalization of risk level strings (trim/lowercase/synonyms if supported)
4. Update any docs/README referencing old columns to the new D/E/F/G contract.


## Tests to create or update in this phase

Create/Update:
- `tests/test_risk_level_ordering.py`
  - test that ordering matches: none < low < medium < high < extreme
  - test `is_medium_plus` returns True only for medium/high/extreme
- `tests/test_risk_level_normalization.py`
  - test trimming/lowercasing and any synonym mapping you implement

## Tests that must pass (gate)

- `pytest`
- Any lint/format checks already in the repo (if present)

## Success checklist (must be ✅ before moving on)

- ✅ A single mapping module exists and is imported everywhere XLSX columns are referenced
- ✅ Risk level ordering and medium+ logic are explicit and tested
- ✅ UI/help text updated to reflect D/E/F/G mapping exactly
