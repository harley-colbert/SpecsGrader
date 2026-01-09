# Phase05 – Derive Specific Risk (Write E for Medium+)

Date: 2026-01-09

## Goal

Auto-generate Column E (Specific Risk) only when risk level (F) is Medium or higher, based on D+F+G.

## Overview

This phase implements best-practice risk-note generation: deterministic, local-first, explainable, and gated by risk level.



## Implementation steps (follow exactly)

1. **MLAgent + BackendAgent**
   - Implement deterministic, local-first generation for **Specific Risk (Column E)**.
   - Gating rule (must use `is_medium_plus(F)`):
     - If F is none/low → E must be blank (or cleared) for that row.
     - If F is medium/high/extreme → E is derived from D + F + G.
2. Implement:
   - `generate_specific_risk(spec_text, risk_level, dept) -> str`
   Requirements:
   - Deterministic (same inputs → same output).
   - Offline/local (no network calls).
   - Template-based text keyed by dept and risk level.
   - Optional “risk hooks” from D using a simple lexicon (hazard terms, units, compliance terms).
3. Overwrite policy for E (choose and implement; must be documented and tested):
   - Recommended default:
     - Only write E when medium+ AND (E is blank OR `overwrite_specific_risk` is True).
4. **FrontendAgent**
   - Update UI/help text:
     - E is auto-generated only for medium+.
     - If overwrite toggle exists, expose as “Advanced”.
5. **TestAgent**
   - Unit tests:
     - none/low => ""
     - medium/high/extreme => non-empty
     - dept changes wording
     - deterministic output
   - XLSX integration tests:
     - E filled only for medium+ rows
     - E blank for none/low
     - overwrite policy honored
6. **QAAgent**
   - Manual check with a workbook containing mixed predicted risk levels.


## Tests to create or update in this phase

Create/Update:
- `tests/test_specific_risk_generator.py`
- `tests/test_xlsx_writes_E_for_medium_plus.py`
- `tests/fixtures/xlsx/contract_v410_specific_risk_in.xlsx`

## Tests that must pass (gate)

- `pytest`
- Manual UI smoke: verify E behavior in downloaded workbook

## Success checklist (must be ✅ before moving on)

- ✅ Column E is derived only when F is medium/high/extreme
- ✅ Column E is blank when F is none/low
- ✅ Generator is deterministic and offline
- ✅ Overwrite policy for E is implemented and tested
- ✅ XLSX tests validate E behavior
