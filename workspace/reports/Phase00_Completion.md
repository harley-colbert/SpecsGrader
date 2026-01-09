# Phase 00 Completion Report (v4.10)

## Summary
- Created branch `upgrade/v4.10-xlsx-contract` and verified the backend/UI loads in headless mode via direct Uvicorn run on `0.0.0.0:7860`.
- Captured baseline UI screenshots for Train and Classify panes.
- Confirmed baseline XLSX ingest behavior: data begins on row 5 with risk text in column E, label level in column F, and label department in column G for training files; classify reads only column E for risk text.

## Key files touched
- `workspace/reports/Phase00_Completion.md`

## Tests run
- `python -m pytest -q` (pass; 58 passed, 1 skipped, 87 warnings)

## UI evidence
- Train pane baseline screenshot: `browser:/tmp/codex_browser_invocations/b24438344314dd42/artifacts/artifacts/train-pane.png`
- Classify pane baseline screenshot: `browser:/tmp/codex_browser_invocations/b24438344314dd42/artifacts/artifacts/classify-pane.png`

## Baseline XLSX behavior
- XLSX/CSV parsing selects the sheet containing “Standards Risk Matrix”, skips four header rows (data starts on row 5), and reads:
  - Column E (index 4): risk text
  - Column F (index 5): label level (training only)
  - Column G (index 6): label department (training only)

## Success checklist
- ✅ Branch created and baseline confirmed. (branch `upgrade/v4.10-xlsx-contract`; Uvicorn run confirmed)
- ✅ Baseline screenshots captured. (see UI evidence paths above)
- ✅ Existing tests executed and results recorded. (`python -m pytest -q`)
- ✅ Phase00 completion report created. (this document)

## Follow-ups
- None.
