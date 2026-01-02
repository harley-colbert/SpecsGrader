# Phase 07 — UI Controls and Evidence Display

## Agents
- Run: **AGENT_05_UI_ENGINEER**
- Optional assist: **AGENT_04_INFERENCE_PIPELINE_ENGINEER**

## Goal
Expose vector DB similarity in the UI and improve report readability:
- Toggle “Use Vector DB Similarity”
- Choose top‑K (default 5)
- Choose similarity threshold (default 0.55)
- Display:
  - `Needs Review`
  - top‑K evidence summary (first match + similarity at minimum)

## Files to Modify
- `ui.py`
- Optional: `logic.py` (if you need parameter plumbing changes)

## Implementation Steps
1. Add UI widgets
   - Checkbox: `Use Vector DB Similarity`
   - SpinBox: `Top K` (min 1, max 20, default 5)
   - DoubleSpinBox/Slider: `Similarity Threshold` (0.0–1.0 default 0.55)

2. Wire parameters through the classify action
   - Ensure the values are passed into `multipass_classify(...)`

3. Display evidence
   - Add columns to the output table:
     - `Needs Review`
     - `Top Similarity`
     - `Top Match (Preview)` — truncate to ~120 chars
     - `Top-K Evidence (JSON)` — optional expandable view

4. Ensure graceful behavior when no vector DB is loaded
   - Disable K/threshold controls when similarity is off or DB missing, OR leave enabled but show warnings.

## Must-Pass Tests (no errors)
- `python -m compileall .`
- `python splash.py`
  - UI must open
  - toggles must not crash when clicked
- Run inference via UI on `tests/data/infer_min.csv` (manual step):
  - results appear in table
  - evidence columns populate when similarity enabled

## Success Checklist
- [ ] UI has similarity toggle, K, and threshold controls
- [ ] Changing controls affects inference output
- [ ] No UI crashes if vector DB is missing
