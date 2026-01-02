# AGENT_05_UI_ENGINEER — Similarity Controls + Evidence Display

## Role
Upgrade the PySide6 UI to expose vector similarity controls and show evidence-backed results:
- toggle similarity on/off
- choose top‑K
- set similarity threshold
- display `Needs Review` + similarity evidence columns
- remain stable even if vector DB/models are missing

## Primary Phase
- Phase 07 (required)

## Inputs
- SpecGrader repo
- Updated `logic.multipass_classify(...)` signature from Phase 05/06
- Phase doc: `Phase_07_UI_Controls.md`

## Outputs
### Modified
- `ui.py` (add controls and wire parameters)
- Optional minor changes in `logic.py` only if needed for parameter plumbing

## UI Requirements
1) Add widgets:
- Checkbox: “Use Vector DB Similarity”
- SpinBox: “Top K” (1–20, default 5)
- DoubleSpinBox (or slider): “Similarity Threshold” (0.0–1.0, default 0.55)

2) Wiring:
- When user clicks “Classify” (or equivalent action), pass:
  - `sim_checkbox`
  - `top_k`
  - `similarity_threshold`
  into `logic.multipass_classify(...)`

3) Display:
Ensure the results table includes at least:
- `Needs Review`
- `Top Similarity`
- `Similarity Match` or `Top Match (Preview)`
- optional: `KNN Evidence` (may be large; keep readable)

4) Graceful behavior:
- If no models are loaded, UI must not crash (show message).
- If similarity enabled but vector DB missing, UI must not crash; show warning and continue.

## Manual UX Acceptance Steps (must not crash)
- Run `python splash.py`
- Toggle similarity checkbox on/off repeatedly
- Change K and threshold values
- Load a model set (if available) and run inference on `tests/data/infer_min.csv`
- Verify evidence columns populate when similarity is enabled

## Must-Pass Tests (no errors)
- `python -m compileall .`
- `python splash.py` (UI must open; toggles must not crash)

## Success Checklist
- [ ] UI has similarity toggle, K, threshold controls
- [ ] Controls are passed into `multipass_classify(...)`
- [ ] Results table shows `Needs Review` and similarity/evidence fields
- [ ] UI remains stable if vector DB is missing or similarity disabled
