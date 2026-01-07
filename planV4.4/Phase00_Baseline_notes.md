# Phase 0 baseline notes (local run)

## Dataset + row count
- Training dataset: `tests/fixtures/training_sample.csv` (2 labeled rows).
- Classification dataset: `workspace/classify_from_training.csv` (same 2 risk_text values as training, labels removed).

## Methods enabled
- Classification run used only rules (`enabled_methods`: rules=true, vector=false, llm=false).

## Observed mismatches
- Training completed successfully and produced model artifacts.
- Classification on the same texts returned `pred_level=None` and `pred_dept=None` for both rows (rule-based abstain), which does **not** match the training labels (`high/mechanical`, `low/controls`).

## Browser console errors
- Not applicable (TestClient API run; no browser session opened).

## Baseline code path findings
- Training writes `workspace/workspace_bundle/level_model.joblib` + `dept_model.joblib`.
- Classification uses only rules/vector/LLM aggregation and does not call the trained model pipelines.
