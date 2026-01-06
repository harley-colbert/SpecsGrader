# Model Bundle Spec (authoritative)

A model bundle is a folder under `bundles/` with a stable ID and metadata.
Required top-level files (must exist):

- `rules_config.json`
- `vector_embedder.json`
- `llm_prompt.json`
- `bundle_meta.json`
- `vector_store/` (directory)

Recommended additional artifacts (can be in subfolders):
- `models/level_model.joblib`
- `models/dept_model.joblib`
- `models/level_calibration.joblib` (if separate)
- `models/dept_calibration.joblib`
- `vector_store/embed_level_clf.joblib` (optional)
- `vector_store/embed_dept_clf.joblib` (optional)

## bundle_meta.json (minimum fields)
```json
{
  "bundle_id": "string",
  "name": "string",
  "created_at": "ISO-8601 string",
  "trained_on_rows": 123,
  "data_fingerprint": "sha256 or similar",
  "label_distributions": {
    "risk_level": {"none":0,"low":0,"medium":0,"high":0,"extreme":0},
    "department": {"mechanical":0,"electrical":0,"controls":0,"project_management":0}
  },
  "metrics": {
    "risk_level": {"macro_f1":0.0,"balanced_accuracy":0.0,"per_class_recall":{}},
    "department": {"macro_f1":0.0,"balanced_accuracy":0.0,"per_class_recall":{}}
  }
}
```

## Never-send mode
Never-send is runtime state (AppState/settings), not a bundle attribute.
However, bundles must be compatible with never-send mode (i.e., no required external dependency).
