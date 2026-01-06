# MethodPrediction schema (authoritative)

Each enabled method (rules / vector / llm) returns a **MethodPrediction** object.
Fields may be null when the method abstains.

```json
{
  "method": "vector|llm|rules",
  "dept_pred": "mechanical|electrical|controls|project_management|null",
  "dept_conf": 0.0,
  "level_pred": "none|low|medium|high|extreme|null",
  "level_conf": 0.0,
  "evidence": { }
}
```

Notes:
- `dept_conf` and `level_conf` must be in [0, 1].
- If a method abstains for dept/level, it must set the corresponding `*_pred` to null and `*_conf` to 0.
- Evidence should be JSON-serializable and safe to include in `methods_used`.

Evidence guidance:
- rules: matched keyword list, counts, tie info
- vector: top neighbors (truncated), similarity distribution, k, index metadata
- llm: model name, parsing status, optional reason (hidden by default in UI, but may be stored)
