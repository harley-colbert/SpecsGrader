# Export Presets

## Preset definitions

### 1) SpecGrader Standard
- Format: CSV or JSON
- Fields (order for CSV):
  - `Risk Description` (snippet/text)
  - `Final Risk Level`
  - `Final Review Dept`
  - `Confidence` (Top Similarity or 0)
  - `Label Source`
  - `Needs Review`
  - `Source File`
  - Metadata: `active_model_name`, `active_model_version`, `trained_at`, `project_name`

### 2) Customer Review
- Format: CSV or JSON
- Fields:
  - `Risk Description`
  - `Final Risk Level`
  - `Final Review Dept`
  - `Confidence`
  - `Top Match (Preview)` (context)
  - `Needs Review`
  - `Source File`
  - Metadata: `active_model_name`, `active_model_version`, `trained_at`, `project_name`

### 3) Internal Engineering
- Format: CSV or JSON
- Fields:
  - `Risk Description`
  - `Final Risk Level`
  - `Final Review Dept`
  - `Confidence`
  - `Rule Trust`
  - `Classic Trust`
  - `Similarity Trust`
  - `Semantic Risk Proba`
  - `Semantic Dept Proba`
  - `Needs Review`
  - `Source File`
  - Metadata: `active_model_name`, `active_model_version`, `trained_at`, `project_name`

## Notes
- Model metadata is always included and cannot be deselected.
- Preset definitions are centralized in `ui_strings.py` under `EXPORT_PRESETS`.
