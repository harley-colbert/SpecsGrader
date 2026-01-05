# Phase 6 — Export UX (Presets + Model Metadata)

## Objective
Make export a first-class, traceable feature that matches real downstream use.

Deliver:
- Export dialog (instead of “Save Results to CSV”)
- Export presets (SpecGrader Standard / Customer Review / Internal Engineering)
- Always include model metadata (active model + trained date)

## Agent references (assumed available)
- `agentsV2.0/export_ux_designer.md`
- `agentsV2.0/export_schema_engineer.md`
- `agentsV2.0/ui_regression_tester.md`

---

## Step-by-step instructions

### 6.1 Replace “Save Results to CSV” with “Export…”
- Button label: “Export…”
- Disabled when no results exist (with tooltip: “Run classification to export results.”)

### 6.2 Export dialog
Fields:
- Format: CSV / JSON
- Preset: dropdown
- Include toggles:
  - confidence scores
  - source info
  - full context (optional)
  - model metadata (always ON; cannot be disabled)

### 6.3 Presets
Define presets in one place so they are stable and testable.
Example presets:
1) SpecGrader Standard
2) Customer Review (more context, friendlier columns)
3) Internal Engineering (more metadata, IDs)

Document in:
- `01_SHARED/EXPORT_PRESETS.md`

### 6.4 Embed model/version metadata
Export must include at minimum:
- active_model_name
- active_model_version
- trained_at timestamp
- project name/id (if applicable)

Also log export action:
- export time
- filename
- preset used

---

## Phase-specific testing

### Manual tests
- [ ] Export is disabled when no results exist.
- [ ] Export dialog opens and shows preset options.
- [ ] Export includes model metadata fields.
- [ ] Each preset results in expected column set.
- [ ] Export action appears in Log.

### Suggested automated tests
- Unit test: export schema for each preset includes required metadata.
- File test: exported CSV/JSON parses successfully.

---

## Success checklist (must complete)
- [ ] Export dialog exists with presets
- [ ] Model metadata is always included
- [ ] Presets documented in `EXPORT_PRESETS.md`
- [ ] Export gated appropriately with clear tooltip
