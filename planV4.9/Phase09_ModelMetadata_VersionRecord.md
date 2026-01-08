# Phase 09 — Model Version Metadata (Training Record)

## Goal
Bundle a full training record with each ModelSet version: dataset snapshot hash, training params, CV metrics, and artifact inventory.

## Primary agents
- BackendAgent
- MLAgent
- TestAgent

## Scope
- Backend: extend `bundle_meta.json` and `version.json` content and ensure export/import preserve it.

## Implementation steps (do in order)
- BackendAgent/MLAgent:
-   1) Extend metadata writing in `training_service.py` and/or `modelset_service.py` to include:
-      - `trained_at` timestamp
-      - dataset snapshot identifier (hash of labeled rows JSON or file hash)
-      - training params (CV folds, oversampling, class_weight, calibration method)
-      - CV summary metrics
-      - artifact paths list (level/dept models, insights models, vector store, rules, policy files)
-   2) Ensure metadata is included inside `.sgm` exports and restored on import.
-   3) Add migration logic: older ModelSets without these fields load with defaults.

## Testing work to CREATE/UPDATE in this phase
- Extend `tests/test_sgm_io.py`:
-   - Export a ModelSet version and re-import it
-   - Assert metadata fields exist and match expected schema
- Add `tests/test_model_metadata_schema.py` for schema validation.

## Tests that MUST pass (gate)
- `python -m pytest -q`

## Success checklist (must be YES for every item)
- ✅ Every ModelSet version has a complete training record.
- ✅ Export/import preserves metadata without loss.
- ✅ Backward compatibility maintained for old versions.
