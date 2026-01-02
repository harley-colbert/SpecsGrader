# Phase 03 — Model Set Schema Upgrade for Vector DB

## Agents
- Run: **AGENT_03_TRAINING_PIPELINE_ENGINEER**
- Optional assist: **AGENT_02_VECTOR_DB_ENGINEER**

## Goal
Extend the existing model set registry (`models/model_sets.json`) so each model set can also point to:
- vector DB persistence directory
- vector collection name
- optional versioning metadata

## Files to Modify
- `logic.py` (where `file_dict` is created in `train_all_models`)
- `model_set_manager.py` (optional validation utilities)
- `README.md` (document the new keys)

## New Keys to Add (per model set entry)
- `vector_db_dir` — path to the Chroma persistence directory (recommended):
  - `models/vector_db/<safe_model_set_name>/chroma`
- `vector_collection` — collection name (sanitized model set name)
- Optional:
  - `vector_db_backend` — `"chromadb"`
  - `vector_db_metric` — `"cosine"` (or what you configure)

## Implementation Steps
1. Add a “safe name” function
   - Convert model set name to a filesystem-safe identifier:
     - lowercase
     - replace spaces with `_`
     - remove characters not in `[a-z0-9_-]`
   - Use this to create consistent DB folders and collection names.

2. Update `logic.train_all_models(...)`
   - When building `file_dict`, include the new vector DB keys.

3. Backward compatibility
   - Existing model sets without these keys must still load.
   - The app must:
     - treat vector DB as “not available” for that model set
     - fall back to existing pickle similarity (until migrated)

## Must-Pass Tests (no errors)
- `python -m compileall .`
- Create a temporary model set entry in `models/model_sets.json` and verify:
  - `logic.list_model_sets()` still works
  - `logic.get_model_set_names_for_dropdown()` still works

## Success Checklist
- [ ] New keys documented and produced during training
- [ ] Legacy model sets still function
- [ ] No UI crashes due to missing vector DB keys
