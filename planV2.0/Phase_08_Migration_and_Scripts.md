# Phase 08 — Migration + Maintenance Scripts

## Agents
- Run: **AGENT_03_TRAINING_PIPELINE_ENGINEER**
- Optional assist: **AGENT_02_VECTOR_DB_ENGINEER**
- Optional assist: **AGENT_07_RELEASE_MANAGER**

## Goal
Support existing model sets that already have `training_data_embeddings.pkl` by adding scripts that:
- migrate embeddings pickle → vector DB
- rebuild a vector DB from scratch
- validate that a model set is “vector-ready”

## Files to Add
- `scripts/migrate_pkl_to_vector_db.py`
- `scripts/rebuild_vector_db.py`
- `scripts/validate_model_set.py`

## Script Requirements
### migrate_pkl_to_vector_db.py
Inputs:
- model set name (string)
Behavior:
- load `models/model_sets.json`
- locate embeddings pickle path from the model set
- open vector DB path for the set (create if missing)
- upsert all rows
- update model set entry with vector DB keys (if missing)
- print a final summary:
  - rows inserted
  - db count
  - output paths

### rebuild_vector_db.py
Inputs:
- model set name
- confirm flag
Behavior:
- delete/reset the collection
- rebuild from embeddings pickle

### validate_model_set.py
Inputs:
- model set name
Behavior:
- confirms all required keys and files exist
- exits non-zero if anything is missing

## Must-Pass Tests (no errors)
- `python -m compileall .`
- After training `v2_smoke`, run:
  - `python scripts/validate_model_set.py v2_smoke`
  - `python scripts/rebuild_vector_db.py v2_smoke --confirm`
  - `python scripts/migrate_pkl_to_vector_db.py v2_smoke`

## Success Checklist
- [ ] Migration scripts exist and run without exceptions
- [ ] They do not corrupt `models/model_sets.json`
- [ ] Validation script provides clear actionable errors
