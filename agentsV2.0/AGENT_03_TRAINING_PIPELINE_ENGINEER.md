# AGENT_03_TRAINING_PIPELINE_ENGINEER — Model Set Schema + Training Builds Vector Index

## Role
Upgrade training so every new model set becomes “vector-ready”:
- extend model set schema in `models/model_sets.json` with vector DB keys
- add safe-name/sanitization for model set names
- after training, build/populate the vector DB from `training_data_embeddings.pkl`
- add migration/maintenance scripts in Phase 08

## Primary Phases
- Phase 03 (required)
- Phase 04 (required)
- Phase 08 (required)

## Inputs
- SpecGrader repo
- `vector_db.py` implemented (Phase 02)
- Phase docs: `Phase_03_ModelSet_Schema.md`, `Phase_04_Training_Build_VectorIndex.md`, `Phase_08_Migration_and_Scripts.md`

## Outputs
### Modified
- `logic.py`:
  - `train_all_models(...)` writes model sets including vector DB keys when `model_set_name` provided
  - training triggers vector DB build after embeddings pickle is created

### Added (Phase 08)
- `scripts/migrate_pkl_to_vector_db.py`
- `scripts/rebuild_vector_db.py`
- `scripts/validate_model_set.py`

## Phase 03: Schema Upgrade Requirements
Add keys per model set entry:
- `vector_db_dir` (recommended: `models/vector_db/<safe_name>/chroma`)
- `vector_collection` (recommended: `<safe_name>`)
Optional:
- `vector_db_backend` = `"chromadb"`
- `vector_db_metric` = `"cosine"`

Backward compatibility requirement:
- existing model sets without these keys must still load and function (vector DB optional)

## Safe Model Set Name Function (mandatory)
Implement a helper function (in `logic.py` or a small new utility module) that:
- lowercases
- replaces spaces with `_`
- removes characters not in `[a-z0-9_-]`
- collapses repeated underscores
- never returns empty; if empty, use `"modelset"`

Use the safe name for:
- vector DB folder name
- collection name

## Phase 04: Build Vector DB During Training
After training creates `models/training_data_embeddings.pkl`, build the vector DB:

1) Load pickle:
- `df = pd.read_pickle(file_dict["embeddings"])` (or direct path)

2) Required fields in the pickle df (verify; fail with clear error if missing):
- `embedding` (vector)
- `text`
- `risk_level`
- `review_dept`

3) Create stable IDs (mandatory)
ID rules:
- stable across runs for identical dataset ordering
- deterministic
Recommended:
- `id = f"{safe_model_set}:{row_index}:{sha1(text)[:12]}"`

4) Batch upserts
- chunk size around 256 (or 128) to be safe
- convert embeddings to plain Python lists of floats (avoid numpy types causing serialization issues)

5) Confirm persistence
- `db.count()` must equal inserted rows (or at least >=; depending on upsert semantics)
- print a `[DEBUG]` summary

## Phase 08: Scripts
Implement scripts exactly per plan:

### scripts/validate_model_set.py
- argument: model set name
- load `models/model_sets.json`
- verify required keys exist and referenced files exist
- verify vector DB keys exist (if missing, explain)
- exit non-zero on failure

### scripts/migrate_pkl_to_vector_db.py
- argument: model set name
- loads pickle and upserts into vector DB
- updates model set entry with vector DB keys if missing
- prints inserted rows and db count summary

### scripts/rebuild_vector_db.py
- arguments: model set name, `--confirm` flag
- resets collection (only if confirm)
- rebuild from pickle

## Must-Pass Tests (no errors)
From repo root:

1) `python -m compileall .`

2) Minimal training dataset:
Create `tests/data/train_min.csv` exactly as described in Phase 04, then run:
- `python -c "import logic; print(logic.train_all_models('tests/data/train_min.csv', return_file_dict=False, model_set_name='v2_smoke'))"`

3) After training, confirm files exist:
- `models/risklevel_classifier.joblib`
- `models/reviewdept_classifier.joblib`
- `models/embedder.joblib`
- `models/training_data_embeddings.pkl`
- `models/sem_risklevel_le.joblib`
- `models/sem_reviewdept_le.joblib`

4) Confirm vector DB populated:
- Use `VectorDB.open(vector_db_dir, vector_collection).count()` and assert > 0

5) Phase 08 scripts:
- `python scripts/validate_model_set.py v2_smoke`
- `python scripts/rebuild_vector_db.py v2_smoke --confirm`
- `python scripts/migrate_pkl_to_vector_db.py v2_smoke`

## Success Checklist
- [ ] Model sets created during training include vector DB keys
- [ ] Training automatically builds/populates the vector DB
- [ ] Scripts run without exceptions and produce clear output
- [ ] Backward compatibility preserved (legacy model sets still load)
