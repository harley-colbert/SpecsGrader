# Phase 04 — Build Vector Index During Training

## Agents
- Run: **AGENT_03_TRAINING_PIPELINE_ENGINEER**
- Optional assist: **AGENT_02_VECTOR_DB_ENGINEER**
- Optional assist: **AGENT_06_TEST_ENGINEER** (for early smoke tests)

## Goal
After training completes, automatically populate the vector DB for that model set using the training embeddings, so inference can do top‑K retrieval without loading a giant DataFrame.

## Files to Modify
- `logic.py` (preferred place: after calling `train_dual_classifiers(...)`)
- Optionally `train_classifier.py` (only if you want it to return embed_df)

## Implementation Steps
1. Choose the build source
   - Recommended: use the embeddings DataFrame already being saved to `models/training_data_embeddings.pkl`
   - Build process:
     - load the pickle via `pd.read_pickle`
     - iterate rows, collect:
       - `id` (stable, includes row index + hash of text)
       - `embedding` list[float]
       - metadata `{text, risk_level, review_dept, row_index, model_set}`

2. Upsert into vector DB
   - Open DB using the model set’s `vector_db_dir` and `vector_collection`
   - Batch inserts (e.g., chunks of 256) to keep memory safe

3. Confirm persistence
   - After build, `db.count()` must equal number of training rows inserted.

4. Keep compatibility artifacts
   - Keep writing `models/training_data_embeddings.pkl` (at least for v2.0) so old code paths and migrations remain possible.

## Must-Pass Tests (no errors)
Create a tiny training dataset at `tests/data/train_min.csv` with at least these columns:
- `Risk Description`
- `Risk Level`
- `Review Department`

Example content (copy exactly):
Risk Description,Risk Level,Review Department
"must comply with nfpa 79 for all wiring",high,electrical
"operator shall wear PPE when handling chemicals",medium,safety
"provide full set of drawings and documentation",low,documentation
"emergency stop circuit shall be category 3 pld",high,controls
"dimensions shall match customer layout",low,mechanical
"perform factory acceptance test with customer witness",medium,quality

Then run:
- `python -m compileall .`
- `python -c "import logic; print(logic.train_all_models('tests/data/train_min.csv', return_file_dict=False, model_set_name='v2_smoke'))"`

After training:
- verify files exist:
  - `models/risklevel_classifier.joblib`
  - `models/reviewdept_classifier.joblib`
  - `models/embedder.joblib`
  - `models/training_data_embeddings.pkl`
  - `models/sem_risklevel_le.joblib`
  - `models/sem_reviewdept_le.joblib`
- verify vector DB count > 0 for that model set’s DB dir

## Success Checklist
- [ ] Training succeeds on the minimal dataset without crashing
- [ ] Vector DB is created and populated automatically
- [ ] The vector DB contains the same number of items as training rows
