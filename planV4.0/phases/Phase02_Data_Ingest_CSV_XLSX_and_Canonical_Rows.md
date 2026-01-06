# Phase 02 — Data ingest (CSV/XLSX) and canonical rows

## Goal
Implement strict file ingest rules for training and classification datasets:

- Accept CSV or XLSX
- XLSX uses first sheet containing "Standards Risk Matrix"
- Data begins at row 5
- Column mapping:
  - E = risk description (always)
  - F = risk level (training only)
  - G = department (training only)

Create canonical internal row objects and dataset summaries for UI.

## Agents
- Agent_DataIngest (lead)
- Agent_BackendEngineer
- Agent_QA
- Agent_FrontendEngineer (for preview UI)

## Backend: IngestService requirements
Create `backend/app/services/ingest_service.py` with:

### Public functions
- `load_training_dataset(path) -> TrainingDataset`
- `load_classify_dataset(path) -> ClassifyDataset`

### Canonical row schema (internal)
- `source_row` (original row number from file; Excel row index)
- `id` (optional, if a column exists; otherwise null)
- `risk_text` (from column E, trimmed)
- Training only:
  - `label_level` (from column F)
  - `label_dept` (from column G)

### Validation rules
- Training load must report:
  - total rows
  - rows missing risk_text
  - rows missing labels (level or dept)
  - invalid label values (not in allowed enums)
- Classify load must report:
  - total rows
  - rows missing risk_text

Do not crash on bad rows; collect errors/warnings in the dataset summary.

## Backend: API endpoints
Add endpoints:
- `POST /api/data/load`
  - body: `{ "mode": "train"|"classify", "path": "..." }`
  - returns dataset summary (counts, errors)
- `GET /api/data/preview?mode=train|classify&limit=20&offset=0`
  - returns preview rows (canonical fields + source_row)

## Frontend UI
Train pane:
- file picker / path input
- “Load” button
- summary panel (total rows, missing, invalid)
- preview table (first N rows)

Classify pane:
- same load + summary + preview

## Testing (must run and pass)
Add fixtures in `tests/fixtures/`:
- `training_sample.xlsx` with multiple tabs (one contains “Standards Risk Matrix”)
- `classify_sample.xlsx`
- CSV equivalents

Automated tests:
```bash
python -m pytest -q
```

Required tests:
- loads correct sheet by name containing “Standards Risk Matrix”
- starts at row 5 exactly
- reads correct columns E/F/G
- validates labels against enums
- preview endpoint returns canonical fields

Manual:
- Load a real file and confirm preview matches Excel visually

## Success checklist
- [ ] CSV and XLSX ingest both work
- [ ] XLSX selects “Standards Risk Matrix” tab correctly
- [ ] Row 5 start honored
- [ ] Column E/F/G mapping honored
- [ ] Dataset summaries show clear errors/warnings (no silent failures)
- [ ] UI displays preview and counts for both modes
