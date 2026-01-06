# API Contracts (authoritative)

This document defines the minimum API endpoints the frontend may rely on.
Implement as FastAPI endpoints under `/api/*`.

## Health
- `GET /api/health` -> `{ "ok": true }`

## UI / state
- `GET /api/state`
- `POST /api/ui/set_active_pane` body: `{ "pane": "train|classify|results" }`

## Data ingest
- `POST /api/data/load` body: `{ "mode": "train|classify", "path": "..." }`
- `GET /api/data/preview?mode=train|classify&limit=20&offset=0`

## Training
- `POST /api/train/start` body: training params
- `GET /api/train/status`
- `POST /api/train/cancel`
- `GET /api/train/metrics`

## Bundles
- `POST /api/bundles/save` body: `{ "name": "...", "notes": "...", "workspace_id": "..." }`
- `GET /api/bundles/list`
- `POST /api/bundles/load` body: `{ "bundle_id": "..." }`
- `GET /api/bundles/meta?bundle_id=...`

## Classify jobs
- `POST /api/classify/start` body: `{ bundle_id, enabled_methods, thresholds, llm_model, k, ... }`
- `GET /api/classify/status?job_id=...`
- `POST /api/classify/cancel` body: `{ job_id }`

## Results
- `GET /api/results/rows?limit=&offset=&filters=`
- `POST /api/results/apply_edits` body: `{ edits: [...] }`
- `POST /api/export/csv` body: `{ path?: "...", job_id?: "...", ... }`

## Settings
- `GET /api/settings`
- `POST /api/settings` body: `{ "never_send_externally": true|false }`

### Never-send hard guard
Backend must refuse LLM calls when never-send is enabled, even if UI erroneously requests them.
