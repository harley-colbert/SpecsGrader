# Phase 07 — Aggregation, jobs (progress/cancel), results editing, overrides + corrections dataset

## Goal
Deliver the end-to-end classification workflow:

Classify pane:
- select model bundle
- select enabled methods (vector/llm/rules) subject to never-send mode
- set confidence thresholds
- run job with progress + cancel

Main app aggregates outputs into one final:
- pred_level
- pred_dept
- confidence
- methods_used (JSON string)
- model_bundle_id

Results pane:
- editable pred_level + pred_dept (user overrides)
- overrides persist (app state + disk)
- corrections dataset auto-updates
- “Add to training data” button generates training-ready rows

## Agents
- Agent_Aggregator (lead)
- Agent_BackendEngineer
- Agent_ResultsUX
- Agent_QA
- Agent_RepoAuditor

## Aggregation spec (mandatory)
Priority order: Vector → LLM → Rules

Use priority weights (defaults):
- vector: 1.0
- llm: 0.6
- rules: 0.2

Compute per-target (dept and level) label scores:
`score(L) = Σ(w_method * conf_method * vote_method(L))`

Final label is argmax score.
Final confidence is normalized and compared to user thresholds (set at classify time).

`methods_used` must record:
- which methods were enabled
- each method’s outputs + confidences
- any abstentions
- final label + confidence
- whether below threshold

## Background job execution (mandatory)
Implement `JobManager`:
- run classification in background thread
- progress updates every N rows (configurable; default 25)
- cancel flag checked frequently
- job status endpoint for polling

Required endpoints:
- `POST /api/classify/start`
- `GET /api/classify/status`
- `POST /api/classify/cancel`
- `GET /api/results/rows?limit=&offset=&filters=`
- `POST /api/results/apply_edits`
- `POST /api/results/export_preview` (optional; helps UX)

## Corrections dataset (mandatory persistence)
Implement `CorrectionsService` storing edits append-only.
Preferred: SQLite at `data/corrections.sqlite`.

Store:
- risk_text
- original predictions (level/dept/conf)
- user overrides
- timestamp
- bundle id
- source file hash (if available)

UI requirements:
- Results table supports bulk edits (optional) and per-row edits (mandatory)
- Autosave indicator when edits persist
- Add-to-training button:
  - outputs a training-format XLSX/CSV that matches:
    - row 5 start
    - column E risk_text
    - column F corrected risk level
    - column G corrected dept
  - OR outputs a normalized training CSV that Train pane can ingest (choose one and document)

## Testing (must run and pass)
```bash
python -m pytest -q
```

Required tests:
- Aggregator respects priority/weights (unit tests with synthetic method outputs)
- Cancel stops a running job
- Progress reaches 100% and final row counts match input
- Edits persist to corrections dataset and survive app restart
- methods_used JSON is valid and contains required fields

Manual performance:
- run 1,000-row classify job; UI stays responsive; cancel works

## Success checklist
- [ ] Classify workflow runs end-to-end
- [ ] Aggregation outputs stable final labels with confidence
- [ ] thresholds applied at classify time
- [ ] progress + cancel are reliable
- [ ] Results grid editable; edits persist automatically
- [ ] Corrections dataset grows correctly
- [ ] Add-to-training produces a valid training dataset for re-train
