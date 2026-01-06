# Phase 08 — Export, polish, docs, and release checks

## Goal
Finalize bundle save/load, export, and documentation so the app is reproducible for internal users.

Deliver:
- Model bundle save/load UI
- Export CSV with appended columns
- Reproducible run instructions
- QA suite and release checklist

## Agents
- Agent_ReleaseManager (lead)
- Agent_QA
- Agent_RepoAuditor
- Agent_BackendEngineer
- Agent_FrontendEngineer

## Bundle save/load (mandatory)
Bundle folder must include:
- `rules_config.json`
- `vector_store/`
- `vector_embedder.json`
- `llm_prompt.json`
- `bundle_meta.json`

Additionally include model artifacts in bundle (within appropriate subfolders):
- sklearn pipelines for TF-IDF models (level/dept)
- calibration artifacts
- any vector classifiers

Endpoints:
- `POST /api/bundles/save`
- `GET /api/bundles/list`
- `POST /api/bundles/load`
- `GET /api/bundles/meta?bundle_id=`

UI:
- Train pane: “Save bundle” (name + notes)
- Classify pane: “Select bundle” dropdown
- Show bundle meta (date, metrics, distributions)

## Export CSV (mandatory)
Export must:
- preserve all original columns
- append required columns (in order):
  1. id (optional)
  2. risk_text
  3. pred_level
  4. pred_dept
  5. confidence
  6. methods_used
  7. model_bundle_id
  8. user_override_level
  9. user_override_dept

If user overrides exist, export `pred_*` as the aggregated predictions and `user_override_*` as edits.
Do NOT overwrite original columns E/F/G unless explicitly requested (not in scope).

Endpoint:
- `POST /api/export/csv`
  - returns file path or triggers save dialog (depending on platform)

## Documentation (mandatory)
Project root `README.md` must include:
- prerequisites (python version)
- venv setup
- `pip install -r requirements.txt`
- `python run.py`
- env var setup for OpenRouter (optional)
- how to enable never-send mode
- file format rules (row 5, col E/F/G, Quote # tab)
- how corrections persistence works

## Testing (must run and pass)
```bash
python -m pytest -q
```

Add release tests:
- bundle round-trip (train → save → load → classify) works
- export CSV contains appended columns in correct order
- never-send mode still blocks LLM after reload
- run.py launches without errors on a clean venv

Manual:
- Fresh clone + venv + install + run (no hidden steps)

## Success checklist
- [ ] Bundles saved/loaded with required artifact files
- [ ] Export CSV correct columns and data integrity
- [ ] Root README enables a new user to run `python run.py`
- [ ] All tests pass on clean environment
- [ ] App is usable end-to-end for 100–1,000 row jobs
