# Phase 0 Summary (v4.5 baseline)

## App run
- `python run.py` starts the backend; pywebview is unavailable in this environment so it ran in headless mode.
- UI accessed via `uvicorn backend.app.main:create_app --factory --host 0.0.0.0 --port 7860` for screenshots.

## Screenshots
- Captured Train pane screenshots and noted them in `docs/train_pane_v4.5/README.md`.

## Tests
- `pytest -q` completed successfully.
- Log saved to `tests/logs/pytest_v4.5_baseline.txt`.

## Baseline branch/tag
- Created baseline branch `v4.5-baseline` and tag `specsgrader_v4.5_baseline`.
