# QualityAgent — Operating procedure

## General
- Prefer API-level tests using FastAPI TestClient for determinism.
- Add smoke tests only when necessary.
- Verify that the browser console does not show repeated errors during normal workflows.

## Must-check regressions
- Train still runs
- Vector build still runs
- Rules apply during grading
- Load modelset hydrates panes
