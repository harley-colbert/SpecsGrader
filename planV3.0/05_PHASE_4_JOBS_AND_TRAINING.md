    # Phase 4 — Job system + training job

    ## Goal
    Run training asynchronously via a job manager with logs and progress reporting.

    ## Prerequisites
    - Phase 3 completed.

    ## Implementation steps (must follow in order)
    1. **Agent:** JobsAndLoggingAgent
   - Implement job manager with:
     - `POST /api/train` → returns `job_id`
     - `GET /api/jobs/{job_id}` → status/progress/result ids
     - `GET /api/jobs/{job_id}/logs` → accumulated logs
   - Use threadpool/executor; no blocking requests.

2. **Agent:** BackendScaffoldAgent
   - Training job should:
     - accept `model_set_name` + `training_file_id`
     - load the DataFrame from dataframe_store
     - call existing training pipeline (whatever `logic` currently uses)
     - store training artifacts for optional “save model set”

3. **Agent:** FrontendESMAgent
   - Add Train page:
     - “Run Training” starts job
     - live log area polling `/logs`
     - completion card with summary
   - Ensure UI remains responsive.

    ## Files to create/change
    - **Create:** `backend/api/routes_jobs.py`, `backend/api/routes_training.py`
- **Create:** `backend/services/job_manager.py`
- **Create:** `backend/schemas/jobs.py`
- **Create:** `frontend/src/pages/train.js`
- **Change:** router/sidebar to include Train step

    ## Tests (must run and pass)
    1. `python -m pytest -q`
2. Add `tests/test_api_jobs_training_unittest.py`:
   - start training job → returns job_id
   - poll status until complete (with timeout)
   - ensure logs contain at least one expected marker
3. Manual:
   - Run Train in UI and confirm logs update

    ## Success checklist (must be true before moving on)
    - ✅ Training request returns immediately with `job_id`
- ✅ Job status transitions: queued → running → complete (or failed with error)
- ✅ Logs are retrievable and visible in UI
- ✅ Existing unit tests still pass

    ## Notes for agents (assumed available)
    - JobsAndLoggingAgent should ensure exceptions are captured and returned in job status.
- ParityTestAgent should validate training outputs match desktop on a known fixture (if available).
