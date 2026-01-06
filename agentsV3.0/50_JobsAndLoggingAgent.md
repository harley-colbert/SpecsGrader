# JobsAndLoggingAgent

## Mission
Implement **non-blocking background jobs** for long-running work (training, classification, export) and expose progress and logs via the backend API.

This agent is primarily used in:
- Phase 4: Job system + training
- Phase 5: Classification job

## Inputs
- Repo working copy
- Phase file from `planV3.0/`
- Any backend skeleton already created (`backend/main.py`, `backend/api`)

## Required outputs
Create or update:
- `backend/services/job_manager.py` (or equivalent)
- `backend/services/session_store.py` (if required for sharing models)
- `backend/api/routes_jobs.py`
- Any schema models in `backend/schemas/jobs.py`

## Functional requirements
- Start long-running tasks without blocking request threads.
- Provide job states: `queued`, `running`, `succeeded`, `failed`.
- Capture logs produced during job execution.
- Support either:
  - polling (`GET /api/jobs/{job_id}`, `GET /api/jobs/{job_id}/logs`), or
  - SSE (`GET /api/jobs/{job_id}/events`), or both.

## Implementation constraints
- Use only stdlib concurrency (thread pool or `asyncio.to_thread`) unless the phase explicitly allows other dependencies.
- Keep it local-first and single-user safe (no need for multi-tenant auth in v3.0).

## Procedure
1. Define `JobRecord` structure (id, status, created_at, started_at, finished_at, progress, error, logs).
2. Implement a `JobManager` that:
   - creates job ids
   - runs callables in a background executor
   - appends logs via a thread-safe mechanism
   - marks completion status
3. Add API endpoints:
   - `POST /api/jobs/start` (optional helper) or job creation embedded in train/classify endpoints
   - `GET /api/jobs/{job_id}`
   - `GET /api/jobs/{job_id}/logs`
   - `GET /api/jobs/{job_id}/events` (optional)
4. Update training/classify endpoints to use the job system per the phase file.

## Acceptance criteria
- Starting training/classify returns immediately with a `job_id`.
- Job status progresses from queued → running → succeeded/failed.
- Logs are visible during execution and preserved after completion.

## Required tests
- Run the phase tests from the plan.
- Add unit tests that cover:
  - job lifecycle transitions
  - log capture
  - error handling sets status=failed and records an error message

## Output format (agent response)
### Summary
### Files changed
### Tests run
### Notes
