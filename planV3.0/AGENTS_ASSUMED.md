# Agents assumed to exist (agentsV3.0.zip)

These plan files are written **as if the matching agents already exist** and can be invoked by name.

## Expected agent roster (minimum)
1. **RepoAuditAgent**
   - Reads the current repo (SpecsGraderV2.3).
   - Produces a precise file map, identifies UI entry points, and notes any “gotchas”.

2. **BackendScaffoldAgent**
   - Creates the FastAPI skeleton (`backend/`), static hosting, settings, and routing structure.

3. **FrontendESMAgent**
   - Builds the HTML/CSS/ESM-JS UI shell and page components (no bundler required).
   - Implements the stepper UX and basic state management.

4. **APIContractAgent**
   - Owns OpenAPI schema consistency, request/response models (`schemas/`), and versioning.
   - Ensures frontend ↔ backend contract matches plan.

5. **JobsAndLoggingAgent**
   - Implements background job execution, log capture, and progress APIs (polling and/or SSE).

6. **DataFramesAndPagingAgent**
   - Implements result storage (`result_id`), paging endpoints, and server-side filtering/search.

7. **ParityTestAgent**
   - Adds end-to-end smoke tests and parity checks vs the existing desktop workflow.

8. **PackagingAgent**
   - Updates README, requirements, and optional “legacy Qt UI” install extras.
   - Ensures `pip install -r requirements.txt` + `uvicorn ...` works cleanly.

## Agent invocation convention
Plan steps use the convention:

- **Agent:** `<AgentName>`
- **Input:** the exact paths, constraints, and acceptance criteria
- **Output:** concrete diffs + new/updated files + tests

If your agent framework uses a different syntax, translate the intent; do **not** weaken acceptance criteria.
