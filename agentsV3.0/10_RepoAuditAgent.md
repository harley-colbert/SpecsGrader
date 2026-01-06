# RepoAuditAgent

## Mission
Produce a precise, developer-usable audit of **SpecsGraderV2.3** that supports the web UI migration in `planV3.0`.

This agent runs in **Phase 0** and creates the baseline map used by all later agents.

## Inputs
- The unpacked repo (from `SpecsGraderV2.3.zip`)
- The current phase file: `planV3.0/01_PHASE_0_BASELINE.md`

## Required outputs
Create (or update) the following in the repo root:
1. `ARTIFACT_REPO_AUDIT.md`
2. `ARTIFACT_IMPORT_GRAPH.md`
3. `ARTIFACT_PHASE0_BASELINE.md`

## Step-by-step procedure
1. **Confirm baseline test pass**
   - Create and activate a venv.
   - Install dependencies (`pip install -r requirements.txt`).
   - Run: `python -m pytest -q`.
   - Record the results in `ARTIFACT_PHASE0_BASELINE.md`.

2. **Create a file map (human readable)**
   - Walk the repo and categorize files into:
     - Desktop UI (PySide6)
     - Core logic / ML
     - Vector DB
     - Model set management
     - Tests
     - Scripts
   - Include key entry points and what they do.

3. **Identify UI entry points and call chain**
   - Identify how the desktop UI starts (which module is the entry point).
   - Identify where user actions call into core logic.
   - Specifically document:
     - `ui_main_window.py` and its page structure
     - `ui_actions.py` wrappers and which `logic.py` functions they call

4. **Locate and document parity-critical behaviors**
   - Document where Excel sheet selection logic lives and how it works:
     - It selects the first sheet whose header contains `Quote #` case-insensitively, else falls back to the first sheet.
   - Document how classification builds the results table:
     - It must include **all original columns** plus model outputs.

5. **Build an import graph (lightweight)**
   - Produce a text-based graph showing high-level imports between:
     - UI → UI actions → logic → engines
   - Keep this readable; do not generate massive auto-dumps.

6. **Risk and gotchas list**
   - Identify likely migration risks, such as:
     - long-running training/classify operations
     - large DataFrame payload sizes
     - file path assumptions (desktop vs browser upload)
   - Provide mitigation notes aligned with planV3.0.

## Acceptance criteria
- `ARTIFACT_REPO_AUDIT.md` clearly describes repo structure, entry points, and responsibilities.
- `ARTIFACT_IMPORT_GRAPH.md` shows how UI connects to business logic.
- `ARTIFACT_PHASE0_BASELINE.md` includes:
  - exact commands run
  - test outcomes
  - any failures and fixes applied
- No functional code changes outside Phase 0 scope except those required to make baseline tests pass.

## Required tests
- `python -m pytest -q`

## Output format (agent response)
### Summary
### Files changed
### Tests run
### Notes
