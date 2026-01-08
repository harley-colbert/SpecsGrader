# Phase 14 — Full QA, Regression, and Release Packaging (SpecsGraderV4.9.zip)

## Goal
Run end-to-end QA across Path A and Path B, confirm all tests, update docs/versioning, and package the final deliverable zip.

## Primary agents
- QAAgent (owner)
- ReleaseAgent
- OrchestratorAgent
- TestAgent

## Scope
- Final verification across backend, UI, artifacts, and exports/imports.

## Implementation steps (do in order)
- QAAgent:
-   1) Run full test suite: `python -m pytest -q`.
-   2) Run app and validate:
-      - Path A: select ModelSet, load sample classify file, classify, open “Why?” on several rows
-      - Path B: create ModelSet, load training dataset, verify Dataset Health, run training (CV), view metrics, build vector store with TF‑IDF and LSA, re-run classify
-      - Confirm Decision Policy is displayed and trace winner matches expected behavior
-   3) Verify ModelSet export/import:
-      - Export `.sgm`
-      - Import into fresh workspace
-      - Confirm policy + metadata + label policy preserved
-   4) Regression spot-check:
-      - Using `tests/fixtures/classify_sample.csv`, compare predictions before/after upgrade if baseline artifacts are available; document intended differences.
- 
- ReleaseAgent:
-   1) Update version references:
-      - README.md top line should say SpecsGraderV4.9
-      - If UI shows version, update it to 4.9
-   2) Add/Update CHANGELOG section describing:
-      - Dataset Health
-      - per-class metrics
-      - Model Insights
-      - Why/Trace
-      - DecisionPolicy
-      - Embedding backends
-   3) Build final zip: `SpecsGraderv4.9.zip` (exclude caches, include code + tests).
-   4) Validate the zip in a clean folder: unzip, install deps, run tests, run app.

## Testing work to CREATE/UPDATE in this phase
- No new tests required; only fix or stabilize tests added earlier.
- If optional transformer/deep deps are not installed, confirm skipped tests are reported as SKIPPED (not FAILED).

## Tests that MUST pass (gate)
- `python -m pytest -q`
- Manual acceptance checklist must be completed with screenshots stored under `workspace/exports/screenshots/phase14_*`.

## Success checklist (must be YES for every item)
- ✅ Full test suite passes.
- ✅ Manual Path A + Path B workflows succeed with no UI dead-ends.
- ✅ Exports/imports preserve metadata and policies.
- ✅ Final zip `SpecsGraderv4.9.zip` created and smoke-tested from a clean unzip.
