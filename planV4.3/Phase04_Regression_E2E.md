# Phase 4 — Regression + E2E validation (Train/Vector/Rules/Grade)

## Objective
Ensure that all core flows remain stable and the new ModelSet features do not regress training, vector building, rules, or grading.

## Scope
Add/extend tests and perform end-to-end manual validation. Minimal feature work; primarily correctness and stability.

## Primary files expected to change
- `tests/test_modelset_routes.py` (if added) — extend
- `tests/test_training_flow_smoke.py` (new)
- `README.md` or docs as needed

## Implementation steps
1. Add a backend smoke test that:
   - starts the app object (without launching the full server if possible)
   - calls key routes using TestClient:
     - create modelset
     - save version (can be mocked if training artifacts absent)
     - load version
     - read active state endpoints
2. Add UI-facing “contract” tests at the API level:
   - “not available” returns 200 with `available:false`
   - no endpoints return 404 in normal UX flows (metrics/rules/vector status)
3. Manual validation:
   - Train with a small dataset
   - Build vectors
   - Set rules
   - Save snapshot
   - Load snapshot
   - Grade a sample input and verify outputs are consistent

## Tests that must pass
- `pytest -q`

Manual test sequence:
1. `python run.py`
2. Train
3. Build vector store
4. Set rules
5. Save snapshot (new version)
6. Load that version
7. Run grading
8. Observe browser console: no repeated errors

## Success checklist
- [ ] Training works end-to-end
- [ ] Vector build works end-to-end
- [ ] Rules set + grading works
- [ ] Save snapshot + load snapshot works
- [ ] API “not available” states return 200 structured payloads (no 404 spam)
- [ ] `pytest -q` passes
