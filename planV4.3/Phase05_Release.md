# Phase 5 — Release packaging (v4.3.x)

## Objective
Finalize version bump, documentation, and deliverable packaging for SpecsGrader v4.3.x.

## Scope
- Version bump to v4.3.x
- Update docs and changelog
- Ensure repo root run path remains `python run.py`
- Ensure `requirements.txt` is correct and complete

## Primary files expected to change
- `backend/app/main.py` (or wherever version is defined)
- `README.md`
- (optional) `CHANGELOG.md`
- `requirements.txt` (only if new deps added)

## Implementation steps
1. Confirm all earlier phases pass tests.
2. Set version to v4.3.0 (or v4.3.1 if minor fix).
3. Update documentation:
   - ModelSet concepts: family + versions
   - How to export/import .sgm
   - Active modelset/version behavior
4. Produce final zip deliverable:
   - `SpecsGraderV4.3.x.zip`

## Tests that must pass
- `pytest -q`
- `python run.py` starts successfully

## Success checklist
- [ ] Version updated to v4.3.x
- [ ] Docs updated for ModelSet CRUD and .sgm
- [ ] App runs from root with `python run.py`
- [ ] `pytest -q` passes
- [ ] Final zip produced
