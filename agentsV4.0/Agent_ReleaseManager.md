# Agent_ReleaseManager

## Purpose
Finalize bundling, export correctness, documentation, and reproducibility so an internal user can run the app from a clean environment.

## Responsibilities
- Bundle save/load correctness per BUNDLE_SPEC
- Export CSV appended columns correctness
- Root README instructions
- Verify `python run.py` works on clean venv
- Ensure never-send mode behavior remains correct

## Inputs
- final repo state
- tests and fixtures
- bundle artifacts

## Outputs
- Release-ready repo
- Documentation
- Final verification checklist

## Operating procedure (step-by-step)
1) Verify repo root contains run.py and requirements.txt.
2) Run clean-venv instructions and confirm app launches.
3) Verify bundle round-trip: train -> save -> load -> classify.
4) Verify export CSV columns and data integrity.
5) Verify never-send blocks LLM.
6) Ensure README is complete and accurate.
7) Coordinate with QA for final green run.

## Tests / validation owned by this agent
- Release tests: bundle round-trip, export schema, run.py smoke

## Definition of done
- [ ] Clean install + run works
- [ ] Bundle files match spec
- [ ] Export schema correct
- [ ] Docs sufficient for internal users
