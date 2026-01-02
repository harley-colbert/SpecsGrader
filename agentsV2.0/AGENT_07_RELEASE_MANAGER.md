# AGENT_07_RELEASE_MANAGER — Documentation + Versioning + Acceptance Checklist

## Role
Finalize SpecGrader v2.0 as a releasable iteration:
- update docs (README)
- add a simple version constant
- ensure tests + manual acceptance pass
- provide a clear release checklist for future upgrades

## Primary Phase
- Phase 09 (assist)

## Inputs
- Repo with v2.0 changes completed
- Phase doc: `Phase_09_Testing_and_Release.md`

## Outputs
- Updated `README.md` describing:
  - dependencies
  - training workflow
  - vector DB storage layout
  - migration scripts usage
- A version constant:
  - either `version.py` with `__version__ = "2.0.0"`
  - or `app.py` constant (choose one and document it)
- Optional UI display of version in footer (only if low risk)

## Release Acceptance Checklist (must pass)
Automated:
- `pip install -r requirements.txt`
- `python -m compileall .`
- `python -m unittest -v`

Manual:
- clean run with no pre-existing models:
  - delete/rename `models/` temporarily
  - `python splash.py` must still open (warnings allowed; crashes not allowed)
- minimal training:
  - `python -c "import logic; print(logic.train_all_models('tests/data/train_min.csv', model_set_name='v2_smoke'))"`
- minimal inference (CLI):
  - `python -c "import pandas as pd, logic; files=logic.load_model_set('v2_smoke'); models=logic.load_all_models(files); df=pd.read_csv('tests/data/infer_min.csv'); out=logic.multipass_classify(df, models, sim_checkbox=True, top_k=5, similarity_threshold=0.55); print(out.head())"`
- UI inference:
  - open `python splash.py`
  - run inference and confirm evidence columns appear when similarity enabled

## Packaging Notes
- Do not remove legacy pickle artifacts in v2.0:
  - keep `models/training_data_embeddings.pkl`
- Ensure migration scripts are documented with exact commands

## Success Checklist
- [ ] README updated and accurate
- [ ] Version constant exists and is correct
- [ ] All automated tests pass
- [ ] Manual acceptance steps pass (fresh install + training + inference + UI)
