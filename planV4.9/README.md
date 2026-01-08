# SpecsGrader planV4.9

Upgrade target: **SpecsGraderV4.6 → SpecsGraderV4.9**

This plan fully integrates “best-practice industrial spec classification” into SpecsGrader **without changing the core philosophy**:
- A **hybrid** pipeline (rules + supervised model + vector similarity + optional LLM)
- Clear “modes” and deterministic production behavior
- Local-first training data and artifact storage (ModelSets in `workspace/`)

## What “done” looks like in v4.9

### User-facing
- Train pane is a **guided stepper** (Path A: Load+Classify, Path B: Build/Update ModelSet)
- Training data load shows **Dataset Health** (class distributions + imbalance warnings)
- Validation shows **per-class precision/recall/F1**, macro/weighted F1, and confusion matrices
- ModelSets include an explicit **Decision Policy** (config-driven ladder for aggregation)
- Classify results include a per-row **“Why?”** explanation:
  - Which layer won (rules/model/vector/weighted/abstain/LLM)
  - Key evidence (rule hits, model confidence + top terms, top neighbors + similarity)
- A **Model Insights** panel shows “top signal terms per class” for risk and department
- A **Label Policy** panel defines what each label means (risk levels and departments)

### Engineering
- Training uses **cross-validation (CV)** for honest stability metrics (default k=5)
- Imbalance strategies are explicit and auditable (class_weight, oversampling, caps)
- ModelSet artifacts include **metadata**: training parameters, dataset snapshot hash, CV results
- Vector store supports a pluggable **Embedding Backend**:
  - `tfidf` (existing)
  - `lsa` (dense semantic via TruncatedSVD; offline-friendly)
  - optional `transformer` (requires extra deps + local model directory; safely skipped if absent)
- Comprehensive tests cover:
  - dataset health
  - metrics schemas
  - decision policy behavior
  - CV training
  - embedding backends
  - trace/explanations

## Assumed agent roster (already exists)

These plan files reference the following agents as if they already exist (they will be created separately in `agentsV4.9.zip`):

- **OrchestratorAgent**: runs phases in order, keeps changelog, enforces “tests must pass”
- **BackendAgent**: FastAPI services, persistence, model training/inference
- **FrontendAgent**: `frontend/src/*` panes, UI gating, rendering metrics
- **MLAgent**: pipelines, CV, embeddings, model insights extraction
- **TestAgent**: adds/updates pytest suites, fixtures, regression tests
- **QAAgent**: end-to-end smoke checks (API + app run) and manual acceptance checklist
- **ReleaseAgent**: version bump, packaging zip, final verification

## Project layout (v4.6/v4.5 family)

- Backend: `backend/app/*`
  - Services: `backend/app/services/*.py`
  - Vector: `backend/app/vector/*`
- Frontend: `frontend/src/panes/*.js`
- Tests: `tests/*.py`
- App entry: `run.py`

## Required global rule for all phases

> If any test fails, the responsible agent must fix the code/tests until the full test suite passes.

## How to run tests

```bash
python -m pytest -q
```

## How to run the app

```bash
python run.py
```

---

Proceed phase-by-phase in numeric order.
