# Phase 4 — Training Run UX + Model History

## Objective
Make training feel like a reproducible “run” with visible outcomes and an explicit active model.

Deliver:
- Training progress stages (Preparing → Training → Evaluating → Saving)
- Post-train summary (key metrics, model name/version, trained date)
- Model history list and “Set Active” behavior
- Active model header always visible

## Agent references (assumed available)
- `agentsV2.0/training_run_ux_engineer.md`
- `agentsV2.0/model_versioning_engineer.md`
- `agentsV2.0/ui_regression_tester.md`

---

## Step-by-step instructions

### 4.1 Implement Training Run staging messages
In Train Model panel:
- show progress in human language (not just raw logs)
- ensure user can always open/copy logs (Log tab)

Suggested stages:
1) Preparing data
2) Training model
3) Evaluating
4) Saving model
5) Done

### 4.2 Post-train summary (minimum viable)
After training completes, display:
- Model name/version (e.g., RiskClassifier_v0.9.3)
- Trained date/time
- At least one metric (preferably F1 + Precision + Recall)
- Dataset snapshot: labeled count used

### 4.3 Model History UI
Add a simple list/modal:
- model versions
- trained date
- metrics
- “Set Active” button

Rule: The user must never wonder which model is being used.

### 4.4 Active model visibility
Add a persistent header in the Results pane:
- Active Project
- Active Model
- Trained date

This must update immediately when switching models.

---

## Phase-specific testing

### Manual tests
- [ ] Training shows stage messages from start to finish.
- [ ] On completion, post-train summary appears.
- [ ] Model History shows at least the latest model with metrics.
- [ ] Switching models updates Active Model header.
- [ ] Classification uses the active model (validate by model name in log and export metadata).

### Suggested automated tests
- Unit test: model registry persistence and “active model” selection.
- Unit test: train run produces a run record with timestamp + metrics.

---

## Success checklist (must complete)
- [ ] Training UX shows human-readable stages
- [ ] Post-train summary exists and is visible without opening logs
- [ ] Model History exists with “Set Active”
- [ ] Active model header is accurate at all times
