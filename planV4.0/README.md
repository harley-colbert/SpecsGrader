# SpecsGrader — planV4.0 (Codex-executable phases)

Date: 2026-01-06

This archive contains **one file per phase** for building SpecsGrader V4.0:
a company-internal desktop app (Python + PyWebView backend; HTML/CSS/ESM-JS frontend)
that trains and classifies **risk descriptions** into:

- Risk Level: `none | low | medium | high | extreme`
- Department for Review: `mechanical | electrical | controls | project_management`

## Non-negotiable run requirements
- User runs the app from the project root with:

```bash
python run.py
```

- `requirements.txt` is at the project root.
- `run.py` starts both backend + frontend (served from Python) on **one local port** (preferred).
- The PyWebView window points to that same local URL (same port).

## Input/Output rules (fixed)
- Inputs: CSV or XLSX.
- XLSX: must select a sheet/tab whose name contains **"Standards Risk Matrix"** (first match).
- All tables start on **row 5** (1-indexed; i.e. Excel row 5 is the first data row).
- Column E = risk description (always).
- Training files:
  - Column F = risk level assigned
  - Column G = department for review
- Classify files:
  - Only Column E is filled (F/G blank)
- Export: CSV with appended columns (do not destroy original columns):

`id (optional), risk_text, pred_level, pred_dept, confidence, methods_used, model_bundle_id, user_override_level, user_override_dept`

## Classifier options (per job)
User can enable/disable:
1) Rules (keyword-based, abstains unless confident)
2) Vector similarity (local embeddings default)
3) OpenRouter LLM (optional; env-only API key)

**Never send data externally mode** must hard-disable LLM calls at backend level.

## Aggregation (mandatory)
Main app aggregates method outputs into a single:
- `pred_level`
- `pred_dept`
using **priority-weighted** merging:

**Vector → LLM → Rules**, weighted by priority.

User sets **confidence threshold(s)** at classify time.

## Severe class imbalance handling (mandatory defaults)
- Cost-sensitive learning (class weights; default balanced/inverse frequency)
- Probability outputs + calibration + confidence gating
- Optional capped oversampling (text-safe; no SMOTE)
- Evaluate with macro metrics (Macro F1, per-class recall, balanced accuracy)
- Guardrail: minimum per-class recall >= configurable X (default set in UI; start with a lenient default)

## Agents assumption
These phase files are written assuming an `agentsV4.0` archive exists with agents that Codex can invoke.
The phase files reference these agent roles by name (do not create them in this plan):

- Agent_RepoAuditor
- Agent_BackendEngineer
- Agent_FrontendEngineer
- Agent_DataIngest
- Agent_MLTrainer
- Agent_VectorStore
- Agent_LLMIntegrator
- Agent_Aggregator
- Agent_ResultsUX
- Agent_QA
- Agent_ReleaseManager

## How Codex should execute phases
Execute phases in order (Phase 00 → Phase 08). Each phase must meet:
- All “Tests to run” pass
- All “Success checklist” items are satisfied

If a test fails:
- Fix code until tests pass
- Re-run tests
- Only then proceed
