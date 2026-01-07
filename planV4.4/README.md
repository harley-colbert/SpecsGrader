# SpecsGrader planV4.4

Upgrade target: **SpecsGraderV4.3.zip → SpecsGraderV4.4.x**

This plan introduces **three explicit inference modes** and a robust best-practice hybrid classification pipeline:

## Three modes (v4.4)
1) **Sanity mode** (training sanity)
   - Purpose: verify the *supervised model pipeline* is correct.
   - Runs **model-only** predictions on the labeled training dataset and reports:
     - train-set accuracy (sanity metric)
     - per-row predicted vs. expected
     - confusion matrix (if feasible)
   - Must not call rules/vector/LLM.

2) **Evaluate mode** (honest metrics)
   - Purpose: get unbiased metrics and understand each signal.
   - Runs a stratified split or cross-validation to compute metrics for:
     - model-only
     - rules-only
     - vector-only (k=1 and k>1)
     - ensemble (production policy simulation)
   - After evaluation, refit **final** model on 100% labeled data for production artifacts.

3) **Production mode** (decision ladder)
   - Purpose: classify new unlabeled items reliably and explainably.
   - Uses a tiered decision policy:
     1. conservative hard rules (high precision)
     2. supervised model when confident
     3. consensus check (model+vector agreement)
     4. vector fallback when similarity is strong
     5. LLM last resort (optional)
     6. abstain when still ambiguous

## Key best practices implemented by v4.4
- The trained supervised model participates in classification as a first-class method: **`method="model"`**
- **No 404 spam** for “not available yet” states; return 200 with structured `available:false`
- Explicit config for thresholds and method enabling
- Clear UI for selecting **mode** and understanding which method won
- Tests proving:
  - sanity mode reproduces training labels at high rate (model-only)
  - evaluate mode produces stable metrics
  - production mode follows decision ladder deterministically

## Assumed agent roster (already exists)
- OrchestratorAgent
- BackendAgent
- FrontendAgent
- QualityAgent
- ReleaseAgent

## Standard commands
From repo root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt

python run.py
pytest -q
```
