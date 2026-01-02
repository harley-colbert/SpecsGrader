# Phase 06 — Ensemble Scoring Upgrade (Use Semantic Model + Better “Needs Review”)

## Agents
- Run: **AGENT_04_INFERENCE_PIPELINE_ENGINEER**
- Optional assist: **AGENT_06_TEST_ENGINEER**

## Goal
Improve accuracy and stability without requiring more training data by:
- actually using the semantic LR model’s probabilities (`predict_proba`)
- combining signals instead of winner-take-all trust
- emitting a clear `Needs Review` field

## Files to Modify
- `logic.py` (`multipass_classify`)

## Implementation Steps
1. Enable semantic probabilities
   - Use:
     - `models['clf_rl'].predict_proba(X_vec)`
     - `models['clf_rd'].predict_proba(X_vec)`
   - Decode predicted classes using `models['risklevel_le']` / `models['reviewdept_le']`
   - Record:
     - `Semantic Risk`
     - `Semantic Dept`
     - `Semantic Risk Proba`
     - `Semantic Dept Proba`

2. Replace winner-take-all trust with a simple score fusion
   - Keep v2.0 simple and transparent:
     - `score_rule` = 1.0 if triggered else 0.0
     - `score_classic` = min(classic probs)
     - `score_semantic` = min(semantic probs)
     - `score_knn` = knn_confidence (from Phase 05), or top1 similarity if you don’t compute confidence
   - Final selection rules (suggested):
     - If rule triggered and it provides both labels → use rule output, but still set `Needs Review` if semantic + knn strongly disagree.
     - Else choose label source by max of (classic, semantic, knn) BUT require a minimum threshold:
       - if max_score < 0.60 → `Needs Review=True`
   - Ensure outputs remain consistent and easy to debug.

3. Keep backward compatible output columns
   - Do not remove existing columns; add new ones.

## Must-Pass Tests (no errors)
- `python -m compileall .`
- Run the inference test command from Phase 05 again and verify:
  - output contains semantic proba columns
  - no exceptions

## Success Checklist
- [ ] Semantic model probabilities are computed and used
- [ ] Final output contains `Needs Review` boolean
- [ ] Decisions are more stable than winner-take-all
