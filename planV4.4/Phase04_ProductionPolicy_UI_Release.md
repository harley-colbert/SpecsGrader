# Phase 4 — Production mode decision ladder + UI mode selector + release

## Objective
Finalize **Production mode** using a deterministic decision ladder, add UI mode selector, and ship v4.4.x.

Production decision ladder (recommended):
1. **Hard rules** (high precision, conservative list)
2. **Supervised model** if `model_conf >= model_conf_threshold`
3. **Consensus**: if model and vector agree (and vector similarity is acceptable)
4. **Vector fallback** if similarity strong and margin strong
5. **LLM last resort** if enabled and still ambiguous
6. **Abstain** if still ambiguous

Also:
- Add UI control for selecting mode: sanity / evaluate / production
  - Sanity + Evaluate live in Train pane
  - Production mode selector lives in Classify pane (default)
- Add a “Why” trace in results:
  - which method won
  - what thresholds fired
  - supporting evidence (top vector neighbors IDs, rule hits, LLM rationale if used)

Finally:
- Version bump to **v4.4.0**
- Update docs
- Regression tests

## Assigned agents
- BackendAgent (decision policy)
- FrontendAgent (mode UI + trace display)
- QualityAgent (regression + tests)
- ReleaseAgent (version/docs/zip)
- OrchestratorAgent (coordination)

## Files expected to change
Backend:
- `backend/app/services/aggregate_service.py` (implement ladder + trace)
- `backend/app/main.py` (pass mode + thresholds config; include trace in classify results)
- `backend/app/state.py` (store thresholds/settings defaults)
- `backend/app/services/rule_service.py` (support hard rules vs soft rules schema extension)
- `backend/app/services/vector_service.py` (return similarity + margin for decision making)
- `backend/app/services/llm_service.py` (ensure structured response shape + confidence)
- (optional) `backend/app/services/model_inference_service.py` (already added)

Frontend:
- `frontend/src/panes/classifyPane.js` (mode selector and enabled methods summary)
- `frontend/src/panes/resultsPane.js` (show “why” trace per row)
- `frontend/src/api/client.js`
- `frontend/styles.css` (small UI adjustments)

Tests:
- `tests/test_production_policy.py` (NEW) — deterministic ladder unit tests
- extend classify route tests for trace + abstain behavior

## Implementation steps
1. Define config defaults (in backend state)
   - `model_conf_threshold` (start: 0.75)
   - `vector_similarity_threshold` (start: 0.35–0.55 depending on embedding; choose a safe default + document)
   - `vector_margin_threshold` (start: 0.05–0.15)
   - `allow_llm` default false unless configured
   - `abstain_enabled` true
2. Update rules schema (minimal extension)
   - Allow optional `hard_keywords` list per department
   - Treat `hard_keywords` hits as Step 1 overrides (only when unambiguous)
3. Update vector service outputs used in decisions
   - Return top1 similarity, top2 similarity, margin
   - Return top neighbor labels/ids for trace
4. Implement decision ladder in `aggregate_service`
   - Inputs: mode, method outputs, config thresholds
   - Output: final dept/level + confidence + trace object
5. Wire into classify worker
   - Always produce a trace
   - Ensure “not available” methods don’t crash the ladder
6. UI mode selector
   - Classify pane shows Mode selector (default production)
   - Show which methods are enabled automatically for each mode
7. Results trace UI
   - Display “winner method”, thresholds fired, rule hits, and vector neighbors (if any)
8. Version bump + docs + zip
   - Update README: explain three modes and decision ladder
   - Bump to v4.4.0
   - Produce `SpecsGraderV4.4.0.zip`

## Tests that must pass
### Automated
- `pytest -q`

Required unit tests:
- Ladder chooses hard rules when present
- Ladder chooses model when `model_conf >= threshold`
- Ladder chooses consensus when model+vector agree at moderate conf
- Ladder chooses vector when model low and vector strong
- Ladder calls LLM only under correct conditions (mock)
- Ladder abstains when ambiguous and LLM disabled

Integration tests:
- classify returns `trace` per row
- `available:false` optional resources still produce 200 responses

### Manual regression
1. `python run.py`
2. Train → Sanity check → Evaluate
3. Classify with production mode and confirm:
   - model contributes
   - trace explains decision
   - abstain occurs when ambiguous (optional test row)
4. Confirm browser console has no repeated errors.

## Success checklist
- [ ] Production decision ladder implemented and deterministic
- [ ] Mode selector exists and defaults to production for classification
- [ ] Sanity + Evaluate modes work as designed (from earlier phases)
- [ ] Results include a trace explaining which method won
- [ ] No 404 spam; missing methods return 200 with `available:false`
- [ ] `pytest -q` passes
- [ ] Version bumped to v4.4.0 and docs updated
- [ ] Final `SpecsGraderV4.4.0.zip` produced
