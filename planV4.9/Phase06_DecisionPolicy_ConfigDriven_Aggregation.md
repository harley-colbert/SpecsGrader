# Phase 06 — Decision Policy: Config-driven Aggregation Ladder

## Goal
Replace hard-coded threshold dicts with an explicit DecisionPolicy object saved per ModelSet version; aggregation becomes deterministic-by-config.

## Primary agents
- BackendAgent
- MLAgent
- FrontendAgent
- TestAgent

## Scope
- Backend: define policy schema, store/load with ModelSet versions, refactor aggregation to evaluate layers per policy.
- Frontend: display policy in Train pane (read-only in v4.9).

## Implementation steps (do in order)
- BackendAgent/MLAgent:
-   1) Define `DecisionPolicy` schema (JSON) stored alongside version artifacts:
-      - Path suggestion: `workspace/modelsets/<id>/versions/<ver>/decision_policy.json`
-   2) Default policy must preserve current behavior in `aggregate_outputs`:
-      - rules hard override
-      - model high confidence
-      - model+vector agreement
-      - vector strong
-      - weighted aggregate
-      - optional llm fallback
-      - abstain
-   3) Refactor `backend/app/services/aggregate_service.py`:
-      - Replace `DEFAULT_THRESHOLDS` with `DEFAULT_DECISION_POLICY`
-      - Implement `evaluate_policy(policy, method_outputs)` that returns final prediction + trace
-      - Keep backwards compatibility: if policy missing, use default policy
-   4) Update `modelset_service.py` to include policy in version bundle metadata and exports.
- 
- FrontendAgent:
-   1) Add a read-only Decision Policy section in Train pane:
-      - show layer order + thresholds
-      - show whether llm fallback is enabled
-   2) Ensure policy is shown when a ModelSet is selected.

## Testing work to CREATE/UPDATE in this phase
- Add `tests/test_decision_policy.py`:
-   - With default policy, ensure results match previous deterministic ladder for a controlled case
-   - With custom policy (fixture), ensure layer ordering changes outcome as expected
- Update `tests/test_production_policy.py` if it relies on old threshold names.

## Tests that MUST pass (gate)
- `python -m pytest -q`

## Success checklist (must be YES for every item)
- ✅ Aggregation behavior is driven by a policy file.
- ✅ Default policy preserves previous behavior.
- ✅ UI displays policy for transparency.
- ✅ Tests cover default and custom policy behavior.
