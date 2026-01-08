# Phase 13 — Optional Deep Classifier Layer (Stretch, Non-blocking by Default)

## Goal
Optionally add a deep classifier (tiny transformer or embedding+ML) as another DecisionPolicy layer. This phase is included for completeness but should be disabled by default so v4.9 remains lightweight.

## Primary agents
- MLAgent
- BackendAgent
- TestAgent

## Scope
- Add a new method output: `deep` (available:false unless configured).
- DecisionPolicy can include a `deep_model` layer.

## Implementation steps (do in order)
- MLAgent/BackendAgent:
-   1) Preferred lightweight option: train LogisticRegression over LSA embeddings (already available) as a ‘deep-ish’ dense classifier.
-      - This keeps dependencies minimal.
-   2) Create `deep_model_service.py` that returns:
-      - available flag
-      - predictions + confidences
-   3) Add a DecisionPolicy layer `deep_model` (disabled by default).
-   4) Ensure classification pipeline includes method outputs for deep only when enabled.
-   5) If you choose a true transformer classifier:
-      - keep deps optional
-      - ensure tests skip when deps absent.

## Testing work to CREATE/UPDATE in this phase
- Add `tests/test_deep_layer_optional.py`:
-   - If deep disabled, assert method output is available:false and aggregation ignores it.
-   - If deep enabled with LSA+LR, assert it produces predictions and aggregation can select it via policy.

## Tests that MUST pass (gate)
- `python -m pytest -q`

## Success checklist (must be YES for every item)
- ✅ Deep layer exists but is disabled by default.
- ✅ When enabled, it can participate via DecisionPolicy.
- ✅ Suite remains green when deep deps are absent.
