# Phase 08 — Imbalance Controls + Audit Logging

## Goal
Make class imbalance strategies explicit, controllable, and recorded in metadata: class_weight, oversampling on/off, cap ratios, and before/after distributions.

## Primary agents
- MLAgent
- BackendAgent
- FrontendAgent
- TestAgent

## Scope
- Backend: extend TrainingParams and logs; store before/after distributions.
- Frontend: add toggles for class_weight and oversampling.

## Implementation steps (do in order)
- BackendAgent/MLAgent:
-   1) Extend `TrainingParams`:
-      - `use_class_weight_balanced: bool`
-      - keep `oversample_enabled` and `oversample_cap_ratio`
-   2) Ensure oversampling happens *before* CV/training and record:
-      - pre_distribution
-      - post_distribution
-   3) Ensure pipelines honor `class_weight` toggle (balanced vs None).
-   4) Include these in metadata written to `bundle_meta.json` / `version.json`.
- 
- FrontendAgent:
-   1) In Train Step 4 UI, add:
-      - Toggle: “Use class_weight='balanced'” (default ON)
-      - Toggle: “Oversample minority classes”
-      - Slider/input for cap ratio
-   2) Echo the pre/post distributions near the controls.

## Testing work to CREATE/UPDATE in this phase
- Add `tests/test_imbalance_controls.py`:
-   - When oversampling enabled, assert post_distribution is closer to balanced and respects cap_ratio
-   - When class_weight disabled, assert pipeline is constructed with `class_weight=None`
- Update any tests that assume fixed pipeline settings.

## Tests that MUST pass (gate)
- `python -m pytest -q`

## Success checklist (must be YES for every item)
- ✅ Users can explicitly control imbalance strategies.
- ✅ Pre/post distributions are recorded and visible.
- ✅ Tests confirm oversampling and class_weight behavior.
