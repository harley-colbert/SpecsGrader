# UX State Map Template

Fill this out in Phase 0 and keep it updated.

## State list (example)
- NO_PROJECT
- PROJECT_LOADED_NO_LABELS
- LABELS_EXIST_NO_MODEL
- MODEL_LOADED_READY_TO_CLASSIFY
- CLASSIFICATION_DONE_NO_UNCERTAIN
- CLASSIFICATION_DONE_HAS_UNCERTAIN
- TRAINING_RUNNING
- CLASSIFY_RUNNING
- ERROR_STATE

## For each state, define:
- **Entry conditions** (what must be true)
- **Primary CTA** (the single main action shown in Results)
- **Secondary actions** (optional)
- **Disabled actions** (and why)
- **Stepper status** (Import/Review/Train/Classify/Export)
- **Results empty-state copy**
- **Left panel highlights** (what is emphasized)
- **Exit conditions / transitions**

## Transition table
| From | Trigger | To | Notes |
|------|---------|----|------|

## Notes
- Avoid “half states.” If it feels ambiguous, split it.
- Your UI should always map to exactly one state.
