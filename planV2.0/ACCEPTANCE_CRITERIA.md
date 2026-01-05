# Shared Acceptance Criteria (All Phases)

Use these criteria as “global constraints” that every phase must respect.

## A. Workflow must be obvious
- The UI must clearly communicate what the user should do next at every step.
- The UI must not require domain knowledge (“you must already understand ML”) to operate.

## B. No dead-end controls
- If an action cannot succeed, the UI must:
  - disable it, **and** provide an explanatory tooltip/hint, or
  - allow it but present an explicit warning and confirm intent.

## C. State visibility
- The user can always see:
  - which Project is active
  - which Model is active (or “none”)
  - when the active model was trained
  - the high-level quality of the active model (at least one metric)

## D. Review loop is first-class
- The Review Queue is easily discoverable.
- “Uncertain” is actionable and clickable.
- Label corrections persist and are available for training.

## E. Export is traceable
- Exports must include model metadata (model name/version + trained date at minimum).
- Export controls are disabled when no results exist.

## F. Error handling is humane
- Errors are human readable.
- Logs can be copied.
- The user gets at least one recovery action.

## G. Consistent language
- Terms are consistent across all panels:
  - Project
  - Active model
  - Train model
  - Classify document
  - Review queue
  - Export
