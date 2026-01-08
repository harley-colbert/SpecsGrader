# Agent_Roles

## OrchestratorAgent
- Owns phase sequencing, gating, and final release.
- Reads the plan files and delegates tasks.
- Ensures each phase produces a completion report.

## BackendAgent
- Implements API/service/persistence changes.
- Ensures configuration is stored in ModelSet artifacts (workspace/).
- Adds logging and defensive handling for missing data.

## FrontendAgent
- Implements Train pane stepper and all UI additions.
- Ensures UI is guided/locked until prior steps succeed.
- Updates UX copy, warnings, and panels (Dataset Health, Metrics, Insights, Why?).

## MLAgent
- Implements: TF-IDF + LR pipeline improvements, CV evaluation, DecisionPolicy engine,
  embeddings abstraction, transformer embeddings (optional), deep layer (optional).
- Ensures outputs are calibrated / confidence is meaningful.
- Adds or updates evaluation metrics (per-class, confusion matrices).

## TestAgent
- Writes/updates unit tests and integration-ish tests.
- Adds synthetic fixtures for imbalanced datasets, missing classes, etc.
- Maintains a single command list of tests that must pass per phase.

## QAAgent
- Runs manual smoke checks and usability walkthroughs.
- Captures screenshots for Train pane and Classify “Why?” panels as evidence.
- Confirms UX guidance and gating are coherent.

## ReleaseAgent
- Version bump, changelog, packaging.
- Produces final SpecsGraderv4.9.zip.
- Ensures install/run instructions are current.
