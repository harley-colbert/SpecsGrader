# Upgrade guide: v4.5 → v4.6

SpecsGrader v4.6 focuses on a Train pane UX refresh. Existing ModelSets and
`.sgm` bundles remain compatible, and the underlying training/classification
logic is unchanged.

## What changed
- New Train pane Quick Start selector that lets users choose between:
  - Path A: load an existing ModelSet and classify.
  - Path B: build/update a ModelSet with a guided stepper.
- Readiness strip and clearer microcopy for training data, model, rules, and
  vector store status.
- Validation UI refactor: optional in Path A, gated step in Path B.

## Compatibility notes
- Existing ModelSets and `.sgm` exports from v4.5 can be loaded in v4.6.
- No changes to core training or classification logic beyond UI gating and
  messaging updates.
