# Phase 2 Summary (Path A: Load ModelSet and Classify)

## Updated files
- `frontend/src/panes/trainPane.js`
- `frontend/styles.css`

## Path A structure
- Added a `Step 1 — Load a ModelSet` card under the Quick Start section for `"use-existing"` mode.
- Introduced Local vs Import tabs for loading ModelSets.
- Added an active ModelSet banner plus a readiness summary for models, rules, and vector store.
- Added a `Next step` card with context-aware CTAs:
  - `Go to Classify` when models are present.
  - `Go to Classify (rules-only)` when only rules are present.
  - `Switch to Build / update ModelSet` when nothing usable is loaded.

## Readiness logic
- Uses `state.capabilities` from `/api/state`:
  - `capabilities.model` → models (level/dept)
  - `capabilities.rules` → rules
  - `capabilities.vector` → vector store

## Navigation
- The CTA triggers the global Classify nav button when available, with a `setActivePane("classify")` fallback.
