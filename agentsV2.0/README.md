# SpecsGrader UI/UX Upgrade — agentsV2.0

**Agents pack version:** V2.0  
**Date:** 2026-01-05  
**Purpose:** A matching set of agent prompt files designed to execute `planV2.0.zip` (SpecsGrader UI/UX upgrade).

These agents are written to be used in an agentic dev environment (e.g., ChatGPT Codex / agent mode).
They assume they can:
- read and modify the codebase
- run tests / scripts
- write markdown artifacts and screenshots to the repo
- iterate until phase tests pass

---

## How these agents map to plan phases

### Phase 0
- `ui_inventory_auditor.md`
- `ux_state_mapper.md`
- `ui_regression_tester.md`

### Phase 1
- `ui_stepper_builder.md`
- `ui_navigation_designer.md`
- `ui_regression_tester.md`

### Phase 2
- `ui_copy_writer.md`
- `ui_guardrails_engineer.md`
- `ui_regression_tester.md`

### Phase 3
- `review_queue_designer.md`
- `label_persistence_engineer.md`
- `ui_keyboard_shortcuts.md`
- `ui_regression_tester.md`

### Phase 4
- `training_run_ux_engineer.md`
- `model_versioning_engineer.md`
- `ui_regression_tester.md`

### Phase 5
- `results_table_designer.md`
- `details_inspector_builder.md`
- `label_persistence_engineer.md`
- `ui_regression_tester.md`

### Phase 6
- `export_ux_designer.md`
- `export_schema_engineer.md`
- `ui_regression_tester.md`

### Phase 7
- `ui_polish_agent.md`
- `ui_accessibility_checker.md`
- `usability_test_runner.md`
- `ui_regression_tester.md`

---

## Conventions (shared expectations)

1) **Always start by locating the UI entry points**
- Identify the “main window” or root component.
- Identify left panel code and right results pane code.

2) **Centralize workflow state**
- Implement (or identify) a single function like:
  - `deriveUxState(appState)` (preferred)
- Keep stepper status, empty-state selection, and gating rules derived from this.

3) **Never break the existing core loop**
- Train
- Classify
- Logs
- Save/export

4) **Write artifacts back to the repo**
- UX state map: `01_SHARED/UX_STATE_MAP.md`
- Labeling policy: `01_SHARED/LABELING_POLICY.md`
- Export presets: `01_SHARED/EXPORT_PRESETS.md`
- Screenshots: `02_ASSETS/*.png`

5) **Testing**
- Each agent includes manual tests and suggested automated tests.
- If tests already exist, integrate rather than duplicating.

---

## Orchestration
Use `orchestrator.md` for a “single prompt” that runs phase-by-phase using these agents.
