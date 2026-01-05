# Phase 1 — Information Architecture + Navigation

## Objective
Make the intended workflow visible by adding:
- a 5-step workflow stepper in the Results area
- clickable counters (Specs/Risks/Uncertain)
- a Review tab entry point

## Agent references (assumed available)
- `agentsV2.0/ui_stepper_builder.md`
- `agentsV2.0/ui_navigation_designer.md`
- `agentsV2.0/ui_regression_tester.md`

---

## Step-by-step instructions

### 1.1 Add the workflow stepper (always visible)
1) Implement a stepper at the top of the right-side Results pane with steps:
   1. Import Data
   2. Label / Review
   3. Train Model
   4. Classify Document
   5. Export

2) Stepper must reflect the UX state map:
- highlight current step
- show status icons (not started / in progress / done / needs attention)
- show a one-line “next step” helper text under the stepper

3) Stepper behavior is driven by a single function:
- `deriveUxState(...)`
- `deriveStepperStatus(uxState, appData)`

**Do not** scatter stepper logic across the UI; keep it centralized.

### 1.2 Make the counters actionable
1) Make “Specs / Risks / Uncertain” counters clickable:
- clicking filters the Table view
- clicking Uncertain opens Review tab (or prompts to open it)

2) If Uncertain > 0, show a banner:
“Review recommended: {N} uncertain items”
with a primary button: “Open Review Queue”.

### 1.3 Add (or expose) the Review tab
1) Add a “Review” tab in the Results pane (next to Table/Details/Log/Stats).
2) The Review tab can be a placeholder in this phase:
- show a message explaining purpose
- show a button that will later open the review queue list

**Note:** Full Review Queue implementation happens in Phase 3.

### 1.4 Navigation clarity
1) Update the right pane header to show:
- Active Project (if applicable)
- Active Model (or “None”)
- Trained date (if model exists)

This is a simple “state visibility header.”

---

## Phase-specific testing

### Manual tests
- [ ] Stepper appears on every Results tab (Table/Details/Review/Log/Stats).
- [ ] Stepper changes status correctly when:
  - no project is loaded
  - a project is loaded without labels
  - labels exist without a model
  - a model is active
  - classification has results
- [ ] Clicking Specs/Risks filters Table.
- [ ] Clicking Uncertain opens Review (or prompts to open it).
- [ ] Uncertain banner appears only when Uncertain > 0.

### Suggested automated tests
- UI test (or headless view test): stepper renders expected labels and state icons for each uxState.
- Unit test: filter toggles update table query state.

---

## Success checklist (must complete)
- [ ] Stepper implemented and wired to `deriveUxState`
- [ ] Specs/Risks/Uncertain are clickable and have predictable behavior
- [ ] Review tab exists and is discoverable
- [ ] State visibility header shows active project/model clearly
- [ ] No regressions in existing Train/Classify behavior (validated via `agentsV2.0/ui_regression_tester.md`)
