# Phase 0 — UX State Map + Baseline

## Objective
Create a shared, explicit understanding of the app’s workflow states and how the UI behaves in each state. Establish baseline screenshots and baseline usability notes.

## Why this phase exists
If the UI does not have an explicit state model, it becomes “a control panel for insiders.” The state map drives:
- the stepper status
- the adaptive empty states
- button enabling/disabling
- warnings and guidance

## Inputs
- Current SpecsGrader codebase
- Current UI screenshot(s)
- Any existing docs about Train/Classify flows

## Agent references (assumed available)
- `agentsV2.0/ux_state_mapper.md`
- `agentsV2.0/ui_inventory_auditor.md`
- `agentsV2.0/ui_regression_tester.md`

---

## Step-by-step instructions

### 0.1 Inventory the current UI
1) Use `agentsV2.0/ui_inventory_auditor.md` to produce:
   - a list of visible UI components
   - current labels (exact strings)
   - current empty states
   - current disabled states (if any)
   - current user flows (what can be done in what order)

2) Save output as:
- `02_ASSETS/phase0_ui_inventory.md`

### 0.2 Define workflow states and transitions
1) Use `agentsV2.0/ux_state_mapper.md` to fill out:
- `01_SHARED/UX_STATE_MAP.md` (copy from template)

2) Include at minimum the states listed in the template and any others you discover.

3) For each state, define:
- entry conditions
- primary CTA
- stepper status
- empty-state copy
- disabled actions + explanation

### 0.3 Capture baseline artifacts
1) Capture baseline screenshots:
- fresh launch
- project loaded (if applicable)
- classify done (if possible)
- log view

2) Store as:
- `02_ASSETS/baseline_*.png`

### 0.4 Baseline usability notes (optional but recommended)
Run the usability script (quickly) against the current app and write a short note:
- where the tester got confused
- where they clicked first
- what they expected to happen

Store as:
- `02_ASSETS/phase0_usability_baseline.md`

---

## Phase-specific testing

### Manual tests
- [ ] Open app fresh: can you clearly identify what to do first?
- [ ] After loading a project/dataset: does the UI show what’s missing (labels/model)?
- [ ] After classification: does the UI clearly point you to Review if uncertain exists?

### Suggested automated tests (implement later if needed)
- Add a unit test for `deriveUxState(appData)`:
  - given mocked data conditions, returns expected UX state.

---

## Success checklist (must complete)
- [ ] `01_SHARED/UX_STATE_MAP.md` exists and maps **every** visible UI situation to exactly one state
- [ ] For each state, there is exactly **one** primary CTA (or an explicit decision rule)
- [ ] Baseline screenshots saved in `02_ASSETS/`
- [ ] Baseline inventory saved in `02_ASSETS/phase0_ui_inventory.md`
