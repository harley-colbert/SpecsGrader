# Phase 2 — Adaptive Empty States + Copy Pack + Guardrails

## Objective
Replace vague empty panels with adaptive guidance, and prevent users from clicking into dead ends.

Deliver:
- adaptive empty states (Results pane)
- consistent copy (labels, buttons, tooltips)
- button gating with clear explanations

## Agent references (assumed available)
- `agentsV2.0/ui_copy_writer.md`
- `agentsV2.0/ui_guardrails_engineer.md`
- `agentsV2.0/ui_regression_tester.md`

---

## Step-by-step instructions

### 2.1 Implement adaptive empty states
1) In the Results pane, replace static messages with state-based empty states:
- NO_PROJECT → “Import Dataset”
- PROJECT_LOADED_NO_LABELS → “Start Labeling”
- LABELS_EXIST_NO_MODEL → “Train Model”
- MODEL_LOADED_READY_TO_CLASSIFY → “Choose File”
- CLASSIFICATION_DONE_HAS_UNCERTAIN → “Open Review Queue”
- CLASSIFICATION_DONE_NO_UNCERTAIN → “Export Results” (or “Classify another doc”)

2) Each empty state must have:
- a short headline
- a 1–2 sentence explanation
- **exactly one** primary CTA button
- optional secondary action links

### 2.2 Apply the Copy Pack (consistent language)
1) Update left panel section titles:
- “Project / Model Set” → “Project & Active Model”
- “Train” → “Train Model”
- “Classify (Multipass)” → “Classify Document”
- “Save Results to CSV” → “Export…”

2) Add tooltips / helper lines where ambiguity exists:
- what is a Project?
- what is Active Model?
- what does Uncertain mean?

3) Store UI strings in one place:
- a constants file or a localization dictionary.
This reduces drift across UI components.

### 2.3 Add guardrails (disable/warn with clarity)
Implement predictable prerequisites:
- Train disabled until label count meets threshold (suggest 50)
- Classify disabled until an active model exists
- Export disabled until results exist

For each disabled action:
- show a tooltip explaining “why”
- optionally show a small inline hint linking to the next step

**Important:** Avoid frustrating the user.
If you disable Train at low labels, allow “Train anyway” under an Advanced expander with a warning.

---

## Phase-specific testing

### Manual tests
- [ ] In each UX state, the Results empty state matches the state map.
- [ ] Each empty state has exactly one primary CTA.
- [ ] Disabled buttons show explanatory tooltips.
- [ ] Advanced “Train anyway” (if implemented) shows warning copy and requires explicit confirm.

### Suggested automated tests
- Snapshot test: empty state content by uxState.
- Unit test: gating rules (e.g., `canTrain(appData)`).

---

## Success checklist (must complete)
- [ ] Adaptive empty states implemented for all states in `UX_STATE_MAP.md`
- [ ] Copy pack applied across left panel + Results
- [ ] Strings centralized (no scattered hard-coded UI labels)
- [ ] Guardrails implemented with tooltips/hints
- [ ] Regression pass: existing workflows still function
