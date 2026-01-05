# Phase 5 — Results Table + Details Inspector

## Objective
Make classification outputs actionable and trustworthy by improving:
- table columns
- row actions (accept/correct/review/ignore)
- details inspector (context + decision)
- consistent data flow into labels (per labeling policy)

## Agent references (assumed available)
- `agentsV2.0/results_table_designer.md`
- `agentsV2.0/details_inspector_builder.md`
- `agentsV2.0/label_persistence_engineer.md`
- `agentsV2.0/ui_regression_tester.md`

---

## Step-by-step instructions

### 5.1 Table columns (minimum viable)
Update Results Table to include:
- Type (Spec/Risk/Uncertain)
- Confidence (0–1 or %)
- Snippet
- Source (document + section/page)
- Action

### 5.2 Row actions
Add action buttons:
- Accept
- Correct…
- Send to Review
- Ignore

Behavior:
- Accept: confirms predicted label (writes label if policy allows)
- Correct…: lets user choose Spec/Risk/Not Relevant and saves
- Send to Review: adds item to Review Queue (if not already)
- Ignore: marks item as ignored (does not become training data)

### 5.3 Details Inspector
When a row is selected:
- show the full snippet
- show context (surrounding text if available)
- show predicted label + confidence
- show “Your decision” buttons: Mark as Spec / Mark as Risk / Not relevant
- optional: “Add note” field

### 5.4 Ensure flows are consistent
- Table actions and Review actions must follow the same persistence policy
- Counts (Specs/Risks/Uncertain) must reflect the latest decisions

---

## Phase-specific testing

### Manual tests
- [ ] Table shows required columns.
- [ ] Clicking a row populates Details view.
- [ ] Accept and Correct actions persist and update counts.
- [ ] Send to Review adds to Review Queue and is visible there.
- [ ] Ignore prevents item from appearing as a labeled example.

### Suggested automated tests
- Unit test: action handlers route into label persistence correctly.
- UI test: selecting a row shows details; correcting updates table display.

---

## Success checklist (must complete)
- [ ] Table columns updated
- [ ] Row actions implemented with consistent behavior
- [ ] Details inspector shows context and supports edits
- [ ] Table edits persist across restart
