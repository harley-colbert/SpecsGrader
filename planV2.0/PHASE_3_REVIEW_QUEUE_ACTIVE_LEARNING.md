# Phase 3 — Review Queue (Active Learning Loop)

## Objective
Turn “Uncertain” into an obvious, fast labeling workflow that feeds training data.

Deliver:
- Review Queue list UI
- fast actions (Accept / Change Label / Not Relevant / Skip)
- “Save & Next” flow
- context preview (Details integration)
- clear persistence rules for how reviewed items become training labels

## Agent references (assumed available)
- `agentsV2.0/review_queue_designer.md`
- `agentsV2.0/label_persistence_engineer.md`
- `agentsV2.0/ui_keyboard_shortcuts.md`
- `agentsV2.0/ui_regression_tester.md`

---

## Step-by-step instructions

### 3.1 Define labeling persistence policy
Decide (and document) one of these approaches:
- **A: Immediate labeling** — review actions immediately write to the project’s labeled dataset.
- **B: Staged labeling** — review actions stage items; user clicks “Add to training set”.

Write this policy in:
- `01_SHARED/LABELING_POLICY.md`

Your policy must define:
- what fields are saved (text, source, label, confidence, timestamp, user note)
- how duplicates are handled
- how edits are tracked (audit trail optional)

### 3.2 Implement the Review Queue UI
In the Results → Review tab:
1) Show a list/table of uncertain items with:
- snippet
- predicted label
- confidence
- source (document + section/page if available)

2) Add controls:
- filters: All / Uncertain / Accepted / Corrected (optional)
- sort: lowest confidence first (default)
- search

3) Add per-item actions:
- Accept
- Change Label (Spec/Risk/Not Relevant)
- Skip

4) Add “remaining” counter:
- “12 remaining”

### 3.3 Implement “Save & Next” and keyboard shortcuts (recommended)
- Enter: Save & Next
- 1: Spec
- 2: Risk
- 3: Not Relevant
- S: Skip

If keyboard shortcuts are not feasible, ensure clicks are fast and consistent.

### 3.4 Integrate Details view for context
When selecting a Review Queue item:
- show context in Details tab (or a side panel)
- allow label correction there too

### 3.5 Wire the loop back to training readiness
- Label counts update immediately (or after staging commit)
- Train panel reflects updated counts
- If labels changed since last model, show “Retrain recommended”

---

## Phase-specific testing

### Manual tests
- [ ] After classification with Uncertain > 0, the banner CTA opens Review.
- [ ] Review queue loads items sorted by lowest confidence.
- [ ] Accepting an item changes its status and reduces the remaining count.
- [ ] Changing label persists and updates label distribution.
- [ ] Details view shows context and allows label edits.
- [ ] Train panel reflects new label counts.

### Suggested automated tests
- Unit test: persistence writes correct schema.
- Unit test: review actions update counts and state transitions.
- UI test: keyboard shortcut triggers correct action (if implemented).

---

## Success checklist (must complete)
- [ ] `LABELING_POLICY.md` exists and matches behavior
- [ ] Review Queue supports fast accept/correct/skip
- [ ] Review changes are persisted and visible after restart
- [ ] Uncertain banner reliably drives users into Review flow
- [ ] Retrain recommendation appears when labels change after training
