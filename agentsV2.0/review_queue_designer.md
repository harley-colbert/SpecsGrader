# review_queue_designer.md

## Purpose
Design and implement the Review Queue UI for uncertain items (active learning loop).

## When to run
Phase 3 (required).

## Inputs
- Classification output schema (items include snippet, predicted label, confidence, source)
- `LABELING_POLICY.md` decision (immediate vs staged)
- Review tab shell (from Phase 1)

## Outputs (files/artifacts)
- Review Queue UI (list/table)
- Filters, sort, search
- Per-item actions: Accept / Change Label / Not Relevant / Skip
- Remaining counter

## Agent Operating Rules (do not skip)

- Make changes incrementally and test frequently.
- Prefer small, reviewable commits (if version control is available).
- Do not introduce new “mystery knobs.” If you add settings, explain them in UI copy.
- Avoid scattering business rules across widgets; centralize:
  - UX state derivation
  - gating rules
  - string/copy constants
- If you cannot determine the UI stack quickly, search the repo for:
  - `main.py`, `app.py`, `__main__`
  - `Tk()`, `QMainWindow`, `App()`, `createRoot`, `ReactDOM`


## Procedure
1) Build data model for review items
   - Each item should include:
     id, text/snippet, source (doc/section/page), predicted_label, confidence, status

2) Implement list/table in Review tab
   - Default sort: lowest confidence first
   - Filters: All / Uncertain / Accepted / Corrected (optional but recommended)
   - Search: substring match over text/source

3) Add actions
   - Accept: confirm prediction and persist
   - Change Label: Spec/Risk/Not Relevant and persist
   - Skip: move to next without persisting label (optional: mark skipped)

4) Add “Save & Next” flow
   - After action, automatically advance selection and update remaining count

5) Integrate with Details inspector
   - Selecting a review item shows context and actions in Details (or inline panel)

## Testing
Manual:
- Uncertain banner opens Review.
- Review list sorted by lowest confidence.
- Accept and Change Label persist and update counts.
- Remaining counter decrements.
- Restart app and confirm persisted decisions appear.

Suggested automated:
- Unit test for review item state transitions and persistence calls.

## Success checklist (must complete)
- [ ] Review queue UI exists and shows uncertain items
- [ ] Actions work and persist
- [ ] Remaining count updates correctly
- [ ] Review flow is fast (no extra modal steps for common actions)


