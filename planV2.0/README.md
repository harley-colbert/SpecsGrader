# SpecsGrader UI/UX Upgrade — planV2.0

**Plan version:** V2.0  
**Plan date:** 2026-01-05 (America/Detroit)  
**Scope:** UI/UX improvements for an ML training + inference desktop app that both **creates** training data and **uses** that data for classification.  
**Key outcome:** A new user can successfully complete the workflow **Import → Label/Review → Train → Classify → Export** without prior knowledge.

---

## How to use this plan

This plan is organized into phases. **Each phase has:**
- Full instructions (step-by-step)
- Phase-specific testing (manual + suggested automated)
- A success checklist (must be fully checked before moving on)

### Agents assumption (important)
You will create an `agentsV2.0.zip` next. These plan files are written **as if those agents already exist**.

Throughout the phases you will see references like:
- `agentsV2.0/ux_state_mapper.md`
- `agentsV2.0/ui_copy_writer.md`
- `agentsV2.0/ui_regression_tester.md`

If you name your agents differently, keep the intent the same:
- **State mapping agent** (defines workflow states + transitions)
- **Copy/microcopy agent** (writes consistent UI strings)
- **UX QA agent** (runs usability script + checks for regressions)
- **UI wiring agent** (implements stepper/empty states/gating)
- **Metrics agent** (verifies model stats display + versioning)

---

## Global Definition of Done (DoD)

The UI/UX upgrade is complete when:

1. **Workflow clarity**
   - A first-time user can complete the core loop in <10 minutes:
     **Import → Label/Review 10 items → Train → Classify → Export**
   - At every point, the UI clearly shows:
     - Active project
     - Active model (or “none”)
     - The next recommended action (stepper + CTA)

2. **Feedback loop**
   - “Uncertain” results funnel into a Review Queue.
   - Review actions update label counts and can be used for retraining.

3. **Guardrails**
   - Buttons are disabled or warn appropriately when prerequisites are missing
   - Error states provide recovery actions (Copy log, open diagnostics, etc.)

4. **Traceability**
   - Exports include model/version metadata
   - Model history is visible and the active model is unambiguous

---

## Folder layout

- `00_PHASES/` — one file per phase
- `01_SHARED/` — acceptance criteria, UX state map template, test scripts
- `02_ASSETS/` — optional mockups / screenshots to store during execution

---

## Quick index

1. Phase 0 — UX State Map + Baseline
2. Phase 1 — Information Architecture + Navigation (Stepper, chips, Review tab)
3. Phase 2 — Adaptive Empty States + Copy Pack + Guardrails
4. Phase 3 — Review Queue (Active Learning Loop)
5. Phase 4 — Training Run UX + Model History
6. Phase 5 — Results Table + Details Inspector
7. Phase 6 — Export UX (Presets + Metadata)
8. Phase 7 — Polish + Usability Test + Release Gate

---

## Notes
- This plan is implementation-toolkit agnostic. Where useful, each phase includes “Implementation Notes” for common desktop UI stacks.
- If you have existing automated tests, integrate the proposed checks into your current framework rather than creating duplicates.
