# Phase 7 — Polish + Usability Test + Release Gate

## Objective
Apply final polish and enforce quality via a release gate:
- run usability script
- verify accessibility basics
- verify error recovery paths
- verify regression checklist

## Agent references (assumed available)
- `agentsV2.0/ui_polish_agent.md`
- `agentsV2.0/ui_accessibility_checker.md`
- `agentsV2.0/ui_regression_tester.md`
- `agentsV2.0/usability_test_runner.md`

---

## Step-by-step instructions

### 7.1 Visual polish (small changes, big impact)
- ensure consistent spacing and alignment across panels
- primary/secondary button hierarchy is consistent
- empty states look intentional (not like missing content)
- tooltips are readable and not truncated

### 7.2 Accessibility basics
- keyboard focus order is logical
- Review Queue usable without mouse (if shortcuts implemented)
- text contrast acceptable (light theme)
- icons have text equivalents/tooltips

### 7.3 Error recovery paths
Ensure each error state provides:
- human-readable message
- “Copy log” button
- at least one recovery action (retry, open diagnostics, choose different model, etc.)

### 7.4 Run the Usability Test Script (release gate)
Use `01_SHARED/USABILITY_TEST_SCRIPT.md`.

Record results in:
- `02_ASSETS/phase7_usability_results.md`

### 7.5 Regression checklist
Use `agentsV2.0/ui_regression_tester.md` to verify:
- Train still works
- Classify still works
- Logs still work
- Model switching works
- Export works
- Review queue works

Record results in:
- `02_ASSETS/phase7_regression_results.md`

---

## Phase-specific testing

### Manual tests (must pass)
- [ ] Usability test script PASS or SOFT PASS (no FAIL)
- [ ] No dead-end actions exist
- [ ] Active model is always obvious
- [ ] Uncertain → Review loop is discoverable and fast

### Suggested automated tests
- Add a “smoke test” command that:
  - loads a sample project
  - runs a classification on a sample file
  - exports results
  - asserts non-empty output

---

## Success checklist (must complete)
- [ ] Usability test complete and recorded
- [ ] Regression test complete and recorded
- [ ] All acceptance criteria in `01_SHARED/ACCEPTANCE_CRITERIA.md` satisfied
- [ ] Final screenshots captured for documentation
