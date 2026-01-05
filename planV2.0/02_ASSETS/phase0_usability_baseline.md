# Phase 0 — Usability Baseline (Lightweight)

Execution notes:
- The GUI could not be run interactively in this headless environment, so findings are derived from code inspection of `ui.py` and expected workflows.

Quick observations to revisit with a real user:
- First-run guidance is minimal; the Control Panel groups rely on prior knowledge of training vs. classification.
- No inline indicator shows whether the selected model set is compatible with the chosen classify file.
- The “Needs Review” concept is present as a boolean column but is not surfaced as a dedicated Review tab or filter.
- Error recovery depends on modal message boxes; there is no “copy log” affordance in the Log tab.

Suggested next steps for live testing:
- Time how long a first-time user takes to find the correct starting action.
- Observe whether users understand when **Classify (Multipass)** is enabled/disabled.
- Ask users where they expect to see uncertain items and exports; note any misclicks.
