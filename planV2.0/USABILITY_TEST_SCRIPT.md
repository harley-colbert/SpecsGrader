# Usability Test Script (10 minutes, first-time user)

Run this in Phase 7 as the release gate.

## Setup
- Start with a fresh install / clean user data folder (or a new test project).
- Do not give the tester help beyond reading the UI.

## Task 1 — Start
**Goal:** The tester identifies the first action in <10 seconds.
- Expected: they click **Import Dataset** (or create/load a project).

PASS/FAIL notes:

## Task 2 — Create labels
**Goal:** The tester labels at least 10 items.
- Expected: they find the **Review Queue** or **Start Labeling** action without help.
- Expected: they can accept/correct/skip items.

PASS/FAIL notes:

## Task 3 — Train
**Goal:** The tester trains a model (or starts training successfully).
- Expected: Train is discoverable and clearly explained.
- Expected: Training shows progress stages.

PASS/FAIL notes:

## Task 4 — Classify
**Goal:** The tester classifies a document and sees results.
- Expected: Choose File → Classify Document.
- Expected: Specs/Risks/Uncertain counts appear and are clickable.

PASS/FAIL notes:

## Task 5 — Export
**Goal:** The tester exports results and understands what the export contains.
- Expected: Export dialog has presets and includes model metadata.

PASS/FAIL notes:

## Task 6 — Trust / Understanding check (2 questions)
1) “What model is currently active?”
2) “What should you do if there are many ‘Uncertain’ results?”

PASS/FAIL notes:

## Scoring
- **PASS:** All tasks completed without guidance.
- **SOFT PASS:** Completed with minor confusion but no dead ends.
- **FAIL:** Could not determine next step or got stuck.
