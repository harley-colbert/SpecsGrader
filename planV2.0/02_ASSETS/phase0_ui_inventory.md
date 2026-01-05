# Phase 0 — UI Inventory

Baseline inventory captured via code inspection (PySide6 UI defined in `ui.py`). The app was not executed in this headless environment, so counts and screenshots reflect static defaults.

## Layout Overview
- **Entry point:** `DualSpecClassifierApp` in `ui.py` (launched via `splash.py`).
- **Left panel (“Control Panel”):**
  - **Project / Model Set** group with status label and dropdown + Refresh button.
  - **Train** group with training file selector, **Train** primary CTA, and **Save Model Set**.
  - **Classify** group with classification file selector and **Classify (Multipass)** primary CTA.
  - **Advanced** collapsible section with similarity toggle, Top K spinbox, and Similarity Threshold spinbox.
- **Right panel (“Results”):**
  - Header chips: `Specs: 0`, `Risks: 0`, `Uncertain: 0`.
  - Primary CTA: **Save Results to CSV** (disabled until results exist).
  - Tabs: **Table**, **Details**, **Log**, **Stats** — each uses a read-only text area with placeholder copy.
  - Status bar initialized to “Ready”.

## Component Inventory (visible text)
- Window title: “Risk Level & Review Department Classifier - Multipass Ensemble”
- Left header: “Control Panel”
- Group titles: “Project / Model Set”, “Train”, “Classify”, “Advanced”
- Labels:
  - Status row: “Loaded: 0 classifiers • Last trained: —”
  - Train file placeholder: “No labeled training file selected” (falls back to “No file selected” when browse canceled)
  - Classify file placeholder: “No file selected for classification” (falls back to “No file selected” when browse canceled)
  - Advanced controls: “Enable Vector DB Similarity”, “Top K”, “Similarity Threshold”
- Buttons / controls:
  - “Refresh”, model set dropdown (first option “None (Unload)”), “Browse” (Train), **“Train”** (primary), “Save Model Set”
  - “Browse” (Classify), **“Classify (Multipass)”** (primary)
  - Advanced toggle labeled “Advanced” (arrow changes on expand/collapse)
  - **“Save Results to CSV”** (primary)
- Right tabs + placeholders:
  - **Table** — placeholder: “Select an input file and click Classify. Results will appear here.”
  - **Details** — placeholder: “Select a result row to see details and evidence.”
  - **Log** — placeholder: “Run logs will appear here.”
  - **Stats** — placeholder: “Summary statistics will appear here.”

## Empty States & Defaults
- On launch, all chips read `Specs: 0`, `Risks: 0`, `Uncertain: 0`; stats panel empty.
- Results text areas show the placeholders above until training or classification is run.
- `train_status_label` and `pred_status_label` start empty; status bar shows “Ready”.
- `save_btn` (Save Results) disabled until `last_pred_df` exists.

## Disabled States & Gating
- **Train** button disabled until a training CSV is chosen.
- **Classify (Multipass)** disabled until a model set is loaded **and** a classify file is chosen.
- **Save Results to CSV** disabled until classification produces `last_pred_df`.
- Similarity controls (Top K, threshold) disabled unless “Enable Vector DB Similarity” is checked **and** a vector DB is loaded; tooltip warns when missing.
- Model dropdown option “None (Unload)” clears models and disables classification.

## Current Workflow (inferred)
1) **Load** a model set from the dropdown (or train a new one).
2) **Train** by browsing for a labeled training CSV → click **Train** → optional prompt to name/save the model set.
3) **Classify** by browsing for a CSV/Excel file → click **Classify (Multipass)**.
4) **Review results** in Table/Details/Log/Stats tabs; summary chips/statistics update; “Needs Review” rows flagged via boolean column.
5) **Export** via **Save Results to CSV**.

## Observed UX Risks / Confusions
- No explicit stepper or guidance on “what to do first” beyond group titles.
- Review/uncertainty flow is implicit (boolean column) and not surfaced as a dedicated queue.
- Error handling relies on modal message boxes; no inline recovery tips in the Results area.
