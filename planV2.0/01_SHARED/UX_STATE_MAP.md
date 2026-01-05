# UX State Map (Phase 0)

Derived facts used below:
- `hasModelSet`: `models` is not `None` (a model set is loaded).
- `hasTrainFile`: `train_csv_path` is set.
- `hasClassifyFile`: `classify_file_path` is set.
- `hasResults`: `last_pred_df` is not `None`.
- `uncertainCount`: number of rows where `Needs Review` is `True` (0 if column missing).

## NO_PROJECT
- **Entry conditions:** `not hasModelSet` and `not hasTrainFile` and `not hasClassifyFile` and `not hasResults`.
- **Primary CTA:** Load a model set from the dropdown.
- **Secondary actions:** Browse for a labeled training CSV to enable Train.
- **Disabled actions:** Train (no training file), Classify (no model + no file), Save Results (no results). Tooltip/hint: “Select a model set or training file first.”
- **Stepper status:** Import highlighted; Review/Train/Classify/Export blocked.
- **Results empty-state copy:** Headline “Load a model set to begin”; body “Pick an existing model set or select a training CSV to train one. Classification and export will unlock afterward.” CTA “Load model set”.
- **Left panel highlights:** Project/Model Set group, Train file picker.
- **Exit conditions / transitions:** Select training file → PROJECT_LOADED_NO_LABELS. Load model set → MODEL_LOADED_READY_TO_CLASSIFY.

## PROJECT_LOADED_NO_LABELS
- **Entry conditions:** `hasTrainFile` and `not hasModelSet` and `not hasResults`.
- **Primary CTA:** Train models from the selected labeled CSV.
- **Secondary actions:** Load an existing model set instead of training.
- **Disabled actions:** Classify (no model), Save Results (no results).
- **Stepper status:** Import done; Review pending; Train highlighted; Classify/Export blocked.
- **Results empty-state copy:** “Training data detected. Train now to activate classification.” CTA “Train models”.
- **Left panel highlights:** Train group shows selected filename; Train button enabled.
- **Exit conditions / transitions:** Click Train → TRAINING_RUNNING. Load model set → MODEL_LOADED_READY_TO_CLASSIFY.

## LABELS_EXIST_NO_MODEL
- **Entry conditions:** Labeled data is available (saved model sets exist or `hasTrainFile`), but `hasModelSet` is False and `not hasResults`. This covers “models unloaded” even if a model set name was used previously.
- **Primary CTA:** Load a model set from disk.
- **Secondary actions:** Train a new model set from the selected CSV.
- **Disabled actions:** Classify, Save Results.
- **Stepper status:** Import done; Review pending; Train highlighted as the next step; Classify/Export blocked.
- **Results empty-state copy:** “Select or train a model set to continue.” CTA “Load model set”.
- **Left panel highlights:** Model dropdown (“None (Unload)” selected) and Train section.
- **Exit conditions / transitions:** Load model set → MODEL_LOADED_READY_TO_CLASSIFY. Start training → TRAINING_RUNNING.

## MODEL_LOADED_READY_TO_CLASSIFY
- **Entry conditions:** `hasModelSet` is True; classification file may or may not be chosen; `not hasResults`.
- **Primary CTA:** If `hasClassifyFile` → **Classify (Multipass)**. Otherwise → browse for a classification file.
- **Secondary actions:** Save Model Set, adjust similarity settings.
- **Disabled actions:** Save Results (no results yet).
- **Stepper status:** Import/Review/Train marked done; Classify highlighted; Export blocked.
- **Results empty-state copy:** “Ready to classify. Select a CSV/XLSX file and run Multipass.” CTA “Browse file” (or “Classify” when file chosen).
- **Left panel highlights:** Classify group; chips remain at zero.
- **Exit conditions / transitions:** Click Classify → CLASSIFY_RUNNING. Clear/unload models → LABELS_EXIST_NO_MODEL.

## CLASSIFICATION_DONE_NO_UNCERTAIN
- **Entry conditions:** `hasResults` and `uncertainCount == 0`.
- **Primary CTA:** Save Results to CSV.
- **Secondary actions:** Re-run classification with different similarity settings.
- **Disabled actions:** Train button only disabled if no train file; otherwise available for retraining.
- **Stepper status:** Import/Review/Train/Classify done; Export highlighted.
- **Results empty-state copy:** Not applicable (table populated); guidance banner should read “All items auto-classified with high confidence.”
- **Left panel highlights:** Save/Export, ability to load another file.
- **Exit conditions / transitions:** Save completes (stay in state), start new classification → CLASSIFY_RUNNING, unload models → LABELS_EXIST_NO_MODEL.

## CLASSIFICATION_DONE_HAS_UNCERTAIN
- **Entry conditions:** `hasResults` and `uncertainCount > 0`.
- **Primary CTA:** Review uncertain rows (currently via Table tab + Needs Review column).
- **Secondary actions:** Save Results to CSV for external review.
- **Disabled actions:** None specific beyond standard gating (e.g., Save disabled if results cleared).
- **Stepper status:** Import/Review/Train/Classify done; Export highlighted; Review badge should draw attention.
- **Results empty-state copy:** Not applicable; recommendation banner “X items need review—filter by Needs Review to resolve.”
- **Left panel highlights:** None special; suggestion to keep models loaded for re-run after relabeling.
- **Exit conditions / transitions:** Resolve and reclassify (after data updates) → CLASSIFY_RUNNING. If retraining occurs → TRAINING_RUNNING.

## TRAINING_RUNNING
- **Entry conditions:** Train button clicked and training job active.
- **Primary CTA:** Wait for training to complete.
- **Secondary actions:** None (UI shows status/log updates).
- **Disabled actions:** Classify (while job running), Save Results (no results), model dropdown changes should be avoided; Train button effectively locked by in-flight operation.
- **Stepper status:** Train in-progress; other steps paused.
- **Results empty-state copy:** “Training in progress… Models and embeddings are being generated.”
- **Left panel highlights:** Train status label shows progress; log tab collects messages.
- **Exit conditions / transitions:** Training success → MODEL_LOADED_READY_TO_CLASSIFY (models auto-loaded). Training error → ERROR_STATE.

## CLASSIFY_RUNNING
- **Entry conditions:** Classification started with models + file present.
- **Primary CTA:** Wait for classification to finish.
- **Secondary actions:** None during run.
- **Disabled actions:** Save Results (until completion); changing model dropdown mid-run should be blocked (behavior implied).
- **Stepper status:** Classify in-progress.
- **Results empty-state copy:** “Classifying… Multipass ensemble is running. This may take a moment.”
- **Left panel highlights:** Classify status label updates; log tab collects messages.
- **Exit conditions / transitions:** Success → CLASSIFICATION_DONE_HAS_UNCERTAIN or CLASSIFICATION_DONE_NO_UNCERTAIN depending on `uncertainCount`. Error → ERROR_STATE.

## ERROR_STATE
- **Entry conditions:** Training or classification raises an exception (status label set to error, QMessageBox shown).
- **Primary CTA:** Read the error message and retry (Train or Classify) after fixing input.
- **Secondary actions:** Open log tab; adjust files or model set selection.
- **Disabled actions:** Current operation aborted; others allowed once error dialog is dismissed.
- **Stepper status:** Step where error occurred marked as failed; previous steps remain done.
- **Results empty-state copy:** “Something went wrong. Check logs and try again. If the issue persists, reload the model set.”
- **Left panel highlights:** The group related to the failed action (Train or Classify).
- **Exit conditions / transitions:** Retry Train → TRAINING_RUNNING; retry Classify → CLASSIFY_RUNNING; unload models → LABELS_EXIST_NO_MODEL.

## Transition table
| From | Trigger | To | Notes |
|------|---------|----|------|
| NO_PROJECT | Training file selected | PROJECT_LOADED_NO_LABELS | Enables Train CTA. |
| NO_PROJECT | Model set loaded | MODEL_LOADED_READY_TO_CLASSIFY | Classification gated on file selection. |
| PROJECT_LOADED_NO_LABELS | Train clicked | TRAINING_RUNNING | Uses selected labeled CSV. |
| PROJECT_LOADED_NO_LABELS | Model set loaded | MODEL_LOADED_READY_TO_CLASSIFY | Skips training path. |
| LABELS_EXIST_NO_MODEL | Model set loaded | MODEL_LOADED_READY_TO_CLASSIFY | Reloads saved models. |
| LABELS_EXIST_NO_MODEL | Train clicked | TRAINING_RUNNING | Builds a new model set. |
| TRAINING_RUNNING | Training success | MODEL_LOADED_READY_TO_CLASSIFY | Models auto-loaded; save prompt shown. |
| TRAINING_RUNNING | Training error | ERROR_STATE | Error dialog displayed. |
| MODEL_LOADED_READY_TO_CLASSIFY | Classify clicked | CLASSIFY_RUNNING | Requires classify file. |
| MODEL_LOADED_READY_TO_CLASSIFY | Models unloaded | LABELS_EXIST_NO_MODEL | Occurs when selecting “None (Unload)”. |
| CLASSIFY_RUNNING | Classification success (uncertain == 0) | CLASSIFICATION_DONE_NO_UNCERTAIN | Results ready to export. |
| CLASSIFY_RUNNING | Classification success (uncertain > 0) | CLASSIFICATION_DONE_HAS_UNCERTAIN | Needs review surfaced. |
| CLASSIFY_RUNNING | Classification error | ERROR_STATE | Error dialog displayed. |
| CLASSIFICATION_DONE_* | Re-run classification | CLASSIFY_RUNNING | Uses current models and file. |
| CLASSIFICATION_DONE_* | Unload models | LABELS_EXIST_NO_MODEL | Clears readiness for classify. |
| ERROR_STATE | Retry succeeds | Appropriate success state | Depends on operation retried. |
