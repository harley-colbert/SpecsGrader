"""
Centralized UI copy for labels, tooltips, and adaptive empty states.
"""

from ux_state import (
    NO_PROJECT,
    PROJECT_LOADED_NO_LABELS,
    LABELS_EXIST_NO_MODEL,
    MODEL_LOADED_READY_TO_CLASSIFY,
    CLASSIFICATION_DONE_HAS_UNCERTAIN,
    CLASSIFICATION_DONE_NO_UNCERTAIN,
)

MIN_LABELS = 50

LABELS = {
    "left_header": "Control Panel",
    "project_group": "Project & Active Model",
    "train_group": "Train Model",
    "classify_group": "Classify Document",
    "advanced_group": "Advanced",
    "results_header": "Results",
    "export": "Export…",
    "classify_cta": "Classify Document",
    "train_cta": "Train",
    "train_override": "Train anyway (override minimum)",
    "model_status_prefix": "Loaded",
    "state_visibility": "Project: {project} • Model: {model} • Trained: {trained}",
    "uncertain_banner": "Review recommended: {count} uncertain items",
    "review_cta": "Open Review Queue",
    "export_disabled": "Export is unavailable until results are generated.",
    "export_dialog_title": "Export Results",
}

TOOLTIPS = {
    "project_group": "Project & Active Model shows which model set is currently loaded.",
    "train_group": "Train a model set from a labeled CSV (min {min_labels} rows recommended).".format(min_labels=MIN_LABELS),
    "classify_group": "Run classification on a CSV or Excel file using the active model.",
    "specs_chip": "Total items in the current results.",
    "risks_chip": "Items with an assigned risk level.",
    "uncertain_chip": "Items flagged as Needs Review.",
    "train_disabled_no_file": "Select a labeled training CSV to enable training.",
    "train_disabled_threshold": "At least {min_labels} labeled rows are recommended before training.".format(min_labels=MIN_LABELS),
    "train_disabled_running": "Training is already in progress.",
    "classify_disabled_no_model": "Load or train an active model before classifying.",
    "classify_disabled_no_file": "Select a CSV or Excel file to classify.",
    "classify_disabled_running": "Classification is already running.",
    "export_disabled": "Export requires classification results.",
    "uncertain_chip_hint": "Uncertain items need human review.",
    "export_gating": "Run classification to enable export.",
}

EMPTY_STATES = {
    NO_PROJECT: {
        "headline": "Import a dataset to begin",
        "body": "Load an existing model set or select a labeled CSV to get started.",
        "cta": "Browse training data",
        "action": "browse_train",
    },
    PROJECT_LOADED_NO_LABELS: {
        "headline": "Ready to train your first model",
        "body": "Train a model set using the selected labeled CSV to unlock classification.",
        "cta": "Train model",
        "action": "train",
    },
    LABELS_EXIST_NO_MODEL: {
        "headline": "Select or train an active model",
        "body": "Load a saved model set or train a new one from your labeled CSV.",
        "cta": "Load model set",
        "action": "refresh_models",
    },
    MODEL_LOADED_READY_TO_CLASSIFY: {
        "headline": "Choose a file to classify",
        "body": "Pick a CSV or Excel file, then run Classify Document.",
        "cta": "Browse file",
        "action": "browse_classify",
    },
    CLASSIFICATION_DONE_HAS_UNCERTAIN: {
        "headline": "Review uncertain items",
        "body": "Some results need human review. Open the Review tab to address them.",
        "cta": "Open Review",
        "action": "open_review",
    },
    CLASSIFICATION_DONE_NO_UNCERTAIN: {
        "headline": "All items classified",
        "body": "Everything is classified with confidence. Export or classify another file.",
        "cta": "Export results",
        "action": "export",
    },
}

EXPORT_PRESETS = {
    "SpecGrader Standard": {
        "include_confidence": True,
        "include_source": True,
        "include_context": False,
    },
    "Customer Review": {
        "include_confidence": True,
        "include_source": True,
        "include_context": True,
    },
    "Internal Engineering": {
        "include_confidence": True,
        "include_source": True,
        "include_context": True,
        "include_trust": True,
    },
}
