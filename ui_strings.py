"""Centralized user-facing copy for the SpecsGrader UI."""

APP_TITLE = "SpecsGrader"

# Workflow step titles
WORKFLOW_STEPS = [
    "Import Data",
    "Label / Review",
    "Train Model",
    "Classify Document",
    "Export",
]

# Common buttons
BUTTONS = {
    "browse": "Browse…",
    "continue_review": "Continue to Review",
    "train": "Train Model",
    "classify": "Run Classification",
    "export": "Export",
    "more": "More",
}

# Context labels
STATUS_LABELS = {
    "project": "Project: {project}",
    "model_set": "Model Set: {model}",
    "trained": "Trained: {trained}",
}

EMPTY_STATES = {
    "no_model": "No model set loaded.",
    "no_results": "No results yet.",
    "no_training": "Select training data to begin.",
}

WARNINGS = {
    "confirm_labels": "Confirm your data is correctly labeled.",
}

STEP_LABELS = {
    "import": "Import Data",
    "review": "Label / Review",
    "train": "Train Model",
    "classify": "Classify Document",
    "export": "Export",
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
