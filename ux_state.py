"""
Centralized UX state derivation and stepper status helpers.

States follow planV2.0/01_SHARED/UX_STATE_MAP.md.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

UXState = str

NO_PROJECT = "NO_PROJECT"
PROJECT_LOADED_NO_LABELS = "PROJECT_LOADED_NO_LABELS"
LABELS_EXIST_NO_MODEL = "LABELS_EXIST_NO_MODEL"
MODEL_LOADED_READY_TO_CLASSIFY = "MODEL_LOADED_READY_TO_CLASSIFY"
CLASSIFICATION_DONE_NO_UNCERTAIN = "CLASSIFICATION_DONE_NO_UNCERTAIN"
CLASSIFICATION_DONE_HAS_UNCERTAIN = "CLASSIFICATION_DONE_HAS_UNCERTAIN"
TRAINING_RUNNING = "TRAINING_RUNNING"
CLASSIFY_RUNNING = "CLASSIFY_RUNNING"
ERROR_STATE = "ERROR_STATE"

STEPS = [
    "Import Data",
    "Label / Review",
    "Train Model",
    "Classify Document",
    "Export",
]

StepStatus = str  # "not_started", "in_progress", "done", "needs_attention"


@dataclass
class AppState:
    has_model_set: bool
    has_train_file: bool
    has_classify_file: bool
    has_results: bool
    uncertain_count: int
    training_in_progress: bool
    classify_in_progress: bool
    label_count: int = 0
    has_saved_model_sets: bool = False
    last_error_stage: str = ""


def derive_ux_state(state: AppState) -> UXState:
    if state.training_in_progress:
        return TRAINING_RUNNING
    if state.classify_in_progress:
        return CLASSIFY_RUNNING
    if state.last_error_stage:
        return ERROR_STATE
    if state.has_results:
        if state.uncertain_count > 0:
            return CLASSIFICATION_DONE_HAS_UNCERTAIN
        return CLASSIFICATION_DONE_NO_UNCERTAIN
    if state.has_model_set:
        return MODEL_LOADED_READY_TO_CLASSIFY
    if state.has_train_file and not state.has_model_set:
        return PROJECT_LOADED_NO_LABELS
    if state.has_saved_model_sets or state.has_train_file:
        return LABELS_EXIST_NO_MODEL
    return NO_PROJECT


def derive_stepper_status(ux_state: UXState, state: AppState) -> List[StepStatus]:
    """Return status per step, aligned to STEPS order."""
    statuses: List[StepStatus] = ["not_started"] * len(STEPS)

    def mark_done(up_to: int):
        for i in range(up_to):
            statuses[i] = "done"

    def mark_attention(index: int):
        statuses[index] = "needs_attention"

    if ux_state == NO_PROJECT:
        mark_attention(0)
        return statuses

    if ux_state == PROJECT_LOADED_NO_LABELS or ux_state == LABELS_EXIST_NO_MODEL:
        statuses[0] = "done"
        mark_attention(2)
        return statuses

    if ux_state == TRAINING_RUNNING:
        statuses[0] = "done"
        statuses[2] = "in_progress"
        return statuses

    if ux_state == MODEL_LOADED_READY_TO_CLASSIFY:
        mark_done(3)  # Import, Review, Train
        statuses[3] = "needs_attention"
        return statuses

    if ux_state == CLASSIFY_RUNNING:
        mark_done(3)
        statuses[3] = "in_progress"
        return statuses

    if ux_state == CLASSIFICATION_DONE_NO_UNCERTAIN:
        mark_done(4)
        statuses[4] = "needs_attention"
        return statuses

    if ux_state == CLASSIFICATION_DONE_HAS_UNCERTAIN:
        mark_done(2)
        statuses[2] = "done"
        statuses[1] = "needs_attention"
        statuses[3] = "done"
        statuses[4] = "needs_attention"
        return statuses

    if ux_state == ERROR_STATE:
        if state.last_error_stage == "train":
            mark_done(1)
            statuses[1] = "done"
            statuses[2] = "needs_attention"
        elif state.last_error_stage == "classify":
            mark_done(3)
            statuses[3] = "needs_attention"
        else:
            mark_attention(0)
        return statuses

    mark_attention(0)
    return statuses


def helper_text_for_state(ux_state: UXState, state: AppState) -> str:
    if ux_state == NO_PROJECT:
        return "Load a model set or select training data to start."
    if ux_state == PROJECT_LOADED_NO_LABELS:
        return "Train models from the selected labeled CSV."
    if ux_state == LABELS_EXIST_NO_MODEL:
        return "Select a saved model set or train a new one."
    if ux_state == MODEL_LOADED_READY_TO_CLASSIFY:
        return "Choose a CSV/XLSX file, then run Classify."
    if ux_state == TRAINING_RUNNING:
        return "Training in progress—please wait."
    if ux_state == CLASSIFY_RUNNING:
        return "Classifying items—please wait."
    if ux_state == CLASSIFICATION_DONE_NO_UNCERTAIN:
        return "All items classified with confidence—export when ready."
    if ux_state == CLASSIFICATION_DONE_HAS_UNCERTAIN:
        return f"{state.uncertain_count} items need review—open the Review tab."
    if ux_state == ERROR_STATE:
        return "An error occurred. Check logs and retry."
    return ""


def summarize_stepper(ux_state: UXState, state: AppState) -> Tuple[List[StepStatus], str]:
    return derive_stepper_status(ux_state, state), helper_text_for_state(ux_state, state)
