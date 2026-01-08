from dataclasses import dataclass, field
from typing import Dict, Optional

from .decision_policy import default_decision_policy


@dataclass
class AppState:
    """In-memory application state placeholder."""

    active_pane: str = "train"
    data_loaded: Dict[str, bool] = field(
        default_factory=lambda: {"train": False, "classify": False}
    )
    # Backwards-compat placeholder (older UI called this "bundle").
    # We keep it, but new code should prefer active_modelset_id.
    active_bundle_id: Optional[str] = None

    # --- ModelSet state ---
    # A "ModelSet" is a named, versioned bundle of:
    # - trained model artifacts (workspace_bundle)
    # - vector store (vector_store)
    # - rules config (rules.json)
    # - training telemetry (params/stats/metrics)
    active_modelset_id: Optional[str] = None
    active_modelset_version_id: Optional[str] = None
    training_dataset: Optional[Dict[str, object]] = None
    classify_dataset: Optional[Dict[str, object]] = None
    rules_config: Dict[str, object] = field(
        default_factory=lambda: {
            "version": "1.0",
            "departments": {
                "mechanical": {"keywords": ["bearing"], "hard_keywords": [], "min_hits": 1},
                "electrical": {"keywords": ["panel"], "hard_keywords": [], "min_hits": 1},
                "controls": {"keywords": ["plc"], "hard_keywords": [], "min_hits": 1},
                "project_management": {"keywords": ["schedule"], "hard_keywords": [], "min_hits": 1},
            },
            "global": {
                "case_sensitive": False,
                "match_mode": "token_contains",
                "abstain_on_tie": True,
            },
        }
    )
    training_job: Dict[str, object] = field(
        default_factory=lambda: {
            "status": "idle",
            "progress": 0.0,
            "phase": None,
            "message": None,
            "events": [],
            "params": None,
            "stats": None,
            "started_at": None,
            "finished_at": None,
            "last_updated_at": None,
            "metrics": None,
            "error": None,
            "level_model_path": None,
            "dept_model_path": None,
            "bundle_meta_path": None,
        }
    )
    sanity_report: Optional[Dict[str, object]] = None
    evaluation_report: Optional[Dict[str, object]] = None
    decision_policy: Dict[str, object] = field(default_factory=default_decision_policy)
    vector_store: Dict[str, object] = field(
        default_factory=lambda: {"built": False, "path": None}
    )
    never_send_externally: bool = False
    results_rows: list = field(default_factory=list)


def get_state() -> AppState:
    """Provide a new application state instance.

    Future phases will evolve this to a singleton or managed instance.
    """

    return AppState()
