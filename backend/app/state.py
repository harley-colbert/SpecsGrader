from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AppState:
    """In-memory application state placeholder."""

    active_pane: str = "train"
    data_loaded: Dict[str, bool] = field(
        default_factory=lambda: {"train": False, "classify": False}
    )
    active_bundle_id: Optional[str] = None
    training_dataset: Optional[Dict[str, object]] = None
    classify_dataset: Optional[Dict[str, object]] = None
    rules_config: Dict[str, object] = field(
        default_factory=lambda: {
            "version": "1.0",
            "departments": {
                "mechanical": {"keywords": ["bearing"], "min_hits": 1},
                "electrical": {"keywords": ["panel"], "min_hits": 1},
                "controls": {"keywords": ["plc"], "min_hits": 1},
                "project_management": {"keywords": ["schedule"], "min_hits": 1},
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
            "metrics": None,
            "error": None,
        }
    )
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
