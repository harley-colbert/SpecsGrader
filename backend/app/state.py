from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AppState:
    """In-memory application state placeholder."""

    initialized: bool = False
    jobs: Dict[str, str] = field(default_factory=dict)
    last_health_check: Optional[str] = None


def get_state() -> AppState:
    """Provide a new application state instance.

    Future phases will evolve this to a singleton or managed instance.
    """

    return AppState()
