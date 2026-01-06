from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from backend.app.state import AppState
from backend.app.vector.vector_store import VectorStore
from backend.app.vector.embedder import EmbedderConfig


@dataclass
class VectorPrediction:
    dept_pred: str | None
    dept_conf: float
    level_pred: str | None
    level_conf: float
    neighbors: List[Dict[str, object]]


class VectorService:
    def __init__(self, workspace: Path, app_state: AppState):
        self.workspace = workspace
        self.app_state = app_state
        self.vector_dir = self.workspace / "vector_store"
        self.store: VectorStore | None = None

    def build(self, rows: List[Dict[str, object]]) -> None:
        cfg = EmbedderConfig()
        self.store = VectorStore.build(self.vector_dir, rows, cfg)
        self.app_state.vector_store = {"built": True, "path": str(self.vector_dir)}

    def ensure_loaded(self) -> None:
        if self.store is None:
            if not self.app_state.vector_store.get("built"):
                raise RuntimeError("Vector store not built")
            path = Path(self.app_state.vector_store.get("path"))
            self.store = VectorStore(path)

    def predict(self, text: str, k: int = 5) -> VectorPrediction:
        self.ensure_loaded()
        neighbors = self.store.query(text, k)
        dept_counts: Dict[str, int] = {}
        level_counts: Dict[str, int] = {}
        for n in neighbors:
            row = n["row"]
            dept = row.get("label_dept")
            level = row.get("label_level")
            if dept:
                dept_counts[dept] = dept_counts.get(dept, 0) + 1
            if level:
                level_counts[level] = level_counts.get(level, 0) + 1

        def best_vote(counts: Dict[str, int]):
            if not counts:
                return None, 0.0
            total = sum(counts.values())
            best_label, best_count = max(counts.items(), key=lambda x: x[1])
            return best_label, best_count / total if total else 0.0

        dept_pred, dept_conf = best_vote(dept_counts)
        level_pred, level_conf = best_vote(level_counts)

        return VectorPrediction(
            dept_pred=dept_pred,
            dept_conf=dept_conf,
            level_pred=level_pred,
            level_conf=level_conf,
            neighbors=neighbors,
        )


__all__ = ["VectorService", "VectorPrediction"]
