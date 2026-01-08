from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.app.state import AppState
from backend.app.vector.vector_store import VectorStore
from backend.app.vector.embedder import EmbedderConfig


@dataclass
class VectorPrediction:
    dept_pred: str | None
    dept_conf: float
    level_pred: str | None
    level_conf: float
    top_similarity: float
    second_similarity: float
    margin: float
    vote_conf_level: Optional[float]
    vote_conf_dept: Optional[float]
    top_neighbors: List[Dict[str, object]]
    neighbors: List[Dict[str, object]]


class VectorService:
    def __init__(self, workspace: Path, app_state: AppState):
        self.workspace = workspace
        self.app_state = app_state
        self.vector_dir = self.workspace / "vector_store"
        self.store: VectorStore | None = None

    def reset_cache(self) -> None:
        """Drop the in-memory vector store cache.

        This is used when a different vector store is loaded from a ModelSet.
        """

        self.store = None

    def build(self, rows: List[Dict[str, object]], config: EmbedderConfig | None = None) -> None:
        cfg = config or EmbedderConfig()
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
        dept_pred, vote_conf_dept = _distance_weighted_vote(neighbors, "label_dept")
        level_pred, vote_conf_level = _distance_weighted_vote(neighbors, "label_level")
        dept_conf = vote_conf_dept or 0.0
        level_conf = vote_conf_level or 0.0
        similarities = [float(n.get("similarity", 0.0)) for n in neighbors]
        top_similarity = similarities[0] if similarities else 0.0
        second_similarity = similarities[1] if len(similarities) > 1 else 0.0
        margin = top_similarity - second_similarity
        top_neighbors = [
            {
                "source_row": n["row"].get("source_row"),
                "label_dept": n["row"].get("label_dept"),
                "label_level": n["row"].get("label_level"),
                "similarity": n.get("similarity"),
            }
            for n in neighbors[:3]
        ]

        return VectorPrediction(
            dept_pred=dept_pred,
            dept_conf=dept_conf,
            level_pred=level_pred,
            level_conf=level_conf,
            top_similarity=top_similarity,
            second_similarity=second_similarity,
            margin=margin,
            vote_conf_level=vote_conf_level,
            vote_conf_dept=vote_conf_dept,
            top_neighbors=top_neighbors,
            neighbors=neighbors,
        )


def _distance_weighted_vote(neighbors: List[Dict[str, object]], field: str) -> Tuple[Optional[str], Optional[float]]:
    scores: Dict[str, float] = {}
    for neighbor in neighbors:
        row = neighbor.get("row") or {}
        label = row.get(field)
        if not label:
            continue
        weight = max(float(neighbor.get("similarity") or 0.0), 0.0)
        if weight <= 0:
            continue
        scores[label] = scores.get(label, 0.0) + weight
    if not scores:
        return None, None
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    total = sum(scores.values()) or 1.0
    return best_label, best_score / total


__all__ = ["VectorService", "VectorPrediction"]
