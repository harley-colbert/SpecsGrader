from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


@dataclass
class UIState:
    """Container for app-wide UI state."""

    project_name: Optional[str] = None
    active_model_set: Optional[str] = None
    trained_timestamp: Optional[str] = None
    train_csv_path: Optional[str] = None
    classify_file_path: Optional[str] = None
    models: Optional[Dict] = None
    last_pred_df: Optional[pd.DataFrame] = None
    review_queue_count: int = 0
    similarity_enabled: bool = True
    similarity_top_k: int = 5
    similarity_threshold: float = 0.55
    log_messages: list[str] = field(default_factory=list)

    def has_training_file(self) -> bool:
        return bool(self.train_csv_path and Path(self.train_csv_path).exists())

    def has_classify_file(self) -> bool:
        return bool(self.classify_file_path and Path(self.classify_file_path).exists())

    def has_models(self) -> bool:
        return bool(self.models)

    def has_results(self) -> bool:
        return self.last_pred_df is not None and not self.last_pred_df.empty

    def update_project_name_from_path(self, path: Optional[str]) -> None:
        if path:
            self.project_name = Path(path).name

    def update_review_count(self) -> None:
        if self.last_pred_df is None or "Needs Review" not in self.last_pred_df.columns:
            self.review_queue_count = 0
            return
        self.review_queue_count = int(self.last_pred_df["Needs Review"].astype(bool).sum())
