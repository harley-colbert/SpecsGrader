from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib


@dataclass
class ModelPrediction:
    available: bool
    level_pred: Optional[str] = None
    level_conf: float = 0.0
    dept_pred: Optional[str] = None
    dept_conf: float = 0.0


class ModelInferenceService:
    def __init__(self, workspace: Path, app_state: Any):
        self.workspace = workspace
        self.app_state = app_state
        self._lock = threading.Lock()
        self._cache: Dict[Path, Tuple[float, Any]] = {}

    def _default_paths(self) -> Tuple[Path, Path]:
        bundle_dir = self.workspace / "workspace_bundle"
        return bundle_dir / "level_model.joblib", bundle_dir / "dept_model.joblib"

    def _resolve_paths(self) -> Tuple[Path, Path]:
        level_path = self.app_state.training_job.get("level_model_path")
        dept_path = self.app_state.training_job.get("dept_model_path")
        if level_path and dept_path:
            return Path(level_path), Path(dept_path)
        return self._default_paths()

    def available(self) -> bool:
        level_path, dept_path = self._resolve_paths()
        return level_path.exists() and dept_path.exists()

    def _load_model(self, path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._cache.get(path)
            if cached and cached[0] == mtime:
                return cached[1]
            model = joblib.load(path)
            self._cache[path] = (mtime, model)
            return model

    @staticmethod
    def _predict_with_model(model: Any, text: str) -> Tuple[Optional[str], float]:
        if model is None:
            return None, 0.0
        pred = model.predict([text])[0]
        conf = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])
            if probs is not None:
                classes = list(getattr(model, "classes_", []))
                try:
                    label_index = classes.index(pred)
                    conf = float(probs[0][label_index])
                except (ValueError, IndexError):
                    conf = 0.0
        return str(pred), conf

    def predict(self, text: str) -> ModelPrediction:
        level_path, dept_path = self._resolve_paths()
        level_model = self._load_model(level_path)
        dept_model = self._load_model(dept_path)
        if level_model is None or dept_model is None:
            return ModelPrediction(available=False)
        level_pred, level_conf = self._predict_with_model(level_model, text)
        dept_pred, dept_conf = self._predict_with_model(dept_model, text)
        return ModelPrediction(
            available=True,
            level_pred=level_pred,
            level_conf=level_conf,
            dept_pred=dept_pred,
            dept_conf=dept_conf,
        )


__all__ = ["ModelInferenceService", "ModelPrediction"]
