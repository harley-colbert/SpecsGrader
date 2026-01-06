import json
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline


@dataclass
class TrainingParams:
    oversample_enabled: bool = False
    oversample_cap_ratio: float = 0.3
    min_recall_per_class: float = 0.5
    calibration_method: str = "sigmoid"


@dataclass
class TrainingResult:
    level_model_path: Path
    dept_model_path: Path
    metrics: Dict[str, object]
    bundle_meta_path: Path


class TrainingService:
    def __init__(self, workspace: Path, app_state):
        self.workspace = workspace
        self.app_state = app_state
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cancel_flag = False
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def oversample_rows(rows: List[Dict[str, object]], label_key: str, cap_ratio: float) -> List[Dict[str, object]]:
        if not rows:
            return rows
        counts: Dict[str, int] = {}
        for row in rows:
            label = str(row.get(label_key) or "")
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            return rows
        majority = max(counts.values())
        target_rows: List[Dict[str, object]] = list(rows)
        for label, count in counts.items():
            if count == 0:
                continue
            max_allowed = int(np.ceil(majority * cap_ratio))
            if count < max_allowed:
                needed = max_allowed - count
                to_add = [r for r in rows if str(r.get(label_key) or "") == label]
                if to_add:
                    target_rows.extend(to_add * (needed // len(to_add)))
                    target_rows.extend(to_add[: needed % len(to_add)])
        return target_rows

    @staticmethod
    def ensure_minimum_per_class(rows: List[Dict[str, object]], label_key: str, min_required: int = 2) -> List[Dict[str, object]]:
        if not rows:
            return rows
        counts: Dict[str, int] = {}
        for row in rows:
            label = str(row.get(label_key) or "")
            counts[label] = counts.get(label, 0) + 1
        augmented = list(rows)
        for label, count in counts.items():
            if count < min_required:
                needed = min_required - count
                examples = [r for r in rows if str(r.get(label_key) or "") == label]
                if examples:
                    augmented.extend(examples * (needed // len(examples)))
                    augmented.extend(examples[: needed % len(examples)])
        return augmented

    @staticmethod
    def _build_pipeline(calibration_method: str) -> Pipeline:
        base = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1)
        calibrated = CalibratedClassifierCV(estimator=base, method=calibration_method, cv=2)
        return Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("clf", calibrated),
            ]
        )

    def _train_task(self, params: TrainingParams) -> None:
        try:
            rows = self.app_state.training_dataset.get("rows", []) if self.app_state.training_dataset else []
            filtered = [r for r in rows if r.get("risk_text") and r.get("label_level") and r.get("label_dept")]
            if not filtered:
                raise ValueError("No labeled training data loaded")

            if params.oversample_enabled:
                filtered = self.oversample_rows(filtered, "label_level", params.oversample_cap_ratio)
                filtered = self.oversample_rows(filtered, "label_dept", params.oversample_cap_ratio)

            filtered = self.ensure_minimum_per_class(filtered, "label_level", min_required=2)
            filtered = self.ensure_minimum_per_class(filtered, "label_dept", min_required=2)

            texts = [str(r.get("risk_text")) for r in filtered]
            levels = [str(r.get("label_level")) for r in filtered]
            depts = [str(r.get("label_dept")) for r in filtered]

            level_pipeline = self._build_pipeline(params.calibration_method)
            dept_pipeline = self._build_pipeline(params.calibration_method)

            with self._lock:
                self.app_state.training_job.update({"progress": 0.35})

            level_pipeline.fit(texts, levels)
            if self._cancel_flag:
                raise RuntimeError("Training canceled")

            with self._lock:
                self.app_state.training_job.update({"progress": 0.65})

            dept_pipeline.fit(texts, depts)
            if self._cancel_flag:
                raise RuntimeError("Training canceled")

            level_preds = level_pipeline.predict(texts)
            dept_preds = dept_pipeline.predict(texts)

            metrics = {
                "level": {
                    "macro_f1": float(f1_score(levels, level_preds, average="macro")),
                    "balanced_accuracy": float(balanced_accuracy_score(levels, level_preds)),
                    "per_class_recall": {
                        cls: float(rec)
                        for cls, rec in zip(
                            level_pipeline.classes_,
                            recall_score(levels, level_preds, average=None, labels=level_pipeline.classes_),
                        )
                    },
                    "confusion_matrix": confusion_matrix(levels, level_preds, labels=list(level_pipeline.classes_)).tolist(),
                },
                "dept": {
                    "macro_f1": float(f1_score(depts, dept_preds, average="macro")),
                    "balanced_accuracy": float(balanced_accuracy_score(depts, dept_preds)),
                    "per_class_recall": {
                        cls: float(rec)
                        for cls, rec in zip(
                            dept_pipeline.classes_,
                            recall_score(depts, dept_preds, average=None, labels=dept_pipeline.classes_),
                        )
                    },
                    "confusion_matrix": confusion_matrix(depts, dept_preds, labels=list(dept_pipeline.classes_)).tolist(),
                },
            }

            bundle_dir = self.workspace / "workspace_bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            level_path = bundle_dir / "level_model.joblib"
            dept_path = bundle_dir / "dept_model.joblib"
            joblib.dump(level_pipeline, level_path)
            joblib.dump(dept_pipeline, dept_path)

            bundle_meta = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "trained_on_rows": len(filtered),
                "label_distribution": {
                    "level": {label: levels.count(label) for label in set(levels)},
                    "dept": {label: depts.count(label) for label in set(depts)},
                },
                "metrics": metrics,
            }
            bundle_meta_path = bundle_dir / "bundle_meta.json"
            bundle_meta_path.write_text(json.dumps(bundle_meta, indent=2), encoding="utf-8")

            with self._lock:
                self.app_state.training_job.update(
                    {
                        "status": "completed",
                        "progress": 1.0,
                        "metrics": metrics,
                        "error": None,
                        "level_model_path": str(level_path),
                        "dept_model_path": str(dept_path),
                        "bundle_meta_path": str(bundle_meta_path),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.app_state.training_job.update(
                    {
                        "status": "error" if not self._cancel_flag else "canceled",
                        "error": None if self._cancel_flag else str(exc),
                    }
                )

    def start_training(self, params: TrainingParams) -> None:
        with self._lock:
            if self.app_state.training_job.get("status") == "running":
                return
            self.app_state.training_job.update({"status": "running", "progress": 0.1, "error": None})
            self._cancel_flag = False

        self._thread = threading.Thread(target=self._train_task, args=(params,), daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        with self._lock:
            self._cancel_flag = True
            self.app_state.training_job["status"] = "canceled"
            self.app_state.training_job["progress"] = 0.0

    def status(self) -> Dict[str, object]:
        with self._lock:
            return dict(self.app_state.training_job)

    def metrics(self) -> Optional[Dict[str, object]]:
        with self._lock:
            return self.app_state.training_job.get("metrics")


__all__ = ["TrainingService", "TrainingParams", "TrainingResult"]
