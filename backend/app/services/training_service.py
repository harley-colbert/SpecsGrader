import json
import tempfile
import threading
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from backend.app.services.rule_service import RuleService
from backend.app.services.aggregate_service import aggregate_outputs
from backend.app.services.ingest_service import DEPARTMENTS, RISK_LEVELS
from backend.app.label_policy import load_label_policy
from backend.app.vector.vector_store import VectorStore
from backend.app.vector.embedder import EmbedderConfig


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
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _log_event(self, message: str, level: str = "info", data: Optional[Dict[str, Any]] = None) -> None:
        """Append an event to the in-memory training job log.

        Notes:
        - We keep this in-memory (AppState) intentionally for local usage.
        - We cap length to avoid unbounded memory growth.
        """

        with self._lock:
            events = self.app_state.training_job.get("events")
            if not isinstance(events, list):
                events = []
            events.append(
                {
                    "ts": self._now_iso(),
                    "level": level,
                    "message": message,
                    "data": data or None,
                }
            )
            # Keep the most recent 200 events.
            if len(events) > 200:
                events = events[-200:]
            self.app_state.training_job["events"] = events
            self.app_state.training_job["last_updated_at"] = self._now_iso()

    def _update_job(self, **updates: Any) -> None:
        with self._lock:
            self.app_state.training_job.update(updates)
            self.app_state.training_job["last_updated_at"] = self._now_iso()

    def _check_cancel(self) -> None:
        if self._cancel_flag:
            raise RuntimeError("Training canceled")

    @staticmethod
    def _label_distribution(rows: List[Dict[str, object]], label_key: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in rows:
            label = str(r.get(label_key) or "").strip().lower()
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    @staticmethod
    def _distribution_pct(counts: Dict[str, int], total: int) -> Dict[str, float]:
        if total <= 0:
            return {label: 0.0 for label in counts}
        return {label: round((count / total) * 100.0, 2) for label, count in counts.items()}

    @staticmethod
    def _ordered_distribution(counts: Dict[str, int], expected: List[str]) -> Dict[str, int]:
        ordered = {label: counts.get(label, 0) for label in expected}
        extras = {label: count for label, count in counts.items() if label not in ordered}
        ordered.update(dict(sorted(extras.items(), key=lambda kv: (-kv[1], kv[0]))))
        return ordered

    def _compute_stats(self, rows: List[Dict[str, object]]) -> Dict[str, Any]:
        total_rows = len(rows)
        missing_risk_text = 0
        missing_level = 0
        missing_dept = 0
        labeled_rows: List[Dict[str, object]] = []
        for r in rows:
            if not r.get("risk_text"):
                missing_risk_text += 1
            if not r.get("label_level"):
                missing_level += 1
            if not r.get("label_dept"):
                missing_dept += 1
            if r.get("risk_text") and r.get("label_level") and r.get("label_dept"):
                labeled_rows.append(r)

        label_distribution_level = self._label_distribution(labeled_rows, "label_level")
        label_distribution_dept = self._label_distribution(labeled_rows, "label_dept")
        expected_levels = sorted(RISK_LEVELS)
        expected_depts = sorted(DEPARTMENTS)
        ordered_level = self._ordered_distribution(label_distribution_level, expected_levels)
        ordered_dept = self._ordered_distribution(label_distribution_dept, expected_depts)
        warnings: List[str] = []
        blocking_errors: List[str] = []

        if total_rows == 0:
            blocking_errors.append("No rows loaded.")
        if missing_risk_text == total_rows and total_rows > 0:
            blocking_errors.append("All rows are missing risk text.")
        if missing_level == total_rows and total_rows > 0:
            blocking_errors.append("All rows are missing risk level labels.")
        if missing_dept == total_rows and total_rows > 0:
            blocking_errors.append("All rows are missing department labels.")
        if len(labeled_rows) == 0 and total_rows > 0:
            blocking_errors.append("No fully labeled rows available for training.")

        if len(labeled_rows) > 0:
            missing_levels = [label for label in expected_levels if ordered_level.get(label, 0) == 0]
            missing_depts = [label for label in expected_depts if ordered_dept.get(label, 0) == 0]
            if missing_levels:
                warnings.append(f"Missing risk level classes: {', '.join(missing_levels)}.")
            if missing_depts:
                warnings.append(f"Missing department classes: {', '.join(missing_depts)}.")

            rare_levels = [label for label, count in ordered_level.items() if count > 0 and count < 2]
            rare_depts = [label for label, count in ordered_dept.items() if count > 0 and count < 2]
            if rare_levels:
                warnings.append(f"Rare risk level classes (n<2): {', '.join(rare_levels)}.")
            if rare_depts:
                warnings.append(f"Rare department classes (n<2): {', '.join(rare_depts)}.")

            if len(labeled_rows) < 5:
                warnings.append("Too few labeled rows for reliable training.")

        return {
            "total_rows": total_rows,
            "labeled_rows": len(labeled_rows),
            "missing_risk_text": missing_risk_text,
            "missing_label_level": missing_level,
            "missing_label_dept": missing_dept,
            "label_distribution": {
                "level": ordered_level,
                "dept": ordered_dept,
            },
            "label_distribution_pct": {
                "level": self._distribution_pct(ordered_level, len(labeled_rows)),
                "dept": self._distribution_pct(ordered_dept, len(labeled_rows)),
            },
            "warnings": warnings,
            "blocking_errors": blocking_errors,
        }

    def dataset_health(self) -> Dict[str, Any]:
        rows = self.app_state.training_dataset.get("rows", []) if self.app_state.training_dataset else []
        stats = self._compute_stats(rows)
        existing_stats = self.app_state.training_job.get("stats") or {}
        self._update_job(stats={**existing_stats, **stats})
        return stats

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

    @staticmethod
    def _build_uncalibrated_pipeline() -> Pipeline:
        base = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1)
        return Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("clf", base),
            ]
        )

    def _train_task(self, params: TrainingParams) -> None:
        try:
            rows = self.app_state.training_dataset.get("rows", []) if self.app_state.training_dataset else []

            self._update_job(phase="validating_data", message="Validating training data", progress=0.12)
            self._log_event("Validating training data")

            filtered = [r for r in rows if r.get("risk_text") and r.get("label_level") and r.get("label_dept")]
            if not filtered:
                raise ValueError("No labeled training data loaded")

            pre_dist = {
                "level": self._label_distribution(filtered, "label_level"),
                "dept": self._label_distribution(filtered, "label_dept"),
            }
            self._update_job(stats={**(self.app_state.training_job.get("stats") or {}), "preprocess_distribution": pre_dist})
            self._log_event("Computed initial label distribution", data=pre_dist)

            self._check_cancel()

            if params.oversample_enabled:
                self._update_job(phase="oversampling", message="Oversampling minority classes", progress=0.18)
                self._log_event(
                    "Oversampling enabled",
                    data={"cap_ratio": params.oversample_cap_ratio},
                )
                filtered = self.oversample_rows(filtered, "label_level", params.oversample_cap_ratio)
                filtered = self.oversample_rows(filtered, "label_dept", params.oversample_cap_ratio)

            self._check_cancel()

            self._update_job(phase="ensuring_minimums", message="Ensuring minimum examples per class", progress=0.24)
            filtered = self.ensure_minimum_per_class(filtered, "label_level", min_required=2)
            filtered = self.ensure_minimum_per_class(filtered, "label_dept", min_required=2)

            post_dist = {
                "level": self._label_distribution(filtered, "label_level"),
                "dept": self._label_distribution(filtered, "label_dept"),
            }
            self._update_job(stats={**(self.app_state.training_job.get("stats") or {}), "training_distribution": post_dist, "trained_on_rows": len(filtered)})
            self._log_event("Final training distribution prepared", data=post_dist)

            self._check_cancel()

            texts = [str(r.get("risk_text")) for r in filtered]
            levels = [str(r.get("label_level")) for r in filtered]
            depts = [str(r.get("label_dept")) for r in filtered]

            insights_level_pipeline = self._build_uncalibrated_pipeline()
            insights_dept_pipeline = self._build_uncalibrated_pipeline()
            insights_level_pipeline.fit(texts, levels)
            insights_dept_pipeline.fit(texts, depts)

            # Build initial calibrated pipelines
            level_pipeline = self._build_pipeline(params.calibration_method)
            dept_pipeline = self._build_pipeline(params.calibration_method)

            # Fit level model with graceful fallback if calibration is not feasible
            self._update_job(
                phase="fit_level_model",
                message="Training risk level model (TF-IDF + calibrated logistic regression)",
                progress=0.32,
            )
            self._log_event("Fitting risk level model")
            try:
                level_pipeline.fit(texts, levels)
            except ValueError as exc:
                # New in v4.5: if calibration cannot run because there are too few
                # examples per class for the requested cross-validation strategy,
                # fall back to an uncalibrated LogisticRegression model.
                if "less than 2 examples for at least one class" in str(exc):
                    self._log_event(
                        "Calibration for level model skipped due to small sample size; "
                        "falling back to uncalibrated LogisticRegression",
                        data={"error": str(exc)},
                    )
                    level_pipeline = self._build_uncalibrated_pipeline()
                    level_pipeline.fit(texts, levels)
                else:
                    raise

            self._check_cancel()

            # Fit department model with the same graceful fallback behaviour
            self._update_job(
                phase="fit_dept_model",
                message="Training department model (TF-IDF + calibrated logistic regression)",
                progress=0.62,
            )
            self._log_event("Fitting department model")
            try:
                dept_pipeline.fit(texts, depts)
            except ValueError as exc:
                if "less than 2 examples for at least one class" in str(exc):
                    self._log_event(
                        "Calibration for dept model skipped due to small sample size; "
                        "falling back to uncalibrated LogisticRegression",
                        data={"error": str(exc)},
                    )
                    dept_pipeline = self._build_uncalibrated_pipeline()
                    dept_pipeline.fit(texts, depts)
                else:
                    raise

            self._check_cancel()

            self._update_job(phase="evaluating", message="Evaluating training set performance", progress=0.78)
            self._log_event("Evaluating models")

            level_preds = level_pipeline.predict(texts)
            dept_preds = dept_pipeline.predict(texts)
            level_labels = list(level_pipeline.classes_)
            dept_labels = list(dept_pipeline.classes_)
            level_precision = precision_score(levels, level_preds, average=None, labels=level_labels, zero_division=0)
            dept_precision = precision_score(depts, dept_preds, average=None, labels=dept_labels, zero_division=0)
            level_recall = recall_score(levels, level_preds, average=None, labels=level_labels, zero_division=0)
            dept_recall = recall_score(depts, dept_preds, average=None, labels=dept_labels, zero_division=0)
            level_f1 = f1_score(levels, level_preds, average=None, labels=level_labels, zero_division=0)
            dept_f1 = f1_score(depts, dept_preds, average=None, labels=dept_labels, zero_division=0)

            metrics = {
                "level": {
                    "macro_f1": float(f1_score(levels, level_preds, average="macro", zero_division=0)),
                    "weighted_f1": float(f1_score(levels, level_preds, average="weighted", zero_division=0)),
                    "balanced_accuracy": float(balanced_accuracy_score(levels, level_preds)),
                    "labels": level_labels,
                    "per_class_precision": {
                        cls: float(prec)
                        for cls, prec in zip(level_labels, level_precision)
                    },
                    "per_class_recall": {
                        cls: float(rec)
                        for cls, rec in zip(level_labels, level_recall)
                    },
                    "per_class_f1": {
                        cls: float(score)
                        for cls, score in zip(level_labels, level_f1)
                    },
                    "confusion_matrix": confusion_matrix(levels, level_preds, labels=level_labels).tolist(),
                },
                "dept": {
                    "macro_f1": float(f1_score(depts, dept_preds, average="macro", zero_division=0)),
                    "weighted_f1": float(f1_score(depts, dept_preds, average="weighted", zero_division=0)),
                    "balanced_accuracy": float(balanced_accuracy_score(depts, dept_preds)),
                    "labels": dept_labels,
                    "per_class_precision": {
                        cls: float(prec)
                        for cls, prec in zip(dept_labels, dept_precision)
                    },
                    "per_class_recall": {
                        cls: float(rec)
                        for cls, rec in zip(dept_labels, dept_recall)
                    },
                    "per_class_f1": {
                        cls: float(score)
                        for cls, score in zip(dept_labels, dept_f1)
                    },
                    "confusion_matrix": confusion_matrix(depts, dept_preds, labels=dept_labels).tolist(),
                },
            }

            self._update_job(metrics=metrics)
            self._log_event("Metrics computed")

            self._check_cancel()

            self._update_job(phase="saving_bundle", message="Saving model bundle to workspace", progress=0.9)
            self._log_event("Saving model artifacts")

            bundle_dir = self.workspace / "workspace_bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            level_path = bundle_dir / "level_model.joblib"
            dept_path = bundle_dir / "dept_model.joblib"
            insights_level_path = bundle_dir / "level_insights_model.joblib"
            insights_dept_path = bundle_dir / "dept_insights_model.joblib"
            joblib.dump(level_pipeline, level_path)
            joblib.dump(dept_pipeline, dept_path)
            joblib.dump(insights_level_pipeline, insights_level_path)
            joblib.dump(insights_dept_pipeline, insights_dept_path)

            bundle_meta = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "trained_on_rows": len(filtered),
                "label_distribution": {
                    "level": {label: levels.count(label) for label in set(levels)},
                    "dept": {label: depts.count(label) for label in set(depts)},
                },
                "metrics": metrics,
                "label_policy": load_label_policy(self.workspace, self.app_state.active_modelset_id),
            }
            bundle_meta_path = bundle_dir / "bundle_meta.json"
            bundle_meta_path.write_text(json.dumps(bundle_meta, indent=2), encoding="utf-8")

            self._update_job(
                status="completed",
                phase="completed",
                message="Training completed successfully",
                progress=1.0,
                error=None,
                finished_at=self._now_iso(),
                level_model_path=str(level_path),
                dept_model_path=str(dept_path),
                level_insights_model_path=str(insights_level_path),
                dept_insights_model_path=str(insights_dept_path),
                bundle_meta_path=str(bundle_meta_path),
            )
            self._log_event(
                "Training completed",
                data={
                    "level_model_path": str(level_path),
                    "dept_model_path": str(dept_path),
                    "bundle_meta_path": str(bundle_meta_path),
                },
            )
        except Exception as exc:  # noqa: BLE001
            status = "canceled" if self._cancel_flag or "canceled" in str(exc).lower() else "error"
            self._update_job(
                status=status,
                phase="canceled" if status == "canceled" else "error",
                message="Training canceled" if status == "canceled" else "Training failed",
                error=None if status == "canceled" else str(exc),
                finished_at=self._now_iso(),
            )
            self._log_event(
                "Training canceled" if status == "canceled" else "Training failed",
                level="warning" if status == "canceled" else "error",
                data={"error": None if status == "canceled" else str(exc), "exception": exc.__class__.__name__},
            )

    def start_training(self, params: TrainingParams) -> None:
        with self._lock:
            if self.app_state.training_job.get("status") == "running":
                return
            rows = self.app_state.training_dataset.get("rows", []) if self.app_state.training_dataset else []
            stats = self._compute_stats(rows)
            self.app_state.training_job.update(
                {
                    "status": "running",
                    "progress": 0.05,
                    "phase": "starting",
                    "message": "Starting training job",
                    "events": [],
                    "params": asdict(params),
                    "stats": stats,
                    "started_at": self._now_iso(),
                    "finished_at": None,
                    "metrics": None,
                    "error": None,
                    "level_model_path": None,
                    "dept_model_path": None,
                    "bundle_meta_path": None,
                    "last_updated_at": self._now_iso(),
                }
            )
            self._cancel_flag = False

        self._log_event("Training job created", data={"params": asdict(params), "stats": stats})

        self._thread = threading.Thread(target=self._train_task, args=(params,), daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        with self._lock:
            self._cancel_flag = True
            self.app_state.training_job["status"] = "canceled"
            self.app_state.training_job["phase"] = "canceled"
            self.app_state.training_job["message"] = "Cancel requested"
            self.app_state.training_job["progress"] = 0.0
            self.app_state.training_job["finished_at"] = self._now_iso()
            self.app_state.training_job["last_updated_at"] = self._now_iso()
        self._log_event("Cancel requested", level="warning")

    def status(self) -> Dict[str, object]:
        with self._lock:
            return dict(self.app_state.training_job)

    def metrics(self) -> Optional[Dict[str, object]]:
        with self._lock:
            return self.app_state.training_job.get("metrics")

    def labeled_rows(self) -> List[Dict[str, object]]:
        rows = self.app_state.training_dataset.get("rows", []) if self.app_state.training_dataset else []
        return [
            r
            for r in rows
            if r.get("risk_text") and r.get("label_level") and r.get("label_dept")
        ]

    @staticmethod
    def _compute_basic_metrics(expected: List[str], predicted: List[str]) -> Dict[str, float]:
        if not expected or not predicted:
            return {"accuracy": 0.0, "macro_f1": 0.0}
        accuracy = sum(exp == pred for exp, pred in zip(expected, predicted)) / len(expected)
        labels = sorted(set(expected) | set(predicted))
        per_class_precision = precision_score(expected, predicted, average=None, labels=labels, zero_division=0)
        per_class_recall = recall_score(expected, predicted, average=None, labels=labels, zero_division=0)
        per_class_f1 = f1_score(expected, predicted, average=None, labels=labels, zero_division=0)
        return {
            "accuracy": float(accuracy),
            "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(expected, predicted, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
            "labels": labels,
            "per_class_precision": {label: float(value) for label, value in zip(labels, per_class_precision)},
            "per_class_recall": {label: float(value) for label, value in zip(labels, per_class_recall)},
            "per_class_f1": {label: float(value) for label, value in zip(labels, per_class_f1)},
            "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
        }

    def evaluate(
        self,
        rules: RuleService,
        vector_k_values: List[int] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, object]:
        rows = self.labeled_rows()
        if not rows:
            return {"available": False, "n_rows": 0}

        vector_k_values = vector_k_values or [1]
        labels = [str(r.get("label_level")) for r in rows]
        warnings: List[str] = []
        stratify = labels if len(set(labels)) > 1 else None
        if len(rows) * test_size < 1:
            test_size = min(0.5, max(test_size, 1 / len(rows)))

        if len(rows) < 4:
            train_rows = list(rows)
            test_rows = list(rows)
            strategy = "full_fit"
            warnings.append("Dataset too small for holdout split; evaluated on full data.")
        else:
            try:
                train_rows, test_rows = train_test_split(
                    rows,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify,
                )
                strategy = "stratified_split" if stratify else "random_split"
            except ValueError:
                train_rows, test_rows = train_test_split(
                    rows,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=None,
                )
                strategy = "random_split"
                warnings.append("Stratified split unavailable; used random split.")

        train_texts = [str(r.get("risk_text")) for r in train_rows]
        train_levels = [str(r.get("label_level")) for r in train_rows]
        train_depts = [str(r.get("label_dept")) for r in train_rows]

        test_texts = [str(r.get("risk_text")) for r in test_rows]
        expected_levels = [str(r.get("label_level")) for r in test_rows]
        expected_depts = [str(r.get("label_dept")) for r in test_rows]

        if len(set(train_levels)) < 2 or len(set(train_depts)) < 2:
            return {
                "available": False,
                "n_rows": len(rows),
                "error": "Need at least two classes per label to evaluate.",
            }

        min_level_count = min([train_levels.count(label) for label in set(train_levels)])
        min_dept_count = min([train_depts.count(label) for label in set(train_depts)])
        if min_level_count < 2 or min_dept_count < 2:
            warnings.append("Too few samples per class for calibration; used uncalibrated model.")
            level_pipeline = self._build_uncalibrated_pipeline()
            dept_pipeline = self._build_uncalibrated_pipeline()
        else:
            level_pipeline = self._build_pipeline("sigmoid")
            dept_pipeline = self._build_pipeline("sigmoid")
        level_pipeline.fit(train_texts, train_levels)
        dept_pipeline.fit(train_texts, train_depts)

        model_level_preds = [str(p) for p in level_pipeline.predict(test_texts)]
        model_dept_preds = [str(p) for p in dept_pipeline.predict(test_texts)]

        model_metrics = {
            "level": self._compute_basic_metrics(expected_levels, model_level_preds),
            "dept": self._compute_basic_metrics(expected_depts, model_dept_preds),
        }

        rules_dept_preds = []
        rules_level_preds = []
        rules_abstain = 0
        for text in test_texts:
            pred = rules.predict(text)
            dept_pred = pred.dept_pred or "None"
            rules_dept_preds.append(dept_pred)
            rules_level_preds.append("None")
            if pred.dept_pred is None:
                rules_abstain += 1
        rules_metrics = {
            "level": self._compute_basic_metrics(expected_levels, rules_level_preds),
            "dept": self._compute_basic_metrics(expected_depts, rules_dept_preds),
            "abstain_rate": rules_abstain / len(test_texts),
        }

        vector_metrics: Dict[str, object] = {}
        vector_predictions: Dict[int, Dict[str, List[str]]] = {}
        for k in vector_k_values:
            with tempfile.TemporaryDirectory(prefix="specsgrader_vector_eval_") as td:
                store = VectorStore.build(Path(td), train_rows, EmbedderConfig(), k=k)
                vector_dept_preds = []
                vector_level_preds = []
                for text in test_texts:
                    neighbors = store.query(text, k=k)
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
                    dept_pred = max(dept_counts, key=dept_counts.get) if dept_counts else "None"
                    level_pred = max(level_counts, key=level_counts.get) if level_counts else "None"
                    vector_dept_preds.append(str(dept_pred))
                    vector_level_preds.append(str(level_pred))
                vector_predictions[k] = {
                    "dept": vector_dept_preds,
                    "level": vector_level_preds,
                }
                vector_metrics[f"k_{k}"] = {
                    "level": self._compute_basic_metrics(expected_levels, vector_level_preds),
                    "dept": self._compute_basic_metrics(expected_depts, vector_dept_preds),
                }

        ensemble_level_preds = []
        ensemble_dept_preds = []
        for idx, text in enumerate(test_texts):
            method_outputs = {
                "model": {
                    "level_pred": model_level_preds[idx],
                    "level_conf": 1.0,
                    "dept_pred": model_dept_preds[idx],
                    "dept_conf": 1.0,
                },
                "rules": {
                    "level_pred": None,
                    "level_conf": 0.0,
                    "dept_pred": rules_dept_preds[idx] if rules_dept_preds[idx] != "None" else None,
                    "dept_conf": 1.0 if rules_dept_preds[idx] != "None" else 0.0,
                },
            }
            vector_pred = vector_predictions.get(vector_k_values[0])
            if vector_pred:
                method_outputs["vector"] = {
                    "level_pred": vector_pred["level"][idx],
                    "level_conf": 1.0,
                    "dept_pred": vector_pred["dept"][idx],
                    "dept_conf": 1.0,
                    "top_similarity": 1.0,
                    "margin": 1.0,
                }
            aggregated = aggregate_outputs(method_outputs)
            ensemble_level_preds.append(str(aggregated["pred_level"] or "None"))
            ensemble_dept_preds.append(str(aggregated["pred_dept"] or "None"))

        ensemble_metrics = {
            "level": self._compute_basic_metrics(expected_levels, ensemble_level_preds),
            "dept": self._compute_basic_metrics(expected_depts, ensemble_dept_preds),
        }

        bundle_dir = self.workspace / "workspace_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        level_path = bundle_dir / "level_model.joblib"
        dept_path = bundle_dir / "dept_model.joblib"
        joblib.dump(level_pipeline, level_path)
        joblib.dump(dept_pipeline, dept_path)

        with self._lock:
            self.app_state.training_job["level_model_path"] = str(level_path)
            self.app_state.training_job["dept_model_path"] = str(dept_path)

        return {
            "available": True,
            "n_rows": len(rows),
            "strategy": strategy,
            "warnings": warnings,
            "metrics": {
                "model": model_metrics,
                "rules": rules_metrics,
                "vector": vector_metrics,
                "ensemble": ensemble_metrics,
            },
        }


__all__ = ["TrainingService", "TrainingParams", "TrainingResult"]
