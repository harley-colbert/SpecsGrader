import hashlib
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from backend.app.services.rule_service import RuleService
from backend.app.services.aggregate_service import aggregate_outputs
from backend.app.config.xlsx_contract import RISK_LEVELS, RISK_ORDER, normalize_risk_level
from backend.app.services.ingest_service import DEPARTMENTS
from backend.app.label_policy import load_label_policy
from backend.app.vector.vector_store import VectorStore
from backend.app.vector.embedder import EmbedderConfig


@dataclass
class TrainingParams:
    oversample_enabled: bool = False
    oversample_cap_ratio: float = 0.3
    min_recall_per_class: float = 0.5
    calibration_method: str = "sigmoid"
    cv_folds: int = 5
    use_class_weight_balanced: bool = True


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

    @staticmethod
    def _training_text(row: Dict[str, object]) -> str:
        return str(row.get("spec_text") or "").strip()

    @staticmethod
    def _label_value(row: Dict[str, object], label_key: str) -> str:
        if label_key == "label_level":
            return normalize_risk_level(row.get(label_key)) or ""
        return str(row.get(label_key) or "").strip().lower()

    @classmethod
    def _valid_rows(
        cls,
        rows: List[Dict[str, object]],
        label_key: str,
        valid_labels: set[str],
    ) -> List[Dict[str, object]]:
        valid_rows = []
        for row in rows:
            text = cls._training_text(row)
            if not text:
                continue
            label_value = cls._label_value(row, label_key)
            if label_value in valid_labels:
                normalized = dict(row)
                normalized["spec_text"] = text
                normalized[label_key] = label_value
                valid_rows.append(normalized)
        return valid_rows

    def _compute_stats(self, rows: List[Dict[str, object]]) -> Dict[str, Any]:
        total_rows = len(rows)
        missing_risk_text = 0
        missing_level = 0
        missing_dept = 0
        labeled_rows: List[Dict[str, object]] = []
        level_rows = self._valid_rows(rows, "label_level", RISK_LEVELS)
        dept_rows = self._valid_rows(rows, "label_dept", DEPARTMENTS)
        for r in rows:
            text = self._training_text(r)
            if not text:
                missing_risk_text += 1
            if text and self._label_value(r, "label_level") not in RISK_LEVELS:
                missing_level += 1
            if text and self._label_value(r, "label_dept") not in DEPARTMENTS:
                missing_dept += 1
            if text and self._label_value(r, "label_level") in RISK_LEVELS and self._label_value(r, "label_dept") in DEPARTMENTS:
                labeled_rows.append(r)

        label_distribution_level = self._label_distribution(level_rows, "label_level")
        label_distribution_dept = self._label_distribution(dept_rows, "label_dept")
        expected_levels = list(RISK_ORDER)
        expected_depts = sorted(DEPARTMENTS)
        ordered_level = self._ordered_distribution(label_distribution_level, expected_levels)
        ordered_dept = self._ordered_distribution(label_distribution_dept, expected_depts)
        warnings: List[str] = []
        blocking_errors: List[str] = []

        if total_rows == 0:
            blocking_errors.append("No rows loaded.")
        if missing_risk_text == total_rows and total_rows > 0:
            blocking_errors.append("All rows are missing risk text.")
        if level_rows == [] and total_rows > 0 and missing_risk_text < total_rows:
            blocking_errors.append("All rows are missing risk level labels.")
        if dept_rows == [] and total_rows > 0 and missing_risk_text < total_rows:
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

    @staticmethod
    def _dataset_snapshot_hash(rows: List[Dict[str, object]]) -> str:
        if not rows:
            return ""
        payload = json.dumps(rows, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _artifact_inventory(self, bundle_dir: Path) -> List[str]:
        if not bundle_dir.exists():
            return []
        return [p.name for p in sorted(bundle_dir.iterdir()) if p.is_file()]

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
    def _build_pipeline(calibration_method: str, use_class_weight_balanced: bool) -> Pipeline:
        class_weight = "balanced" if use_class_weight_balanced else None
        base = LogisticRegression(max_iter=1000, class_weight=class_weight, n_jobs=1)
        calibrated = CalibratedClassifierCV(estimator=base, method=calibration_method, cv=2)
        return Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("clf", calibrated),
            ]
        )

    @staticmethod
    def _build_uncalibrated_pipeline(use_class_weight_balanced: bool) -> Pipeline:
        class_weight = "balanced" if use_class_weight_balanced else None
        base = LogisticRegression(max_iter=1000, class_weight=class_weight, n_jobs=1)
        return Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("clf", base),
            ]
        )

    @staticmethod
    def _compute_detailed_metrics(expected: List[str], predicted: List[str]) -> Dict[str, object]:
        if not expected or not predicted:
            return {
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "balanced_accuracy": 0.0,
                "labels": [],
                "per_class_precision": {},
                "per_class_recall": {},
                "per_class_f1": {},
                "confusion_matrix": [],
            }
        labels = sorted(set(expected) | set(predicted))
        precision = precision_score(expected, predicted, average=None, labels=labels, zero_division=0)
        recall = recall_score(expected, predicted, average=None, labels=labels, zero_division=0)
        f1 = f1_score(expected, predicted, average=None, labels=labels, zero_division=0)
        return {
            "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(expected, predicted, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
            "labels": labels,
            "per_class_precision": {label: float(value) for label, value in zip(labels, precision)},
            "per_class_recall": {label: float(value) for label, value in zip(labels, recall)},
            "per_class_f1": {label: float(value) for label, value in zip(labels, f1)},
            "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
        }

    def _summarize_cv_metrics(
        self, fold_metrics: List[Dict[str, object]], labels: List[str]
    ) -> Dict[str, object]:
        if not fold_metrics:
            return {"averages": {}, "folds": [], "labels": labels, "confusion_matrix": []}
        averages = {
            "macro_f1": float(np.mean([fold["macro_f1"] for fold in fold_metrics])),
            "weighted_f1": float(np.mean([fold["weighted_f1"] for fold in fold_metrics])),
            "balanced_accuracy": float(np.mean([fold["balanced_accuracy"] for fold in fold_metrics])),
        }
        per_class_precision = {label: [] for label in labels}
        per_class_recall = {label: [] for label in labels}
        per_class_f1 = {label: [] for label in labels}
        confusion = np.zeros((len(labels), len(labels)), dtype=float)
        for fold in fold_metrics:
            for label in labels:
                per_class_precision[label].append(float(fold["per_class_precision"].get(label, 0.0)))
                per_class_recall[label].append(float(fold["per_class_recall"].get(label, 0.0)))
                per_class_f1[label].append(float(fold["per_class_f1"].get(label, 0.0)))
            confusion += np.array(fold.get("confusion_matrix") or np.zeros_like(confusion), dtype=float)
        return {
            "averages": averages,
            "folds": fold_metrics,
            "labels": labels,
            "per_class_precision": {label: float(np.mean(values)) for label, values in per_class_precision.items()},
            "per_class_recall": {label: float(np.mean(values)) for label, values in per_class_recall.items()},
            "per_class_f1": {label: float(np.mean(values)) for label, values in per_class_f1.items()},
            "confusion_matrix": confusion.tolist(),
        }

    def _run_cv(
        self,
        texts: List[str],
        labels: List[str],
        params: TrainingParams,
        label_name: str,
    ) -> Dict[str, object]:
        folds = max(2, int(params.cv_folds))
        counts = {label: labels.count(label) for label in set(labels)}
        min_count = min(counts.values()) if counts else 0
        if min_count < 2:
            return {"averages": {}, "folds": [], "labels": sorted(set(labels)), "confusion_matrix": []}
        if folds > min_count:
            folds = min_count
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        labels_all = sorted(set(labels))
        fold_metrics: List[Dict[str, object]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels), start=1):
            self._check_cancel()
            self._update_job(
                phase=f"cv_{label_name}",
                message=f"Cross-validation ({label_name}) fold {fold_idx} of {folds}",
                progress=0.28 + (0.2 * (fold_idx / folds)),
                cv_status={
                    "label": label_name,
                    "current_fold": fold_idx,
                    "total_folds": folds,
                },
            )
            pipeline = self._build_uncalibrated_pipeline(params.use_class_weight_balanced)
            train_texts = [texts[i] for i in train_idx]
            test_texts = [texts[i] for i in test_idx]
            train_labels = [labels[i] for i in train_idx]
            test_labels = [labels[i] for i in test_idx]
            pipeline.fit(train_texts, train_labels)
            preds = pipeline.predict(test_texts)
            metrics = self._compute_detailed_metrics(test_labels, list(preds))
            fold_metrics.append(metrics)
            summary = self._summarize_cv_metrics(fold_metrics, labels_all)
            self._update_job(
                cv_status={
                    "label": label_name,
                    "current_fold": fold_idx,
                    "total_folds": folds,
                    "averages": summary.get("averages"),
                }
            )
        return self._summarize_cv_metrics(fold_metrics, labels_all)

    def _train_task(self, params: TrainingParams) -> None:
        try:
            rows = self.app_state.training_dataset.get("rows", []) if self.app_state.training_dataset else []

            self._update_job(phase="validating_data", message="Validating training data", progress=0.12)
            self._log_event("Validating training data")

            level_rows = self._valid_rows(rows, "label_level", RISK_LEVELS)
            dept_rows = self._valid_rows(rows, "label_dept", DEPARTMENTS)
            if not level_rows or not dept_rows:
                raise ValueError("No labeled training data loaded")

            pre_dist = {
                "level": self._label_distribution(level_rows, "label_level"),
                "dept": self._label_distribution(dept_rows, "label_dept"),
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
                level_rows = self.oversample_rows(level_rows, "label_level", params.oversample_cap_ratio)
                dept_rows = self.oversample_rows(dept_rows, "label_dept", params.oversample_cap_ratio)

            self._check_cancel()

            self._update_job(phase="ensuring_minimums", message="Ensuring minimum examples per class", progress=0.24)
            level_rows = self.ensure_minimum_per_class(level_rows, "label_level", min_required=2)
            dept_rows = self.ensure_minimum_per_class(dept_rows, "label_dept", min_required=2)

            post_dist = {
                "level": self._label_distribution(level_rows, "label_level"),
                "dept": self._label_distribution(dept_rows, "label_dept"),
            }
            trained_row_count = min(len(level_rows), len(dept_rows))
            self._update_job(stats={**(self.app_state.training_job.get("stats") or {}), "training_distribution": post_dist, "trained_on_rows": trained_row_count})
            self._log_event("Final training distribution prepared", data=post_dist)

            self._check_cancel()

            level_texts = [self._training_text(r) for r in level_rows]
            level_labels = [str(r.get("label_level")) for r in level_rows]
            dept_texts = [self._training_text(r) for r in dept_rows]
            dept_labels = [str(r.get("label_dept")) for r in dept_rows]
            dataset_snapshot_hash = self._dataset_snapshot_hash(level_rows + dept_rows)

            insights_level_pipeline = self._build_uncalibrated_pipeline(params.use_class_weight_balanced)
            insights_dept_pipeline = self._build_uncalibrated_pipeline(params.use_class_weight_balanced)
            insights_level_pipeline.fit(level_texts, level_labels)
            insights_dept_pipeline.fit(dept_texts, dept_labels)

            # Build initial calibrated pipelines
            level_pipeline = self._build_pipeline(params.calibration_method, params.use_class_weight_balanced)
            dept_pipeline = self._build_pipeline(params.calibration_method, params.use_class_weight_balanced)

            self._update_job(
                phase="cv_level",
                message=f"Running {params.cv_folds}-fold CV for risk level model",
                progress=0.28,
                cv_status={"label": "level", "current_fold": 0, "total_folds": params.cv_folds},
            )
            self._log_event("Cross-validation started", data={"label": "level", "folds": params.cv_folds})
            cv_level = self._run_cv(level_texts, level_labels, params, "level")

            self._check_cancel()

            self._update_job(
                phase="cv_dept",
                message=f"Running {params.cv_folds}-fold CV for department model",
                progress=0.48,
                cv_status={"label": "dept", "current_fold": 0, "total_folds": params.cv_folds},
            )
            self._log_event("Cross-validation started", data={"label": "dept", "folds": params.cv_folds})
            cv_dept = self._run_cv(dept_texts, dept_labels, params, "dept")

            cv_metrics = {"level": cv_level, "dept": cv_dept}
            self._update_job(cv_metrics=cv_metrics, cv_status=None)
            self._log_event("Cross-validation completed", data={"cv_metrics": cv_metrics})

            # Fit level model with graceful fallback if calibration is not feasible
            self._update_job(
                phase="fit_level_model",
                message="Training risk level model (TF-IDF + calibrated logistic regression)",
                progress=0.32,
            )
            self._log_event("Fitting risk level model")
            try:
                level_pipeline.fit(level_texts, level_labels)
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
                    level_pipeline = self._build_uncalibrated_pipeline(params.use_class_weight_balanced)
                    level_pipeline.fit(level_texts, level_labels)
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
                dept_pipeline.fit(dept_texts, dept_labels)
            except ValueError as exc:
                if "less than 2 examples for at least one class" in str(exc):
                    self._log_event(
                        "Calibration for dept model skipped due to small sample size; "
                        "falling back to uncalibrated LogisticRegression",
                        data={"error": str(exc)},
                    )
                    dept_pipeline = self._build_uncalibrated_pipeline(params.use_class_weight_balanced)
                    dept_pipeline.fit(dept_texts, dept_labels)
                else:
                    raise

            self._check_cancel()

            self._update_job(phase="evaluating", message="Evaluating training set performance", progress=0.78)
            self._log_event("Evaluating models")

            level_preds = level_pipeline.predict(level_texts)
            dept_preds = dept_pipeline.predict(dept_texts)
            metrics = {
                "level": self._compute_detailed_metrics(level_labels, list(level_preds)),
                "dept": self._compute_detailed_metrics(dept_labels, list(dept_preds)),
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

            artifacts = self._artifact_inventory(bundle_dir)
            if "bundle_meta.json" not in artifacts:
                artifacts.append("bundle_meta.json")
            bundle_meta = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "training_record": {
                    "trained_at": self._now_iso(),
                    "dataset_snapshot_hash": dataset_snapshot_hash,
                    "training_params": asdict(params),
                    "cv_summary": {
                        "level": (cv_metrics.get("level") or {}).get("averages", {}),
                        "dept": (cv_metrics.get("dept") or {}).get("averages", {}),
                    },
                    "artifacts": artifacts,
                },
                "trained_on_rows": trained_row_count,
                "label_distribution": {
                    "level": {label: level_labels.count(label) for label in set(level_labels)},
                    "dept": {label: dept_labels.count(label) for label in set(dept_labels)},
                },
                "imbalance": {
                    "class_weight_balanced": params.use_class_weight_balanced,
                    "oversample_enabled": params.oversample_enabled,
                    "oversample_cap_ratio": params.oversample_cap_ratio,
                    "pre_distribution": pre_dist,
                    "post_distribution": post_dist,
                },
                "metrics": metrics,
                "cv_metrics": self.app_state.training_job.get("cv_metrics"),
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
                    "cv_metrics": None,
                    "cv_status": None,
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
            if self._training_text(r)
            and self._label_value(r, "label_level") in RISK_LEVELS
            and self._label_value(r, "label_dept") in DEPARTMENTS
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
        labels = [self._label_value(r, "label_level") for r in rows]
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

        train_texts = [self._training_text(r) for r in train_rows]
        train_levels = [self._label_value(r, "label_level") for r in train_rows]
        train_depts = [self._label_value(r, "label_dept") for r in train_rows]

        test_texts = [self._training_text(r) for r in test_rows]
        expected_levels = [self._label_value(r, "label_level") for r in test_rows]
        expected_depts = [self._label_value(r, "label_dept") for r in test_rows]

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
            level_pipeline = self._build_uncalibrated_pipeline(True)
            dept_pipeline = self._build_uncalibrated_pipeline(True)
        else:
            level_pipeline = self._build_pipeline("sigmoid", True)
            dept_pipeline = self._build_pipeline("sigmoid", True)
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
