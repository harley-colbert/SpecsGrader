from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np


class ModelInsightsService:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def get_insights(self, bundle_dir: Path, top_n: int = 20) -> Dict[str, Any]:
        level_path = bundle_dir / "level_insights_model.joblib"
        dept_path = bundle_dir / "dept_insights_model.joblib"
        if not level_path.exists() or not dept_path.exists():
            raise FileNotFoundError("Insights models not found in bundle")
        return {
            "level": self._extract_top_terms(level_path, top_n),
            "dept": self._extract_top_terms(dept_path, top_n),
        }

    def _extract_top_terms(self, model_path: Path, top_n: int) -> Dict[str, Any]:
        pipeline = joblib.load(model_path)
        vectorizer = pipeline.named_steps.get("tfidf")
        classifier = pipeline.named_steps.get("clf")
        if vectorizer is None or classifier is None:
            raise ValueError("Insights model is missing expected pipeline steps")

        feature_names = vectorizer.get_feature_names_out()
        classes = list(classifier.classes_)
        coef = classifier.coef_

        weights_by_class: Dict[str, np.ndarray] = {}
        if coef.shape[0] == 1 and len(classes) == 2:
            weights_by_class[classes[1]] = coef[0]
            weights_by_class[classes[0]] = -coef[0]
        else:
            for idx, label in enumerate(classes):
                weights_by_class[label] = coef[idx]

        top_terms: Dict[str, List[Dict[str, float]]] = {}
        for label, weights in weights_by_class.items():
            sorted_idx = np.argsort(weights)[::-1]
            entries = []
            for idx in sorted_idx:
                weight = float(weights[idx])
                if weight <= 0:
                    continue
                entries.append({"term": str(feature_names[idx]), "weight": weight})
                if len(entries) >= top_n:
                    break
            top_terms[label] = entries

        metadata = {
            "top_n": top_n,
            "ngram_range": list(vectorizer.ngram_range),
            "max_features": vectorizer.max_features,
        }
        return {
            "labels": classes,
            "metadata": metadata,
            "top_terms": top_terms,
        }
