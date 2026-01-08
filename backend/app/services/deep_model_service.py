from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from backend.app.state import AppState


@dataclass
class DeepPrediction:
    available: bool
    level_pred: Optional[str] = None
    level_conf: float = 0.0
    dept_pred: Optional[str] = None
    dept_conf: float = 0.0


def _select_labels(rows: List[Dict[str, object]], label_key: str) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for row in rows:
        text = str(row.get("risk_text") or "").strip()
        label = row.get(label_key)
        if not text or not label:
            continue
        texts.append(text)
        labels.append(str(label))
    return texts, labels


class DeepModelService:
    def __init__(self, app_state: AppState):
        self.app_state = app_state
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.level_model: Optional[LogisticRegression] = None
        self.dept_model: Optional[LogisticRegression] = None

    def available(self) -> bool:
        return self.level_model is not None and self.dept_model is not None

    def _train_if_needed(self) -> None:
        if self.available():
            return
        dataset = self.app_state.training_dataset
        if not dataset:
            return
        rows = dataset.get("rows") or []
        level_texts, level_labels = _select_labels(rows, "label_level")
        dept_texts, dept_labels = _select_labels(rows, "label_dept")
        if not level_texts or not dept_texts:
            return
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        level_matrix = vectorizer.fit_transform(level_texts)
        dept_matrix = vectorizer.transform(dept_texts)
        svd_components = min(256, max(2, level_matrix.shape[1] - 1))
        svd = TruncatedSVD(n_components=svd_components, random_state=42)
        level_vectors = svd.fit_transform(level_matrix)
        dept_vectors = svd.transform(dept_matrix)

        level_model = LogisticRegression(max_iter=300)
        dept_model = LogisticRegression(max_iter=300)
        level_model.fit(level_vectors, level_labels)
        dept_model.fit(dept_vectors, dept_labels)

        self.vectorizer = vectorizer
        self.svd = svd
        self.level_model = level_model
        self.dept_model = dept_model

    def predict(self, text: str) -> DeepPrediction:
        self._train_if_needed()
        if not self.available():
            return DeepPrediction(available=False)
        vectorizer = self.vectorizer
        svd = self.svd
        if vectorizer is None or svd is None:
            return DeepPrediction(available=False)
        matrix = vectorizer.transform([text])
        vector = svd.transform(matrix)
        level_model = self.level_model
        dept_model = self.dept_model
        if level_model is None or dept_model is None:
            return DeepPrediction(available=False)
        level_probs = level_model.predict_proba(vector)[0]
        dept_probs = dept_model.predict_proba(vector)[0]
        level_idx = int(np.argmax(level_probs))
        dept_idx = int(np.argmax(dept_probs))
        return DeepPrediction(
            available=True,
            level_pred=level_model.classes_[level_idx],
            level_conf=float(level_probs[level_idx]),
            dept_pred=dept_model.classes_[dept_idx],
            dept_conf=float(dept_probs[dept_idx]),
        )


__all__ = ["DeepModelService", "DeepPrediction"]
