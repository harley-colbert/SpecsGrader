import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class EmbedderConfig:
    model: str = "tfidf"
    max_features: int = 5000
    ngram_range: tuple = (1, 2)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmbedderConfig":
        return cls(
            model=data.get("model", "tfidf"),
            max_features=int(data.get("max_features", 5000)),
            ngram_range=tuple(data.get("ngram_range", (1, 2))),
        )


class Embedder:
    def __init__(self, config: EmbedderConfig, vectorizer: TfidfVectorizer):
        self.config = config
        self.vectorizer = vectorizer

    @classmethod
    def train(cls, texts: List[str], config: EmbedderConfig | None = None) -> "Embedder":
        cfg = config or EmbedderConfig()
        vectorizer = TfidfVectorizer(
            max_features=cfg.max_features,
            ngram_range=cfg.ngram_range,
        )
        vectorizer.fit(texts)
        return cls(cfg, vectorizer)

    @classmethod
    def load(cls, path: Path) -> "Embedder":
        config_path = path / "vector_embedder.json"
        vectorizer_path = path / "vectorizer.joblib"
        cfg = EmbedderConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
        vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)
        return cls(cfg, vectorizer)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "vector_embedder.json").write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")
        joblib.dump(self.vectorizer, path / "vectorizer.joblib")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()


__all__ = ["Embedder", "EmbedderConfig"]
