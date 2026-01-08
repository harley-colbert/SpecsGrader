import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


@dataclass
class EmbedderConfig:
    model: str = "tfidf"
    max_features: int = 5000
    ngram_range: tuple = (1, 2)
    svd_components: int = 256

    def to_dict(self) -> dict:
        payload = {
            "model": self.model,
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
        }
        if self.model == "lsa":
            payload["svd_components"] = self.svd_components
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "EmbedderConfig":
        return cls(
            model=data.get("model", "tfidf"),
            max_features=int(data.get("max_features", 5000)),
            ngram_range=tuple(data.get("ngram_range", (1, 2))),
            svd_components=int(data.get("svd_components", 256)),
        )


class BaseEmbedder:
    def __init__(self, config: EmbedderConfig, vectorizer: TfidfVectorizer):
        self.config = config
        self.vectorizer = vectorizer
        self.svd: Optional[TruncatedSVD] = None

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "vector_embedder.json").write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")
        joblib.dump(self.vectorizer, path / "vectorizer.joblib")
        if self.svd is not None:
            joblib.dump(self.svd, path / "svd.joblib")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(BaseEmbedder):
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()


class LsaEmbedder(BaseEmbedder):
    def __init__(self, config: EmbedderConfig, vectorizer: TfidfVectorizer, svd: TruncatedSVD):
        super().__init__(config, vectorizer)
        self.svd = svd

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        tfidf = self.vectorizer.transform(texts)
        return self.svd.transform(tfidf)


def train_embedder(texts: List[str], config: EmbedderConfig | None = None) -> BaseEmbedder:
    cfg = config or EmbedderConfig()
    vectorizer = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
    )
    tfidf = vectorizer.fit_transform(texts)
    if cfg.model == "lsa":
        svd = TruncatedSVD(n_components=cfg.svd_components, random_state=42)
        svd.fit(tfidf)
        return LsaEmbedder(cfg, vectorizer, svd)
    return TfidfEmbedder(cfg, vectorizer)


def load_embedder(path: Path) -> BaseEmbedder:
    config_path = path / "vector_embedder.json"
    vectorizer_path = path / "vectorizer.joblib"
    if not config_path.exists():
        cfg = EmbedderConfig()
    else:
        cfg = EmbedderConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)
    if cfg.model == "lsa":
        svd_path = path / "svd.joblib"
        if not svd_path.exists():
            cfg = EmbedderConfig(model="tfidf", max_features=cfg.max_features, ngram_range=cfg.ngram_range)
            return TfidfEmbedder(cfg, vectorizer)
        svd: TruncatedSVD = joblib.load(svd_path)
        return LsaEmbedder(cfg, vectorizer, svd)
    return TfidfEmbedder(cfg, vectorizer)


__all__ = [
    "EmbedderConfig",
    "BaseEmbedder",
    "TfidfEmbedder",
    "LsaEmbedder",
    "train_embedder",
    "load_embedder",
]
