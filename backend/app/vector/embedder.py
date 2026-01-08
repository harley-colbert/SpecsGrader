import json
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class EmbedderConfig:
    model: str = "tfidf"
    max_features: int = 5000
    ngram_range: tuple = (1, 2)
    svd_components: int = 256
    transformer_model_path: Optional[str] = None

    def to_dict(self) -> dict:
        payload = {
            "model": self.model,
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
        }
        if self.model == "lsa":
            payload["svd_components"] = self.svd_components
        if self.model == "transformer":
            payload["transformer_model_path"] = self.transformer_model_path
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "EmbedderConfig":
        return cls(
            model=data.get("model", "tfidf"),
            max_features=int(data.get("max_features", 5000)),
            ngram_range=tuple(data.get("ngram_range", (1, 2))),
            svd_components=int(data.get("svd_components", 256)),
            transformer_model_path=data.get("transformer_model_path"),
        )


class BaseEmbedder:
    def __init__(self, config: EmbedderConfig, vectorizer: Optional[TfidfVectorizer] = None):
        self.config = config
        self.vectorizer = vectorizer
        self.svd: Optional[TruncatedSVD] = None

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "vector_embedder.json").write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, path / "vectorizer.joblib")
        if self.svd is not None and self.vectorizer is not None:
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


class TransformerEmbedder(BaseEmbedder):
    def __init__(self, config: EmbedderConfig, model_path: Path):
        super().__init__(config, None)
        self.model_path = model_path
        self.model = _load_sentence_transformer(model_path)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


TRANSFORMER_MODEL_SUBDIR = Path("models/sentence_transformers/all-MiniLM-L6-v2")


def default_transformer_model_path(workspace_dir: Path) -> Path:
    return workspace_dir / TRANSFORMER_MODEL_SUBDIR


def transformer_backend_status(workspace_dir: Path, model_path: Optional[str] = None) -> dict:
    if find_spec("sentence_transformers") is None:
        return {
            "available": False,
            "reason": "Missing optional dependency: sentence-transformers",
            "missing": "dependency",
        }
    resolved_path = Path(model_path) if model_path else default_transformer_model_path(workspace_dir)
    if not resolved_path.exists():
        return {
            "available": False,
            "reason": f"Missing model files at {resolved_path}",
            "missing": "model_files",
        }
    return {"available": True, "model_path": str(resolved_path)}


def embedding_backends(workspace_dir: Path) -> dict:
    transformer = transformer_backend_status(workspace_dir)
    return {
        "tfidf": {"available": True},
        "lsa": {"available": True},
        "transformer": transformer,
    }


def _load_sentence_transformer(model_path: Path):
    module = import_module("sentence_transformers")
    return module.SentenceTransformer(str(model_path))


def train_embedder(texts: List[str], config: EmbedderConfig | None = None) -> BaseEmbedder:
    cfg = config or EmbedderConfig()
    vectorizer = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
    )
    if cfg.model == "lsa":
        tfidf = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=cfg.svd_components, random_state=42)
        svd.fit(tfidf)
        return LsaEmbedder(cfg, vectorizer, svd)
    if cfg.model == "transformer":
        if not cfg.transformer_model_path:
            raise RuntimeError("Transformer model path is required for transformer embeddings.")
        model_path = Path(cfg.transformer_model_path)
        if not model_path.exists():
            raise RuntimeError(f"Transformer model path not found: {model_path}")
        return TransformerEmbedder(cfg, model_path)
    tfidf = vectorizer.fit_transform(texts)
    return TfidfEmbedder(cfg, vectorizer)


def load_embedder(path: Path) -> BaseEmbedder:
    config_path = path / "vector_embedder.json"
    if not config_path.exists():
        cfg = EmbedderConfig()
    else:
        cfg = EmbedderConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    if cfg.model == "transformer":
        if not cfg.transformer_model_path:
            raise RuntimeError("Transformer model path missing from embedder config.")
        model_path = Path(cfg.transformer_model_path)
        if not model_path.exists():
            raise RuntimeError(f"Transformer model path not found: {model_path}")
        return TransformerEmbedder(cfg, model_path)
    vectorizer_path = path / "vectorizer.joblib"
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
    "TransformerEmbedder",
    "default_transformer_model_path",
    "transformer_backend_status",
    "embedding_backends",
    "train_embedder",
    "load_embedder",
]
