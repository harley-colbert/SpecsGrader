import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

from .embedder import Embedder, EmbedderConfig


class VectorStore:
    def __init__(self, root: Path):
        self.root = root
        self.embeddings_path = root / "embeddings.npy"
        self.rows_path = root / "rows.jsonl"
        self.index_path = root / "index.joblib"
        self.embedder = Embedder.load(root)
        self.embeddings = np.load(self.embeddings_path)
        self.rows = [json.loads(line) for line in self.rows_path.read_text(encoding="utf-8").splitlines() if line]
        self.index: NearestNeighbors = joblib.load(self.index_path)

    @classmethod
    def build(
        cls,
        root: Path,
        rows: List[Dict[str, str]],
        embedder_cfg: EmbedderConfig | None = None,
        k: int = 5,
    ) -> "VectorStore":
        root.mkdir(parents=True, exist_ok=True)
        texts = [str(r.get("risk_text", "")) for r in rows]
        embedder = Embedder.train(texts, embedder_cfg)
        embeddings = embedder.embed_texts(texts)

        index = NearestNeighbors(metric="cosine", algorithm="brute")
        index.fit(embeddings)

        np.save(root / "embeddings.npy", embeddings)
        with (root / "rows.jsonl").open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row) + "\n")
        joblib.dump(index, root / "index.joblib")
        embedder.save(root)
        return cls(root)

    def query(self, text: str, k: int = 5) -> List[Dict[str, object]]:
        vector = self.embedder.embed_texts([text])
        distances, indices = self.index.kneighbors(vector, n_neighbors=min(k, len(self.rows)))
        neighbors = []
        for dist, idx in zip(distances[0], indices[0]):
            neighbors.append(
                {
                    "distance": float(dist),
                    "similarity": float(1 - dist),
                    "row": self.rows[int(idx)],
                }
            )
        return neighbors


__all__ = ["VectorStore"]
