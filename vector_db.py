import os
import re
from dataclasses import dataclass
from typing import Any

import chromadb


def _sanitize_name(name: str) -> str:
    if not name:
        return "_default"
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return sanitized or "_default"


@dataclass
class VectorDB:
    persist_dir: str
    collection_name: str
    _client: Any
    _collection: Any
    _distance_metric: str

    @staticmethod
    def open(persist_dir: str, collection_name: str) -> "VectorDB":
        os.makedirs(persist_dir, exist_ok=True)
        safe_name = _sanitize_name(collection_name)
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(
            name=safe_name,
            metadata={"hnsw:space": "cosine"},
        )
        distance_metric = collection.metadata.get("hnsw:space", "cosine")
        return VectorDB(
            persist_dir=persist_dir,
            collection_name=safe_name,
            _client=client,
            _collection=collection,
            _distance_metric=distance_metric,
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        documents = [meta.get("text", "") for meta in metadatas]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        if top_k <= 0:
            return []
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        matches: list[dict] = []
        for match_id, metadata, document, distance in zip(
            ids, metadatas, documents, distances
        ):
            if metadata is None:
                metadata = {}
            text = metadata.get("text") or document or ""
            similarity = self._distance_to_similarity(distance)
            matches.append(
                {
                    "id": match_id,
                    "text": text,
                    "risk_level": metadata.get("risk_level", ""),
                    "review_dept": metadata.get("review_dept", ""),
                    "similarity": similarity,
                    "metadata": metadata,
                }
            )
        return matches

    def count(self) -> int:
        return self._collection.count()

    def reset_collection(self, confirm: bool) -> None:
        if not confirm:
            raise ValueError("confirm must be True to reset collection")
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self._distance_metric},
        )

    def _distance_to_similarity(self, distance: float) -> float:
        if distance is None:
            return 0.0
        if self._distance_metric == "cosine":
            return 1.0 - float(distance)
        return 1.0 / (1.0 + float(distance))
