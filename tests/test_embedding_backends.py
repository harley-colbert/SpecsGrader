import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app.vector.embedder import EmbedderConfig
from backend.app.vector.vector_store import VectorStore
from backend.app.main import create_app


def _rows() -> list[dict[str, str]]:
    return [
        {"risk_text": "Mechanical risk about bearings", "label_level": "low", "label_dept": "mechanical"},
        {"risk_text": "Electrical risk about panels", "label_level": "high", "label_dept": "electrical"},
        {"risk_text": "Controls risk about plc", "label_level": "medium", "label_dept": "controls"},
        {"risk_text": "Project schedule risk", "label_level": "medium", "label_dept": "project_management"},
    ]


def test_tfidf_embedder_build_and_query(tmp_path: Path) -> None:
    root = tmp_path / "tfidf_store"
    store = VectorStore.build(root, _rows(), EmbedderConfig(model="tfidf"))
    assert store.embeddings.shape[0] == len(_rows())
    neighbors = store.query("bearing issue", k=2)
    assert len(neighbors) == 2
    config = json.loads((root / "vector_embedder.json").read_text(encoding="utf-8"))
    assert config["model"] == "tfidf"

    loaded = VectorStore(root)
    assert np.array_equal(store.embeddings, loaded.embeddings)


def test_lsa_embedder_build_and_query(tmp_path: Path) -> None:
    root = tmp_path / "lsa_store"
    config = EmbedderConfig(model="lsa", svd_components=2)
    store = VectorStore.build(root, _rows(), config)
    assert store.embeddings.shape == (len(_rows()), 2)
    neighbors = store.query("schedule risk", k=2)
    assert len(neighbors) == 2
    assert (root / "svd.joblib").exists()

    loaded = VectorStore(root)
    assert loaded.embeddings.shape == (len(_rows()), 2)


def test_embeddings_backends_endpoint() -> None:
    client = TestClient(create_app())
    resp = client.get("/api/embeddings/backends")
    assert resp.status_code == 200
    data = resp.json()
    backends = data.get("backends", {})
    assert backends.get("tfidf", {}).get("available") is True
    assert backends.get("lsa", {}).get("available") is True
    assert "transformer" in backends
    transformer = backends.get("transformer", {})
    assert "available" in transformer
    if transformer.get("available") is False:
        assert transformer.get("reason")
