from pathlib import Path

import pytest

from backend.app.vector.embedder import EmbedderConfig, default_transformer_model_path, train_embedder


def test_transformer_embedder_optional() -> None:
    pytest.importorskip("sentence_transformers")
    repo_root = Path(__file__).resolve().parents[1]
    workspace_dir = repo_root / "workspace"
    model_path = default_transformer_model_path(workspace_dir)
    if not model_path.exists():
        pytest.skip("Local transformer model not available")
    config = EmbedderConfig(model="transformer", transformer_model_path=str(model_path))
    embedder = train_embedder(["risk one", "risk two"], config)
    embeddings = embedder.embed_texts(["risk one"])
    assert embeddings.shape[0] == 1
