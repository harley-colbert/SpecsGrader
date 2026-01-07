import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_evaluate_returns_unavailable_without_data() -> None:
    client = TestClient(create_app())
    resp = client.post("/api/train/evaluate")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["available"] is False


def test_evaluate_after_load() -> None:
    app = create_app()
    client = TestClient(app)

    training_path = Path(__file__).resolve().parent / "fixtures" / "training_sample.csv"
    bundle_dir = Path(__file__).resolve().parents[1] / "workspace" / "workspace_bundle"
    backup_dir = None

    if bundle_dir.exists():
        backup_dir = bundle_dir.with_name(f"{bundle_dir.name}_backup")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(bundle_dir, backup_dir)

    try:
        load_train = client.post("/api/data/load", data={"mode": "train", "path": str(training_path)})
        assert load_train.status_code == 200

        eval_resp = client.post("/api/train/evaluate", json={"vector_k_values": [1]})
        assert eval_resp.status_code == 200
        payload = eval_resp.json()
        assert payload["available"] is True
        metrics = payload["metrics"]
        assert "model" in metrics
        assert "rules" in metrics
        assert "vector" in metrics
        assert "ensemble" in metrics
        assert (bundle_dir / "level_model.joblib").exists()
        assert (bundle_dir / "dept_model.joblib").exists()
    finally:
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        if backup_dir and backup_dir.exists():
            shutil.copytree(backup_dir, bundle_dir)
            shutil.rmtree(backup_dir)
