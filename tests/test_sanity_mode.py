import shutil
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import create_app


def _wait_for_training(client: TestClient, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    status = {}
    while time.time() < deadline:
        status = client.get("/api/train/status").json()
        if status.get("status") in {"completed", "error", "canceled"}:
            break
        time.sleep(0.05)
    return status


def test_sanity_returns_unavailable_without_model_or_data() -> None:
    client = TestClient(create_app())
    resp = client.post("/api/train/sanity")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["available"] is False
    assert payload["n_rows"] == 0


def test_sanity_after_training() -> None:
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

        start_resp = client.post("/api/train/start", json={})
        assert start_resp.status_code == 200
        status = _wait_for_training(client)
        assert status.get("status") == "completed"

        sanity_resp = client.post("/api/train/sanity")
        assert sanity_resp.status_code == 200
        payload = sanity_resp.json()
        assert payload["available"] is True
        assert payload["n_rows"] == 2
        assert 0.0 <= payload["accuracy_level"] <= 1.0
        assert 0.0 <= payload["accuracy_dept"] <= 1.0
        assert payload["accuracy_level"] >= 0.5
        assert payload["accuracy_dept"] >= 0.5
        assert payload["rows"]
    finally:
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        if backup_dir and backup_dir.exists():
            shutil.copytree(backup_dir, bundle_dir)
            shutil.rmtree(backup_dir)
