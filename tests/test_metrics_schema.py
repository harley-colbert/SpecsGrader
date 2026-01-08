import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import create_app


def _load_training_fixture(client: TestClient, name: str) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / name
    with fixture_path.open("rb") as handle:
        resp = client.post(
            "/api/data/load",
            data={"mode": "train"},
            files={"file": (fixture_path.name, handle, "text/csv")},
        )
    assert resp.status_code == 200


def _wait_for_training(client: TestClient, timeout_s: float = 5.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status_resp = client.get("/api/train/status")
        status_resp.raise_for_status()
        status = status_resp.json()
        if status.get("status") != "running":
            return status
        time.sleep(0.1)
    raise AssertionError("Training did not finish before timeout")


def _assert_metric_schema(metrics: dict) -> None:
    assert "per_class_precision" in metrics
    assert "per_class_recall" in metrics
    assert "per_class_f1" in metrics
    assert "weighted_f1" in metrics
    labels = metrics.get("labels", [])
    confusion = metrics.get("confusion_matrix", [])
    assert isinstance(labels, list)
    assert len(confusion) == len(labels)
    assert all(len(row) == len(labels) for row in confusion)


def test_training_metrics_schema() -> None:
    client = TestClient(create_app())
    _load_training_fixture(client, "training_balanced.csv")
    start_resp = client.post("/api/train/start", json={})
    assert start_resp.status_code == 200
    status = _wait_for_training(client)
    assert status.get("status") == "completed"
    metrics = status.get("metrics", {})
    _assert_metric_schema(metrics.get("level", {}))
    _assert_metric_schema(metrics.get("dept", {}))
