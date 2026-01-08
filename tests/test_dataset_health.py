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


def test_dataset_health_balanced_distribution():
    client = TestClient(create_app())
    _load_training_fixture(client, "training_balanced.csv")
    resp = client.get("/api/data/health?mode=train")
    assert resp.status_code == 200
    payload = resp.json()
    assert "label_distribution" in payload
    assert "label_distribution_pct" in payload
    assert "level" in payload["label_distribution"]
    assert "dept" in payload["label_distribution"]
    assert payload["blocking_errors"] == []
    assert payload["warnings"] == []


def test_dataset_health_missing_and_rare_classes_warn():
    client = TestClient(create_app())
    _load_training_fixture(client, "training_missing_extreme.csv")
    resp = client.get("/api/data/health?mode=train")
    assert resp.status_code == 200
    payload = resp.json()
    warnings = payload.get("warnings", [])
    assert payload["label_distribution"]["level"]["extreme"] == 0
    assert any("Missing risk level classes" in warning for warning in warnings)
    assert any("Rare risk level classes" in warning for warning in warnings)


def test_dataset_health_unlabeled_is_blocking():
    client = TestClient(create_app())
    _load_training_fixture(client, "training_unlabeled.csv")
    resp = client.get("/api/data/health?mode=train")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["labeled_rows"] == 0
    assert any("No fully labeled rows" in error for error in payload.get("blocking_errors", []))
