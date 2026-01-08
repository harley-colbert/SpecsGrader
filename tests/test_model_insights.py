import time
from pathlib import Path

from fastapi.testclient import TestClient
from uuid import uuid4

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


def test_model_insights_top_terms() -> None:
    client = TestClient(create_app())
    _load_training_fixture(client, "insights_synthetic.csv")

    start_resp = client.post("/api/train/start", json={})
    assert start_resp.status_code == 200
    status = _wait_for_training(client)
    assert status.get("status") == "completed"

    modelset_id = f"insights_{uuid4().hex[:8]}"
    create_resp = client.post(
        "/api/modelsets",
        json={"modelset_id": modelset_id, "name": "Insights Test", "description": ""},
    )
    assert create_resp.status_code == 200

    save_resp = client.post(
        f"/api/modelsets/{modelset_id}/versions",
        json={"notes": "insights"},
    )
    assert save_resp.status_code == 200

    insights_resp = client.get(f"/api/modelsets/{modelset_id}/insights?top_n=5")
    assert insights_resp.status_code == 200
    payload = insights_resp.json()
    insights = payload["insights"]

    level_terms = insights["level"]["top_terms"]
    dept_terms = insights["dept"]["top_terms"]

    assert any(term["term"] == "low" for term in level_terms.get("low", []))
    assert any(term["term"] == "high" for term in level_terms.get("high", []))
    assert any(term["term"] == "bearing" for term in dept_terms.get("mechanical", []))
    assert any(term["term"] == "circuit" for term in dept_terms.get("electrical", []))
