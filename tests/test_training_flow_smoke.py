from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_training_vector_rules_smoke_contracts():
    app = create_app()
    client = TestClient(app)

    rules_resp = client.get("/api/rules/get")
    assert rules_resp.status_code == 200
    assert "departments" in rules_resp.json()

    metrics_resp = client.get("/api/train/metrics")
    assert metrics_resp.status_code == 200
    assert metrics_resp.json()["available"] is False

    vector_status = client.get("/api/vector/status")
    assert vector_status.status_code == 200
    assert vector_status.json()["available"] is False

    modelset_id = f"smoke_{uuid4().hex[:8]}"
    create_resp = client.post(
        "/api/modelsets",
        json={"modelset_id": modelset_id, "name": "Smoke Test", "description": ""},
    )
    assert create_resp.status_code == 200

    save_resp = client.post(
        f"/api/modelsets/{modelset_id}/versions",
        json={"notes": "smoke"},
    )
    assert save_resp.status_code == 200
    version_id = save_resp.json()["version"]["version_id"]

    load_resp = client.post(
        f"/api/modelsets/{modelset_id}/load",
        json={"version_id": version_id},
    )
    assert load_resp.status_code == 200
    assert load_resp.json()["loaded"] is True

    state_resp = client.get("/api/state")
    assert state_resp.status_code == 200
    state = state_resp.json()
    assert state["active_modelset_id"] == modelset_id
    assert state["active_modelset_version_id"] == version_id
