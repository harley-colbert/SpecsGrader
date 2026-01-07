from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_modelset_routes_crud_and_guard():
    app = create_app()
    client = TestClient(app)

    modelset_id = f"route_{uuid4().hex[:8]}"
    create_resp = client.post(
        "/api/modelsets",
        json={
            "modelset_id": modelset_id,
            "name": "Route Test",
            "description": "Initial",
            "tags": ["qa", "phase2"],
        },
    )
    assert create_resp.status_code == 200
    created = create_resp.json()["modelset"]
    assert created["modelset_id"] == modelset_id
    assert created["tags"] == ["qa", "phase2"]

    update_resp = client.patch(
        f"/api/modelsets/{modelset_id}",
        json={"name": "Route Test Updated", "description": "Updated", "tags": ["updated"]},
    )
    assert update_resp.status_code == 200
    updated = update_resp.json()["modelset"]
    assert updated["name"] == "Route Test Updated"
    assert updated["tags"] == ["updated"]

    save_resp = client.post(
        f"/api/modelsets/{modelset_id}/versions",
        json={"notes": "v1"},
    )
    assert save_resp.status_code == 200
    version_id = save_resp.json()["version"]["version_id"]

    load_resp = client.post(
        f"/api/modelsets/{modelset_id}/load",
        json={"version_id": version_id},
    )
    assert load_resp.status_code == 200
    assert load_resp.json()["loaded"] is True

    delete_active_resp = client.delete(
        f"/api/modelsets/{modelset_id}/versions/{version_id}",
    )
    assert delete_active_resp.status_code == 409

    delete_force_resp = client.delete(
        f"/api/modelsets/{modelset_id}/versions/{version_id}?force=true",
    )
    assert delete_force_resp.status_code == 200
    assert delete_force_resp.json()["forced"] is True

    delete_resp = client.delete(f"/api/modelsets/{modelset_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] is True
