from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_health_endpoint_returns_ok():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_state_endpoint_returns_structured_state():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/state")
    assert resp.status_code == 200
    payload = resp.json()
    assert "capabilities" in payload
    assert "default_mode" in payload


def test_modelsets_endpoint_lists_modelsets():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/modelsets")
    assert resp.status_code == 200
