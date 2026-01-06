from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_health_endpoint_returns_ok():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
