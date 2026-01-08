from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_label_policy_endpoint_returns_policy() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/label-policy")
    assert resp.status_code == 200
    payload = resp.json()
    policy = payload.get("policy")
    assert isinstance(policy, dict)
    assert "risk_levels" in policy
    assert "departments" in policy
