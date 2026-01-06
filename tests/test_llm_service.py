import json
import pytest

from backend.app.services.llm_service import LLMService


def test_missing_api_key_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    svc = LLMService()
    with pytest.raises(RuntimeError):
        svc.predict("text", "model", never_send=False)


def test_never_send_blocks(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    svc = LLMService()
    with pytest.raises(PermissionError):
        svc.predict("text", "model", never_send=True)


def test_invalid_labels_rejected(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({"risk_level": "bad", "department": "x"})}}]}

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    svc = LLMService(api_key="dummy")
    with pytest.raises(ValueError):
        svc.predict("text", "model", never_send=False)
