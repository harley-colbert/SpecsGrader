import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import requests

RISK_LEVELS = {"none", "low", "medium", "high", "extreme"}
DEPARTMENTS = {"mechanical", "electrical", "controls", "project_management"}


@dataclass
class LlmPrediction:
    risk_level: Optional[str]
    department: Optional[str]
    confidence: Optional[float]
    reason: Optional[str]


class LLMService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

    @staticmethod
    def _validate_payload(payload: Dict[str, object]) -> LlmPrediction:
        level = (payload.get("risk_level") or "").lower()
        dept = (payload.get("department") or "").lower()
        confidence = payload.get("confidence")
        reason = payload.get("reason")

        if level and level not in RISK_LEVELS:
            raise ValueError("Invalid risk_level")
        if dept and dept not in DEPARTMENTS:
            raise ValueError("Invalid department")

        return LlmPrediction(
            risk_level=level or None,
            department=dept or None,
            confidence=float(confidence) if confidence is not None else None,
            reason=str(reason) if reason else None,
        )

    def predict(self, risk_text: str, model_name: str, never_send: bool = False) -> LlmPrediction:
        if never_send:
            raise PermissionError("LLM disabled by never-send mode")
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a classifier that returns JSON."},
                {
                    "role": "user",
                    "content": f"Classify this risk: {risk_text}. Respond with JSON containing risk_level and department.",
                },
            ],
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=10,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"LLM request failed: {response.text}")
        try:
            data = response.json()
            message = data["choices"][0]["message"]["content"]
            parsed = json.loads(message)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Invalid LLM response format") from exc

        return self._validate_payload(parsed)


__all__ = ["LLMService", "LlmPrediction"]
