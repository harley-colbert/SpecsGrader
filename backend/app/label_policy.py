import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_LABEL_POLICY: Dict[str, Any] = {
    "version": "1.0",
    "risk_levels": [
        {
            "id": "none",
            "label": "NONE",
            "description": "No risk identified or not applicable.",
        },
        {
            "id": "low",
            "label": "LOW",
            "description": "Low impact risk with straightforward mitigation.",
        },
        {
            "id": "medium",
            "label": "MEDIUM",
            "description": "Moderate impact risk requiring planning and tracking.",
        },
        {
            "id": "high",
            "label": "HIGH",
            "description": "High impact risk that needs active mitigation and escalation.",
        },
        {
            "id": "extreme",
            "label": "EXTREME",
            "description": "Critical risk likely to affect schedule, scope, or safety.",
        },
    ],
    "departments": [
        {
            "id": "mechanical",
            "label": "MECHANICAL",
            "description": "Mechanical design, tooling, fabrication, or structural risks.",
        },
        {
            "id": "electrical",
            "label": "ELECTRICAL",
            "description": "Electrical design, power distribution, or wiring risks.",
        },
        {
            "id": "controls",
            "label": "CONTROLS",
            "description": "Controls, PLC, software, or automation logic risks.",
        },
        {
            "id": "project_management",
            "label": "PROJECT_MANAGEMENT",
            "description": "Schedule, budget, scope, or coordination risks.",
        },
    ],
}


def default_label_policy() -> Dict[str, Any]:
    return deepcopy(DEFAULT_LABEL_POLICY)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")


def get_modelset_policy_path(workspace: Path, modelset_id: str) -> Path:
    return workspace / "modelsets" / modelset_id / "label_policy.json"


def load_label_policy(workspace: Path, modelset_id: Optional[str] = None) -> Dict[str, Any]:
    if modelset_id:
        policy_path = get_modelset_policy_path(workspace, modelset_id)
        if policy_path.exists():
            return _read_json(policy_path)
    return default_label_policy()


def save_label_policy(workspace: Path, modelset_id: str, policy: Dict[str, Any]) -> None:
    policy_path = get_modelset_policy_path(workspace, modelset_id)
    _write_json(policy_path, policy)


__all__ = [
    "DEFAULT_LABEL_POLICY",
    "default_label_policy",
    "get_modelset_policy_path",
    "load_label_policy",
    "save_label_policy",
]
