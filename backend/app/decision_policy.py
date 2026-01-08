import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_WEIGHTS: Dict[str, float] = {"model": 1.2, "vector": 1.0, "llm": 0.6, "rules": 0.2}

DEFAULT_DECISION_POLICY: Dict[str, Any] = {
    "version": "1.0",
    "weights": DEFAULT_WEIGHTS,
    "layers": [
        {"id": "hard_rules", "type": "rules_hard"},
        {"id": "model", "type": "model_confidence", "min_confidence": 0.75},
        {"id": "consensus", "type": "model_vector_consensus", "min_similarity": 0.45},
        {"id": "vector", "type": "vector_confidence", "min_similarity": 0.45, "min_margin": 0.1},
        {"id": "llm", "type": "llm", "enabled": False},
        {"id": "abstain", "type": "abstain", "enabled": True},
        {"id": "weighted", "type": "weighted"},
    ],
}


def default_decision_policy() -> Dict[str, Any]:
    return deepcopy(DEFAULT_DECISION_POLICY)


def resolve_decision_policy(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    resolved = default_decision_policy()
    if not isinstance(policy, dict):
        return resolved
    version = policy.get("version")
    if version:
        resolved["version"] = version
    weights = policy.get("weights")
    if isinstance(weights, dict):
        resolved["weights"] = {**resolved["weights"], **weights}
    layers = policy.get("layers")
    if isinstance(layers, list) and layers:
        resolved["layers"] = layers
    return resolved


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")


def load_decision_policy(path: Path) -> Dict[str, Any]:
    if path.exists():
        return _read_json(path)
    return default_decision_policy()


def save_decision_policy(path: Path, policy: Dict[str, Any]) -> None:
    _write_json(path, policy)


__all__ = [
    "DEFAULT_DECISION_POLICY",
    "DEFAULT_WEIGHTS",
    "default_decision_policy",
    "resolve_decision_policy",
    "load_decision_policy",
    "save_decision_policy",
]
