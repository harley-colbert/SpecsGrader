from __future__ import annotations

import re
from typing import Iterable

from backend.app.config.xlsx_contract import is_medium_plus, normalize_risk_level


_DEPT_PHRASES = {
    "mechanical": "mechanical system",
    "electrical": "electrical system",
    "controls": "controls system",
    "project_management": "project planning",
}

_HOOKS = [
    "bearing",
    "overheat",
    "leak",
    "pressure",
    "voltage",
    "corrosion",
    "fire",
    "shock",
    "fall",
    "spill",
    "short",
    "overload",
]


def _extract_hooks(text: str, hooks: Iterable[str] = _HOOKS) -> list[str]:
    lowered = text.lower()
    found = [hook for hook in hooks if hook in lowered]
    if found:
        return found[:3]
    units = re.findall(r"\b\d+(?:\.\d+)?\s?(?:psi|bar|v|kv|ma|amp|amps|°c|degc|degf)\b", lowered)
    return units[:2]


def generate_specific_risk(spec_text: str, risk_level: str, dept: str) -> str:
    normalized_level = normalize_risk_level(risk_level) or ""
    if not is_medium_plus(normalized_level):
        return ""

    spec_text = (spec_text or "").strip()
    dept_key = (dept or "").strip().lower()
    dept_phrase = _DEPT_PHRASES.get(dept_key, "operational")
    hooks = _extract_hooks(spec_text)
    hook_text = ", ".join(hooks) if hooks else "operational issues"
    summary_words = spec_text.split()
    summary = " ".join(summary_words[:10]) if summary_words else "specified conditions"

    return f"{normalized_level.capitalize()} {dept_phrase} risk: {hook_text} in {summary}."
