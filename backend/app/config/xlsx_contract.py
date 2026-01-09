from __future__ import annotations

from typing import Optional

SPEC_TEXT_COL = "D"
SPECIFIC_RISK_COL = "E"
RISK_LEVEL_COL = "F"
DEPT_COL = "G"

RISK_ORDER = ["none", "low", "medium", "high", "extreme"]
RISK_LEVELS = set(RISK_ORDER)

_SYNONYM_MAP = {
    "med": "medium",
    "mid": "medium",
    "hi": "high",
}


def normalize_risk_level(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    return _SYNONYM_MAP.get(normalized, normalized)


def is_medium_plus(level: Optional[str]) -> bool:
    return normalize_risk_level(level) in {"medium", "high", "extreme"}


def column_index(column_letter: str) -> int:
    """Convert an Excel column letter to a zero-based index."""
    normalized = column_letter.strip().upper()
    if not normalized.isalpha():
        raise ValueError(f"Invalid column letter: {column_letter!r}")
    index = 0
    for char in normalized:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1
