import json
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

DEFAULT_CONFIG = {
    "version": "1.0",
    "departments": {
        "mechanical": {"keywords": ["bearing"], "min_hits": 1},
        "electrical": {"keywords": ["panel"], "min_hits": 1},
        "controls": {"keywords": ["plc"], "min_hits": 1},
        "project_management": {"keywords": ["schedule"], "min_hits": 1},
    },
    "global": {
        "case_sensitive": False,
        "match_mode": "token_contains",
        "abstain_on_tie": True,
    },
}


@dataclass
class RulePrediction:
    dept_pred: Optional[str]
    dept_conf: float
    matched: Dict[str, List[str]]


class RuleService:
    def __init__(self, config: Optional[Dict[str, object]] = None):
        self.config = config or DEFAULT_CONFIG

    @staticmethod
    def load_rules(path_or_bundle: Optional[str]) -> Dict[str, object]:
        if path_or_bundle is None:
            return DEFAULT_CONFIG
        path = pathlib.Path(path_or_bundle)
        if not path.exists():
            raise FileNotFoundError(f"Rules file not found: {path}")
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if "version" not in data:
            raise ValueError("rules_config missing version")
        return data

    def predict(self, risk_text: str) -> RulePrediction:
        cfg = self.config
        text = risk_text if cfg["global"].get("case_sensitive") else risk_text.lower()
        matched: Dict[str, List[str]] = {dept: [] for dept in cfg["departments"]}

        tokens = text.split()
        for dept, rule in cfg["departments"].items():
            keywords: Iterable[str] = rule.get("keywords", [])
            for kw in keywords:
                kw_norm = kw if cfg["global"].get("case_sensitive") else kw.lower()
                if any(kw_norm in token for token in tokens):
                    matched[dept].append(kw)

        hits = {dept: len(words) for dept, words in matched.items()}
        best_dept = max(hits, key=hits.get)
        best_hits = hits[best_dept]
        sorted_hits = sorted(hits.values(), reverse=True)
        second_best = sorted_hits[1] if len(sorted_hits) > 1 else 0

        min_hits = cfg["departments"][best_dept].get("min_hits", 1)
        abstain = False
        if best_hits < min_hits:
            abstain = True
        if cfg["global"].get("abstain_on_tie", True) and best_hits == second_best:
            abstain = True

        if abstain or best_hits == 0:
            return RulePrediction(dept_pred=None, dept_conf=0.0, matched=matched)

        confidence = best_hits / (best_hits + second_best + 1)
        return RulePrediction(dept_pred=best_dept, dept_conf=confidence, matched=matched)


__all__ = ["RuleService", "RulePrediction", "DEFAULT_CONFIG"]
