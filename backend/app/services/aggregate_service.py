from typing import Dict, Optional, Tuple

DEFAULT_WEIGHTS = {"vector": 1.0, "llm": 0.6, "rules": 0.2}


def _best_label(scores: Dict[str, float]) -> Tuple[Optional[str], float]:
    if not scores:
        return None, 0.0
    label, score = max(scores.items(), key=lambda item: item[1])
    total = sum(scores.values()) or 1.0
    return label, score / total


def aggregate_outputs(method_outputs: Dict[str, Dict[str, object]], weights: Dict[str, float] | None = None) -> Dict[str, object]:
    weights = weights or DEFAULT_WEIGHTS
    level_scores: Dict[str, float] = {}
    dept_scores: Dict[str, float] = {}

    for method, output in method_outputs.items():
        if not output:
            continue
        weight = weights.get(method, 0.0)
        level = output.get("level_pred")
        dept = output.get("dept_pred")
        level_conf = float(output.get("level_conf") or 0.0)
        dept_conf = float(output.get("dept_conf") or 0.0)
        if level:
            level_scores[level] = level_scores.get(level, 0.0) + weight * level_conf
        if dept:
            dept_scores[dept] = dept_scores.get(dept, 0.0) + weight * dept_conf

    final_level, level_conf = _best_label(level_scores)
    final_dept, dept_conf = _best_label(dept_scores)

    return {
        "pred_level": final_level,
        "pred_dept": final_dept,
        "conf_level": level_conf,
        "conf_dept": dept_conf,
    }


__all__ = ["aggregate_outputs", "DEFAULT_WEIGHTS"]
