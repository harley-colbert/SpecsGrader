from typing import Dict, Optional, Tuple

DEFAULT_WEIGHTS = {"model": 1.2, "vector": 1.0, "llm": 0.6, "rules": 0.2}

DEFAULT_THRESHOLDS = {
    "model_conf_threshold": 0.75,
    "vector_similarity_threshold": 0.45,
    "vector_margin_threshold": 0.1,
    "allow_llm": False,
    "abstain_enabled": True,
}


def _best_label(scores: Dict[str, float]) -> Tuple[Optional[str], float]:
    if not scores:
        return None, 0.0
    label, score = max(scores.items(), key=lambda item: item[1])
    total = sum(scores.values()) or 1.0
    return label, score / total


def _weighted_aggregate(method_outputs: Dict[str, Dict[str, object]], weights: Dict[str, float]) -> Dict[str, object]:
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


def aggregate_outputs(
    method_outputs: Dict[str, Dict[str, object]],
    weights: Dict[str, float] | None = None,
    mode: str = "production",
    thresholds: Dict[str, object] | None = None,
) -> Dict[str, object]:
    weights = weights or DEFAULT_WEIGHTS
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    trace = {
        "mode": mode,
        "winner": None,
        "thresholds": thresholds,
        "steps": [],
        "evidence": {},
    }

    if mode in {"sanity", "evaluate"}:
        aggregated = _weighted_aggregate(method_outputs, weights)
        trace["winner"] = "weighted"
        trace["steps"].append({"step": "weighted", "selected": True})
        return {**aggregated, "trace": trace}

    rules_output = method_outputs.get("rules") or {}
    model_output = method_outputs.get("model") or {}
    vector_output = method_outputs.get("vector") or {}
    llm_output = method_outputs.get("llm") or {}
    vector_neighbors = vector_output.get("top_neighbors") or vector_output.get("neighbors")
    if isinstance(vector_neighbors, list):
        vector_neighbors = vector_neighbors[:3]
    trace["evidence"] = {
        "rules": {
            "matched": rules_output.get("matched"),
            "hard_hits": rules_output.get("hard_hits"),
            "is_hard": rules_output.get("hard"),
        },
        "model": {
            "level_proba": model_output.get("level_proba"),
            "dept_proba": model_output.get("dept_proba"),
            "level_top_terms": model_output.get("level_top_terms"),
            "dept_top_terms": model_output.get("dept_top_terms"),
        },
        "vector": {
            "neighbors": vector_neighbors,
            "top_similarity": vector_output.get("top_similarity"),
            "margin": vector_output.get("margin"),
        },
        "llm": {
            "reason": llm_output.get("reason"),
        },
    }

    hard_rules = bool(rules_output.get("hard"))
    if hard_rules and rules_output.get("dept_pred"):
        trace["winner"] = "hard_rules"
        trace["steps"].append({"step": "hard_rules", "selected": True})
        return {
            "pred_level": model_output.get("level_pred"),
            "pred_dept": rules_output.get("dept_pred"),
            "conf_level": float(model_output.get("level_conf") or 0.0),
            "conf_dept": float(rules_output.get("dept_conf") or 1.0),
            "trace": trace,
        }
    trace["steps"].append({"step": "hard_rules", "selected": False})

    model_conf_threshold = float(thresholds.get("model_conf_threshold") or 0.0)
    model_level_conf = float(model_output.get("level_conf") or 0.0)
    model_dept_conf = float(model_output.get("dept_conf") or 0.0)
    if (
        model_output.get("level_pred")
        and model_output.get("dept_pred")
        and model_level_conf >= model_conf_threshold
        and model_dept_conf >= model_conf_threshold
    ):
        trace["winner"] = "model"
        trace["steps"].append({"step": "model", "selected": True})
        return {
            "pred_level": model_output.get("level_pred"),
            "pred_dept": model_output.get("dept_pred"),
            "conf_level": model_level_conf,
            "conf_dept": model_dept_conf,
            "trace": trace,
        }
    trace["steps"].append({"step": "model", "selected": False})

    vector_similarity = float(vector_output.get("top_similarity") or 0.0)
    vector_margin = float(vector_output.get("margin") or 0.0)
    vector_similarity_threshold = float(thresholds.get("vector_similarity_threshold") or 0.0)
    vector_margin_threshold = float(thresholds.get("vector_margin_threshold") or 0.0)
    model_vector_agree = (
        model_output.get("dept_pred")
        and model_output.get("dept_pred") == vector_output.get("dept_pred")
        and model_output.get("level_pred")
        and model_output.get("level_pred") == vector_output.get("level_pred")
    )
    if model_vector_agree and vector_similarity >= vector_similarity_threshold:
        trace["winner"] = "consensus"
        trace["steps"].append({"step": "consensus", "selected": True})
        return {
            "pred_level": model_output.get("level_pred"),
            "pred_dept": model_output.get("dept_pred"),
            "conf_level": max(model_level_conf, float(vector_output.get("level_conf") or 0.0)),
            "conf_dept": max(model_dept_conf, float(vector_output.get("dept_conf") or 0.0)),
            "trace": trace,
        }
    trace["steps"].append({"step": "consensus", "selected": False})

    if (
        vector_output.get("dept_pred")
        and vector_output.get("level_pred")
        and vector_similarity >= vector_similarity_threshold
        and vector_margin >= vector_margin_threshold
    ):
        trace["winner"] = "vector"
        trace["steps"].append({"step": "vector", "selected": True})
        return {
            "pred_level": vector_output.get("level_pred"),
            "pred_dept": vector_output.get("dept_pred"),
            "conf_level": float(vector_output.get("level_conf") or 0.0),
            "conf_dept": float(vector_output.get("dept_conf") or 0.0),
            "trace": trace,
        }
    trace["steps"].append({"step": "vector", "selected": False})

    allow_llm = bool(thresholds.get("allow_llm"))
    if allow_llm and llm_output.get("dept_pred") and llm_output.get("level_pred"):
        trace["winner"] = "llm"
        trace["steps"].append({"step": "llm", "selected": True})
        return {
            "pred_level": llm_output.get("level_pred"),
            "pred_dept": llm_output.get("dept_pred"),
            "conf_level": float(llm_output.get("level_conf") or 0.0),
            "conf_dept": float(llm_output.get("dept_conf") or 0.0),
            "trace": trace,
        }
    trace["steps"].append({"step": "llm", "selected": False})

    if thresholds.get("abstain_enabled"):
        trace["winner"] = "abstain"
        trace["steps"].append({"step": "abstain", "selected": True})
        return {
            "pred_level": None,
            "pred_dept": None,
            "conf_level": 0.0,
            "conf_dept": 0.0,
            "trace": trace,
        }

    aggregated = _weighted_aggregate(method_outputs, weights)
    trace["winner"] = "weighted"
    trace["steps"].append({"step": "weighted", "selected": True})
    return {**aggregated, "trace": trace}


__all__ = ["aggregate_outputs", "DEFAULT_WEIGHTS", "DEFAULT_THRESHOLDS"]
