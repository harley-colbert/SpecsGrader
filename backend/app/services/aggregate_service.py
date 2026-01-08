from typing import Any, Dict, Optional, Tuple

from ..decision_policy import DEFAULT_DECISION_POLICY, DEFAULT_WEIGHTS, resolve_decision_policy


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


def _build_trace_evidence(method_outputs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    rules_output = method_outputs.get("rules") or {}
    model_output = method_outputs.get("model") or {}
    vector_output = method_outputs.get("vector") or {}
    deep_output = method_outputs.get("deep") or {}
    llm_output = method_outputs.get("llm") or {}
    vector_neighbors = vector_output.get("top_neighbors") or vector_output.get("neighbors")
    if isinstance(vector_neighbors, list):
        vector_neighbors = vector_neighbors[:3]
    return {
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
            "vote_conf_level": vector_output.get("vote_conf_level"),
            "vote_conf_dept": vector_output.get("vote_conf_dept"),
        },
        "deep": {
            "available": deep_output.get("available"),
            "level_pred": deep_output.get("level_pred"),
            "dept_pred": deep_output.get("dept_pred"),
            "level_conf": deep_output.get("level_conf"),
            "dept_conf": deep_output.get("dept_conf"),
        },
        "llm": {
            "reason": llm_output.get("reason"),
        },
    }


def _trace_step(trace: Dict[str, Any], step: str, selected: bool) -> None:
    trace["steps"].append({"step": step, "selected": selected})


def evaluate_policy(policy: Dict[str, Any], method_outputs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    resolved_policy = resolve_decision_policy(policy)
    trace = {
        "mode": "production",
        "winner": None,
        "policy": resolved_policy,
        "steps": [],
        "evidence": _build_trace_evidence(method_outputs),
    }

    rules_output = method_outputs.get("rules") or {}
    model_output = method_outputs.get("model") or {}
    vector_output = method_outputs.get("vector") or {}
    deep_output = method_outputs.get("deep") or {}
    llm_output = method_outputs.get("llm") or {}

    model_level_conf = float(model_output.get("level_conf") or 0.0)
    model_dept_conf = float(model_output.get("dept_conf") or 0.0)
    vector_similarity = float(vector_output.get("top_similarity") or 0.0)
    vector_margin = float(vector_output.get("margin") or 0.0)
    vector_vote_level = vector_output.get("vote_conf_level")
    vector_vote_dept = vector_output.get("vote_conf_dept")
    vector_vote_available = vector_vote_level is not None and vector_vote_dept is not None
    if vector_vote_available:
        vector_vote_min = min(float(vector_vote_level or 0.0), float(vector_vote_dept or 0.0))
    else:
        vector_vote_min = 0.0

    for layer in resolved_policy.get("layers", []):
        layer_type = layer.get("type")
        layer_id = layer.get("id") or layer_type or "layer"
        if layer.get("enabled") is False:
            _trace_step(trace, layer_id, False)
            continue

        if layer_type == "rules_hard":
            hard_rules = bool(rules_output.get("hard"))
            if hard_rules and rules_output.get("dept_pred"):
                trace["winner"] = layer_id
                _trace_step(trace, layer_id, True)
                return {
                    "pred_level": model_output.get("level_pred"),
                    "pred_dept": rules_output.get("dept_pred"),
                    "conf_level": model_level_conf,
                    "conf_dept": float(rules_output.get("dept_conf") or 1.0),
                    "trace": trace,
                }
            _trace_step(trace, layer_id, False)
            continue

        if layer_type == "model_confidence":
            min_confidence = float(layer.get("min_confidence") or 0.0)
            if (
                model_output.get("level_pred")
                and model_output.get("dept_pred")
                and model_level_conf >= min_confidence
                and model_dept_conf >= min_confidence
            ):
                trace["winner"] = layer_id
                _trace_step(trace, layer_id, True)
                return {
                    "pred_level": model_output.get("level_pred"),
                    "pred_dept": model_output.get("dept_pred"),
                    "conf_level": model_level_conf,
                    "conf_dept": model_dept_conf,
                    "trace": trace,
                }
            _trace_step(trace, layer_id, False)
            continue

        if layer_type == "model_vector_consensus":
            min_similarity = float(layer.get("min_similarity") or 0.0)
            model_vector_agree = (
                model_output.get("dept_pred")
                and model_output.get("dept_pred") == vector_output.get("dept_pred")
                and model_output.get("level_pred")
                and model_output.get("level_pred") == vector_output.get("level_pred")
            )
            vector_signal_ok = vector_vote_min >= min_similarity if vector_vote_available else vector_similarity >= min_similarity
            if model_vector_agree and vector_signal_ok:
                trace["winner"] = layer_id
                _trace_step(trace, layer_id, True)
                vector_level_conf = float(vector_vote_level) if vector_vote_level is not None else float(vector_output.get("level_conf") or 0.0)
                vector_dept_conf = float(vector_vote_dept) if vector_vote_dept is not None else float(vector_output.get("dept_conf") or 0.0)
                return {
                    "pred_level": model_output.get("level_pred"),
                    "pred_dept": model_output.get("dept_pred"),
                    "conf_level": max(model_level_conf, vector_level_conf),
                    "conf_dept": max(model_dept_conf, vector_dept_conf),
                    "trace": trace,
                }
            _trace_step(trace, layer_id, False)
            continue

        if layer_type == "vector_confidence":
            min_similarity = float(layer.get("min_similarity") or 0.0)
            min_margin = float(layer.get("min_margin") or 0.0)
            vector_level_conf = float(vector_vote_level) if vector_vote_level is not None else float(vector_output.get("level_conf") or 0.0)
            vector_dept_conf = float(vector_vote_dept) if vector_vote_dept is not None else float(vector_output.get("dept_conf") or 0.0)
            vector_signal_ok = vector_vote_min >= min_similarity if vector_vote_available else vector_similarity >= min_similarity
            if (
                vector_output.get("dept_pred")
                and vector_output.get("level_pred")
                and vector_signal_ok
                and vector_margin >= min_margin
            ):
                trace["winner"] = layer_id
                _trace_step(trace, layer_id, True)
                return {
                    "pred_level": vector_output.get("level_pred"),
                    "pred_dept": vector_output.get("dept_pred"),
                    "conf_level": vector_level_conf,
                    "conf_dept": vector_dept_conf,
                    "trace": trace,
                }
            _trace_step(trace, layer_id, False)
            continue

        if layer_type == "deep_model":
            min_confidence = float(layer.get("min_confidence") or 0.0)
            deep_level_conf = float(deep_output.get("level_conf") or 0.0)
            deep_dept_conf = float(deep_output.get("dept_conf") or 0.0)
            if (
                deep_output.get("available")
                and deep_output.get("level_pred")
                and deep_output.get("dept_pred")
                and deep_level_conf >= min_confidence
                and deep_dept_conf >= min_confidence
            ):
                trace["winner"] = layer_id
                _trace_step(trace, layer_id, True)
                return {
                    "pred_level": deep_output.get("level_pred"),
                    "pred_dept": deep_output.get("dept_pred"),
                    "conf_level": deep_level_conf,
                    "conf_dept": deep_dept_conf,
                    "trace": trace,
                }
            _trace_step(trace, layer_id, False)
            continue

        if layer_type == "weighted":
            aggregated = _weighted_aggregate(method_outputs, resolved_policy.get("weights", DEFAULT_WEIGHTS))
            trace["winner"] = layer_id
            _trace_step(trace, layer_id, True)
            return {**aggregated, "trace": trace}

        if layer_type == "llm":
            if llm_output.get("dept_pred") and llm_output.get("level_pred"):
                trace["winner"] = layer_id
                _trace_step(trace, layer_id, True)
                return {
                    "pred_level": llm_output.get("level_pred"),
                    "pred_dept": llm_output.get("dept_pred"),
                    "conf_level": float(llm_output.get("level_conf") or 0.0),
                    "conf_dept": float(llm_output.get("dept_conf") or 0.0),
                    "trace": trace,
                }
            _trace_step(trace, layer_id, False)
            continue

        if layer_type == "abstain":
            trace["winner"] = layer_id
            _trace_step(trace, layer_id, True)
            return {
                "pred_level": None,
                "pred_dept": None,
                "conf_level": 0.0,
                "conf_dept": 0.0,
                "trace": trace,
            }

        _trace_step(trace, layer_id, False)

    aggregated = _weighted_aggregate(method_outputs, resolved_policy.get("weights", DEFAULT_WEIGHTS))
    trace["winner"] = "weighted"
    _trace_step(trace, "weighted", True)
    return {**aggregated, "trace": trace}


def aggregate_outputs(
    method_outputs: Dict[str, Dict[str, object]],
    weights: Dict[str, float] | None = None,
    mode: str = "production",
    policy: Dict[str, object] | None = None,
) -> Dict[str, object]:
    resolved_policy = resolve_decision_policy(policy)
    if weights:
        resolved_policy["weights"] = {**resolved_policy.get("weights", DEFAULT_WEIGHTS), **weights}

    trace = {
        "mode": mode,
        "winner": None,
        "policy": resolved_policy,
        "steps": [],
        "evidence": _build_trace_evidence(method_outputs),
    }

    if mode in {"sanity", "evaluate"}:
        aggregated = _weighted_aggregate(method_outputs, resolved_policy.get("weights", DEFAULT_WEIGHTS))
        trace["winner"] = "weighted"
        trace["steps"].append({"step": "weighted", "selected": True})
        return {**aggregated, "trace": trace}

    result = evaluate_policy(resolved_policy, method_outputs)
    result["trace"]["mode"] = mode
    return result


__all__ = ["aggregate_outputs", "evaluate_policy", "DEFAULT_WEIGHTS", "DEFAULT_DECISION_POLICY"]
