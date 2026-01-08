from pathlib import Path

from backend.app.services.training_service import TrainingParams, TrainingService


def _make_rows(majority: int, minority: int) -> list[dict[str, str]]:
    rows = []
    for idx in range(majority):
        rows.append(
            {
                "risk_text": f"Mechanical issue {idx}",
                "label_level": "low",
                "label_dept": "mechanical",
            }
        )
    for idx in range(minority):
        rows.append(
            {
                "risk_text": f"Electrical issue {idx}",
                "label_level": "high",
                "label_dept": "electrical",
            }
        )
    return rows


def test_oversample_respects_cap_ratio(tmp_path: Path) -> None:
    service = TrainingService(workspace=tmp_path / "workspace", app_state=type("obj", (), {})())
    rows = _make_rows(majority=10, minority=2)
    oversampled = service.oversample_rows(rows, "label_dept", cap_ratio=0.5)
    post_counts = {}
    for row in oversampled:
        label = row["label_dept"]
        post_counts[label] = post_counts.get(label, 0) + 1
    assert post_counts["electrical"] == 5
    assert post_counts["mechanical"] == 10


def test_pipeline_class_weight_toggle(tmp_path: Path) -> None:
    service = TrainingService(workspace=tmp_path / "workspace", app_state=type("obj", (), {})())
    pipeline_balanced = service._build_pipeline("sigmoid", True)
    pipeline_unbalanced = service._build_pipeline("sigmoid", False)
    assert pipeline_balanced.named_steps["clf"].estimator.class_weight == "balanced"
    assert pipeline_unbalanced.named_steps["clf"].estimator.class_weight is None
