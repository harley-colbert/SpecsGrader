from backend.app.config.xlsx_contract import RISK_LEVELS
from backend.app.services.ingest_service import DEPARTMENTS
from backend.app.services.training_service import TrainingService


def test_training_extraction_uses_spec_text_and_valid_labels():
    rows = [
        {
            "spec_text": "Spec A",
            "risk_text": "Risk A",
            "label_level": "medium",
            "label_dept": "mechanical",
        },
        {
            "spec_text": "Spec B",
            "risk_text": "Risk B",
            "label_level": "invalid",
            "label_dept": "controls",
        },
        {
            "spec_text": "",
            "risk_text": "Risk C",
            "label_level": "high",
            "label_dept": "electrical",
        },
        {
            "spec_text": "Spec D",
            "risk_text": "Risk D",
            "label_level": "low",
            "label_dept": "bad_dept",
        },
    ]

    level_rows = TrainingService._valid_rows(rows, "label_level", RISK_LEVELS)
    dept_rows = TrainingService._valid_rows(rows, "label_dept", DEPARTMENTS)

    assert [TrainingService._training_text(row) for row in level_rows] == ["Spec A", "Spec D"]
    assert [row["label_level"] for row in level_rows] == ["medium", "low"]
    assert [TrainingService._training_text(row) for row in dept_rows] == ["Spec A", "Spec B"]
    assert [row["label_dept"] for row in dept_rows] == ["mechanical", "controls"]
