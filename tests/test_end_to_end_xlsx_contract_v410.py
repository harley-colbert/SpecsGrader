from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from backend.app.services.xlsx_output_service import write_classify_predictions_to_xlsx


def test_end_to_end_xlsx_contract_v410(tmp_path: Path):
    target = tmp_path / "contract_v410_classify_in.xlsx"
    rows = [["", "", "", "", "", "", ""] for _ in range(4)]
    rows += [
        ["", "", "", "Spec A", "Risk A", "", ""],
        ["", "", "", "Spec B", "Risk B", "", ""],
    ]
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Standards Risk Matrix Sample", index=False, header=False)

    results = [
        {"source_row": 5, "pred_level": "medium", "pred_dept": "mechanical"},
        {"source_row": 6, "pred_level": "low", "pred_dept": "controls"},
    ]

    output_path = write_classify_predictions_to_xlsx(str(target), results, overwrite_predictions=True)

    wb = load_workbook(output_path)
    sheet = wb.active
    assert sheet["D5"].value == "Spec A"
    assert sheet["F5"].value == "medium"
    assert sheet["G5"].value == "mechanical"
    assert sheet["F6"].value == "low"
    assert sheet["G6"].value == "controls"
