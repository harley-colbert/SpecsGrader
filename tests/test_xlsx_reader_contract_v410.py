import pandas as pd

from backend.app.services.ingest_service import load_classify_dataset, load_training_dataset


def test_contract_v410_reads_columns_and_skips_blank_rows(tmp_path):
    path = tmp_path / "contract_v410_input.xlsx"
    rows = [["", "", "", "", "", "", ""] for _ in range(4)]
    rows += [
        ["", "", "", "Spec A", "Risk A", " Medium ", "Electrical"],
        ["", "", "", "", "Risk B", "low", "controls"],
        ["", "", "", "Spec C", "", "low", "Controls"],
        ["", "", "", "Spec D", "Risk D", "", ""],
        ["", "", "", "Spec E", "Risk E", "HIGH", "Project_Management"],
        ["", "", "", "Spec F", "", "", ""],
    ]
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Standards Risk Matrix V4.10", index=False, header=False)

    dataset = load_training_dataset(str(path))

    assert dataset["summary"]["total_rows"] == 5
    first = dataset["rows"][0]
    assert first["spec_text"] == "Spec A"
    assert first["specific_risk_existing"] == "Risk A"
    assert first["risk_level_existing"] == "medium"
    assert first["dept_existing"] == "electrical"
    assert first["risk_text"] == "Risk A"
    assert first["label_level"] == "medium"
    assert first["label_dept"] == "electrical"

    third = dataset["rows"][2]
    assert third["spec_text"] == "Spec D"
    assert third["specific_risk_existing"] == "Risk D"
    assert third["risk_level_existing"] == ""
    assert third["dept_existing"] == ""

    fourth = dataset["rows"][3]
    assert fourth["risk_level_existing"] == "high"
    assert fourth["dept_existing"] == "project_management"


def test_reader_handles_missing_optional_columns(tmp_path):
    path = tmp_path / "only_d.xlsx"
    rows = [["", "", "", "", "", "", ""] for _ in range(4)]
    rows.append(["", "", "", "Spec Only"])
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Standards Risk Matrix Missing", index=False, header=False)

    dataset = load_classify_dataset(str(path))

    assert dataset["summary"]["total_rows"] == 1
    row = dataset["rows"][0]
    assert row["spec_text"] == "Spec Only"
    assert row["specific_risk_existing"] == ""
    assert row["risk_level_existing"] == ""
    assert row["dept_existing"] == ""
