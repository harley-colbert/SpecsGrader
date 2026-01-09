from backend.app.config.xlsx_contract import normalize_risk_level


def test_normalize_risk_level_trims_and_lowercases():
    assert normalize_risk_level(" Medium ") == "medium"
    assert normalize_risk_level("HIGH") == "high"


def test_normalize_risk_level_synonyms():
    assert normalize_risk_level("med") == "medium"
    assert normalize_risk_level("hi") == "high"


def test_normalize_risk_level_handles_empty():
    assert normalize_risk_level("") is None
    assert normalize_risk_level("   ") is None
    assert normalize_risk_level(None) is None
