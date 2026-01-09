from backend.app.config.xlsx_contract import RISK_ORDER, is_medium_plus


def test_risk_ordering_sequence():
    assert RISK_ORDER == ["none", "low", "medium", "high", "extreme"]


def test_is_medium_plus():
    assert not is_medium_plus("none")
    assert not is_medium_plus("low")
    assert is_medium_plus("medium")
    assert is_medium_plus("high")
    assert is_medium_plus("extreme")
