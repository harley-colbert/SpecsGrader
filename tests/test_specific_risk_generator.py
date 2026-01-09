from backend.app.services.specific_risk_service import generate_specific_risk


def test_specific_risk_blank_for_low_and_none():
    assert generate_specific_risk("Spec text", "none", "mechanical") == ""
    assert generate_specific_risk("Spec text", "low", "mechanical") == ""


def test_specific_risk_generated_for_medium_plus():
    assert generate_specific_risk("Spec text", "medium", "mechanical")
    assert generate_specific_risk("Spec text", "high", "mechanical")
    assert generate_specific_risk("Spec text", "extreme", "mechanical")


def test_specific_risk_varies_by_department():
    mech = generate_specific_risk("Spec text", "high", "mechanical")
    elec = generate_specific_risk("Spec text", "high", "electrical")
    assert mech != elec


def test_specific_risk_is_deterministic():
    first = generate_specific_risk("Spec text", "high", "controls")
    second = generate_specific_risk("Spec text", "high", "controls")
    assert first == second
