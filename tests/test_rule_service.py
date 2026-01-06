import pytest

from backend.app.services.rule_service import RuleService


def test_abstains_when_no_hits():
    svc = RuleService()
    pred = svc.predict("unrelated text")
    assert pred.dept_pred is None
    assert pred.dept_conf == 0.0


def test_predicts_when_one_department_wins():
    cfg = {
        "version": "1.0",
        "departments": {
            "mechanical": {"keywords": ["bearing"], "min_hits": 1},
            "electrical": {"keywords": ["panel"], "min_hits": 1},
        },
        "global": {"case_sensitive": False, "match_mode": "token_contains", "abstain_on_tie": True},
    }
    svc = RuleService(cfg)
    pred = svc.predict("bearing vibration")
    assert pred.dept_pred == "mechanical"
    assert pred.dept_conf > 0


def test_abstains_on_tie():
    cfg = {
        "version": "1.0",
        "departments": {
            "mechanical": {"keywords": ["bearing"], "min_hits": 1},
            "electrical": {"keywords": ["panel"], "min_hits": 1},
        },
        "global": {"case_sensitive": False, "match_mode": "token_contains", "abstain_on_tie": True},
    }
    svc = RuleService(cfg)
    pred = svc.predict("bearing panel")
    assert pred.dept_pred is None


def test_version_required_on_load(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        RuleService.load_rules(str(path))
