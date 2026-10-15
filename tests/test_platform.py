from run_evaluation import TIERS


def test_smoke_tier_has_five_cases():
    assert "smoke" in TIERS["smoke"]
    assert "critical" not in TIERS["smoke"]
    assert "deep" not in TIERS["smoke"]


def test_critical_tier_includes_smoke():
    assert TIERS["smoke"].issubset(TIERS["critical"])


def test_deep_tier_includes_all_tiers():
    assert TIERS["critical"].issubset(TIERS["deep"])
    assert "deep" in TIERS["deep"]


def test_tiers_are_ordered_by_coverage():
    assert len(TIERS["smoke"]) < len(TIERS["critical"])
    assert len(TIERS["critical"]) < len(TIERS["deep"])
