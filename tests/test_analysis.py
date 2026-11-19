import numpy as np
import pandas as pd
import pytest

from affordability.analysis import (
    benjamini_hochberg,
    better_alternative,
    borderline_declines,
    check_depth,
    fit_arrears_model,
    group_outcomes,
    headroom_curve,
    jeffreys,
    policy_frontier,
)
from affordability.simulate import current_policy, simulate_applications


@pytest.fixture(scope="module")
def apps():
    return simulate_applications(20000, seed=3)


def test_outcomes_only_observed_when_lent(apps):
    lent = apps["decision"].str.startswith("approve")
    assert apps.loc[lent, "arrears_3m"].notna().all()
    assert apps.loc[~lent, "arrears_3m"].isna().all()
    assert (apps["decision"] == "approve").sum() == current_policy(apps).sum()


def test_bh_is_monotone_and_bounded():
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.all((q >= 0) & (q <= 1))
    assert q[0] <= q[2] <= q[1] <= q[3] or np.isclose(q[2], q[1])


def test_jeffreys_zero_defaults():
    lo, hi = jeffreys(0, 40)
    assert lo == 0 and hi < 0.1


def test_check_depth_increases_with_amount(apps):
    d = check_depth(apps)
    assert d["open_banking_share"].is_monotonic_increasing


def test_headroom_curve_is_monotone_decreasing(apps):
    c = headroom_curve(apps)
    assert c["fitted"].is_monotonic_decreasing
    assert c["arrears_rate"].iloc[0] > c["arrears_rate"].iloc[-1]


def test_group_flags_require_materiality(apps):
    g = group_outcomes(apps, ["age_band", "vulnerable"], min_gap=0.5)
    assert not g["flag"].any()  # nothing has a 50-point gap


def test_vulnerable_customers_flagged_for_arrears(apps):
    g = group_outcomes(apps, ["vulnerable"])
    row = g[(g["group"] == "Vulnerable customers") & (g["metric"] == "arrears_rate")].iloc[0]
    assert row["gap"] > 0


def test_frontier_and_alternative(apps):
    m = fit_arrears_model(apps, 1 / 0.03)
    f = policy_frontier(apps, m, headroom_grid=np.arange(0, 301, 50), score_grid=np.arange(520, 621, 20))
    eff = f[f["efficient"]].sort_values("approval_rate")
    assert eff["expected_arrears"].is_monotonic_increasing
    cur = f[(f["min_headroom"] == 150) & (f["min_score"] == 560)].iloc[0].to_dict()
    alt = better_alternative(f, cur)
    if alt is not None:
        assert alt["expected_arrears"] < cur["expected_arrears"]
        assert alt["approval_rate"] >= cur["approval_rate"] - 0.005


def test_borderline_declines_are_declines_near_the_line(apps):
    b = borderline_declines(apps, 150, 560)
    assert (b["decision"] == "decline").all()
    near_h = (b["headroom"] >= 100) & (b["bureau_score"] >= 560)
    near_s = (b["bureau_score"] >= 540) & (b["headroom"] >= 150)
    assert (near_h | near_s).all()
