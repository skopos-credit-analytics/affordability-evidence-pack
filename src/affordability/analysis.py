"""The four questions an affordability evidence pack should answer.

1. Are the checks proportionate?           -> check_depth()
2. Is the threshold where risk rises?      -> headroom_curve()
3. Do outcomes differ across groups?       -> group_outcomes()
4. What would a policy change cost?        -> policy_frontier()
Plus a sample of borderline declines for manual review -> borderline_declines()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer


def jeffreys(d: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    a = (1 - level) / 2
    lo = 0.0 if d == 0 else stats.beta.ppf(a, d + 0.5, n - d + 0.5)
    hi = 1.0 if d == n else stats.beta.ppf(1 - a, d + 0.5, n - d + 0.5)
    return float(lo), float(hi)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def _observed(df: pd.DataFrame) -> pd.DataFrame:
    """Applications approved by policy (test cell excluded: it is not the live customer outcome)."""
    return df[df["decision"] == "approve"]


def check_depth(df: pd.DataFrame, bands=(0, 250, 500, 1000, np.inf)) -> pd.DataFrame:
    labels = [f"£{int(a)}–£{int(b)}" if np.isfinite(b) else f"over £{int(a)}" for a, b in zip(bands[:-1], bands[1:])]
    d = df.assign(amount_band=pd.cut(df["amount"], bands, labels=labels, right=True, include_lowest=True))
    obs = _observed(d)
    rows = []
    for band, g in d.groupby("amount_band", observed=True):
        o = obs[obs["amount_band"] == band]
        rows.append(
            {
                "amount_band": str(band),
                "applications": len(g),
                "open_banking_share": float((g["verification"] == "open banking").mean()),
                "approval_rate": float((g["decision"] == "approve").mean()),
                "arrears_rate": float(o["arrears_3m"].mean()) if len(o) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def headroom_curve(df: pd.DataFrame, step: float = 50.0, lo: float = -200.0, hi: float = 1000.0) -> pd.DataFrame:
    """Arrears rate by monthly headroom after repayment, using approved loans plus the test cell."""
    seen = df[df["arrears_3m"].notna()]
    edges = np.arange(lo, hi + step, step)
    b = pd.cut(seen["headroom"].clip(lo, hi - 1e-9), edges, right=False)
    rows = []
    for interval, g in seen.groupby(b, observed=True):
        n, d = len(g), int(g["arrears_3m"].sum())
        if n < 30:
            continue
        ci = jeffreys(d, n)
        rows.append({"headroom_from": interval.left, "headroom_to": interval.right, "loans": n, "arrears_rate": d / n, "ci_low": ci[0], "ci_high": ci[1]})
    t = pd.DataFrame(rows)
    iso = IsotonicRegression(increasing=False)
    t["fitted"] = iso.fit_transform(t["headroom_from"], t["arrears_rate"], sample_weight=t["loans"])
    return t


def threshold_evidence(curve: pd.DataFrame, threshold: float) -> dict:
    """Arrears just below versus just above the threshold, and the headroom at which risk levels off."""
    below = curve[(curve["headroom_to"] <= threshold) & (curve["headroom_from"] >= threshold - 150)]
    above = curve[(curve["headroom_from"] >= threshold) & (curve["headroom_to"] <= threshold + 150)]
    rate = lambda t: float((t["arrears_rate"] * t["loans"]).sum() / t["loans"].sum()) if len(t) else np.nan  # noqa: E731
    floor = curve["fitted"].min()
    level_off = curve.loc[curve["fitted"] <= floor * 1.25, "headroom_from"].min()
    return {
        "threshold": threshold,
        "arrears_below": rate(below),
        "arrears_above": rate(above),
        "loans_below": int(below["loans"].sum()),
        "risk_levels_off_at": float(level_off),
        "floor_rate": float(floor),
    }


def _label(col, val) -> str:
    if col == "vulnerable":
        return "Vulnerable customers" if int(val) == 1 else "Not flagged vulnerable"
    if col == "age_band":
        return f"Age {val}"
    if col == "income_type":
        return f"Income: {val}"
    return f"{col} = {val}"


def group_outcomes(df: pd.DataFrame, columns: list[str], min_gap: float = 0.02) -> pd.DataFrame:
    """Approval and arrears by customer group, each group compared with everyone else.

    A group is flagged when the gap is statistically significant after a Benjamini-Hochberg
    correction across all comparisons AND larger than `min_gap` (materiality).
    """
    obs = _observed(df)
    rows = []
    for col in columns:
        for val in sorted(df[col].dropna().unique()):
            g_all, r_all = df[df[col] == val], df[df[col] != val]
            g_obs, r_obs = obs[obs[col] == val], obs[obs[col] != val]
            for metric, g, r, flag_if in (
                ("approval_rate", (g_all["decision"] == "approve"), (r_all["decision"] == "approve"), "lower"),
                ("arrears_rate", g_obs["arrears_3m"], r_obs["arrears_3m"], "higher"),
            ):
                n1, n2 = len(g), len(r)
                if n1 < 30 or n2 < 30:
                    continue
                p1, p2 = float(np.mean(g)), float(np.mean(r))
                pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
                se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
                z = (p1 - p2) / se if se > 0 else 0.0
                rows.append(
                    {
                        "group": _label(col, val),
                        "metric": metric,
                        "n": n1,
                        "rate": p1,
                        "rest": p2,
                        "gap": p1 - p2,
                        "p_value": float(2 * stats.norm.sf(abs(z))),
                        "worse_if": flag_if,
                    }
                )
    t = pd.DataFrame(rows)
    t["q_value"] = benjamini_hochberg(t["p_value"].to_numpy())
    worse = np.where(t["worse_if"] == "lower", t["gap"] < -min_gap, t["gap"] > min_gap)
    t["flag"] = (t["q_value"] < 0.05) & worse
    return t.drop(columns="worse_if")


def fit_arrears_model(df: pd.DataFrame, test_cell_weight: float):
    """Arrears model on approved loans plus the test cell, the test cell upweighted to stand for all declines."""
    seen = df[df["arrears_3m"].notna()]
    w = np.where(seen["decision"] == "approve (test cell)", test_cell_weight, 1.0)
    pre = ColumnTransformer(
        [
            ("spline", SplineTransformer(n_knots=6, degree=3), ["headroom"]),
            ("lin", StandardScaler(), ["bureau_score", "income_volatility", "vulnerable"]),
        ]
    )
    model = make_pipeline(pre, LogisticRegression(max_iter=2000))
    model.fit(seen[["headroom", "bureau_score", "income_volatility", "vulnerable"]], seen["arrears_3m"], logisticregression__sample_weight=w)
    return model


def policy_frontier(df: pd.DataFrame, model, headroom_grid=None, score_grid=None) -> pd.DataFrame:
    """Approval rate and expected arrears for every (headroom, score) policy on the grid; marks efficient ones."""
    headroom_grid = np.arange(0, 401, 25) if headroom_grid is None else headroom_grid
    score_grid = np.arange(500, 661, 10) if score_grid is None else score_grid
    x = df[["headroom", "bureau_score", "income_volatility", "vulnerable"]]
    p = model.predict_proba(x)[:, 1]
    rows = []
    for h in headroom_grid:
        for s in score_grid:
            ok = (df["headroom"] >= h) & (df["bureau_score"] >= s)
            if ok.sum() == 0:
                continue
            rows.append({"min_headroom": h, "min_score": s, "approval_rate": float(ok.mean()), "expected_arrears": float(p[ok].mean())})
    t = pd.DataFrame(rows).sort_values(["approval_rate", "expected_arrears"], ascending=[False, True]).reset_index(drop=True)
    best = np.inf
    eff = []
    for v in t["expected_arrears"]:
        eff.append(v < best)
        best = min(best, v)
    t["efficient"] = eff
    return t


def better_alternative(frontier: pd.DataFrame, current: dict, tolerance: float = 0.005) -> dict | None:
    """An efficient policy with about the same approval rate and lower expected arrears, if one exists."""
    eff = frontier[frontier["efficient"]]
    near = eff[(eff["approval_rate"] >= current["approval_rate"] - tolerance)]
    if near.empty:
        return None
    alt = near.sort_values("expected_arrears").iloc[0]
    if alt["expected_arrears"] >= current["expected_arrears"] - 0.001:
        return None
    return alt.to_dict()


def borderline_declines(df: pd.DataFrame, min_headroom: float, min_score: float, sample: int = 50, seed: int = 0) -> pd.DataFrame:
    """Declines within £50 of the headroom threshold or 20 points of the score cut-off, sampled across age bands."""
    d = df[df["decision"] == "decline"]
    near = d[((d["headroom"] >= min_headroom - 50) & (d["bureau_score"] >= min_score)) | ((d["bureau_score"] >= min_score - 20) & (d["headroom"] >= min_headroom))]
    if near.empty:
        return near
    per = max(1, sample // near["age_band"].nunique())
    picked = pd.concat([g.sample(min(len(g), per), random_state=seed) for _, g in near.groupby("age_band")])
    return picked.sort_values("application_id")
