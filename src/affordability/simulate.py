"""Synthetic BNPL / small-loan applications with affordability data and outcomes.

Each application has open-banking style income and spending, a bureau score, the
lender's decision under a simple policy, and a 3-month arrears outcome. Outcomes are
observed for approved applications and for a small random "test cell" of applications
that the policy would have declined. Test cells like this are how a lender can learn
what happens below its threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGE_BANDS = ["18-24", "25-34", "35-49", "50-64", "65+"]


def current_policy(df: pd.DataFrame, min_headroom: float = 150.0, min_score: float = 560.0) -> np.ndarray:
    return (df["headroom"] >= min_headroom) & (df["bureau_score"] >= min_score)


def simulate_applications(n: int = 30000, seed: int = 0, test_cell_rate: float = 0.03) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age_band = rng.choice(AGE_BANDS, n, p=[0.22, 0.30, 0.27, 0.15, 0.06])
    young = age_band == "18-24"
    income_type = rng.choice(["salaried", "self-employed", "mainly benefits"], n, p=[0.72, 0.13, 0.15])
    income = np.exp(rng.normal(np.log(2100), 0.35, n)) * np.where(young, 0.75, 1.0)
    income *= np.where(income_type == "mainly benefits", 0.6, 1.0)
    essentials = income * np.clip(rng.normal(0.62, 0.12, n), 0.3, 0.95)
    commitments = np.where(rng.random(n) < 0.55, rng.gamma(2.0, 90, n), 0.0)
    amount = np.round(np.exp(rng.normal(np.log(350), 0.7, n)), -1).clip(50, 2000)
    term = rng.choice([3, 6, 12], n, p=[0.6, 0.25, 0.15])
    repayment = amount / term
    headroom = income - essentials - commitments - repayment
    income_volatility = rng.gamma(2.0, 0.07, n) * np.where(income_type == "self-employed", 1.8, 1.0)
    bureau_score = rng.normal(610, 55, n) - 25 * young
    vulnerable = (
        (rng.random(n) < 0.06)  # e.g. recent gambling spend or a support flag
        | ((income_type == "mainly benefits") & (rng.random(n) < 0.4))
    )

    # deeper checks for bigger amounts: open-banking verification more likely above £500
    p_ob = np.where(amount > 1000, 0.9, np.where(amount > 500, 0.6, 0.25))
    verification = np.where(rng.random(n) < p_ob, "open banking", "bureau only")

    logit = (
        -3.3
        + 1.9 * np.exp(-np.clip(headroom, -400, None) / 220.0)  # risk climbs as headroom shrinks
        - 0.010 * (bureau_score - 610)
        + 2.0 * income_volatility
        + 0.35 * vulnerable
        + 0.25 * young
    )
    p_arrears = 1 / (1 + np.exp(-logit))
    arrears = rng.binomial(1, p_arrears)

    df = pd.DataFrame(
        {
            "application_id": np.arange(1, n + 1),
            "age_band": age_band,
            "income_type": income_type,
            "vulnerable": vulnerable.astype(int),
            "net_monthly_income": income.round(0),
            "essential_spend": essentials.round(0),
            "credit_commitments": commitments.round(0),
            "amount": amount,
            "term_months": term,
            "monthly_repayment": repayment.round(2),
            "headroom": headroom.round(0),
            "income_volatility": income_volatility.round(3),
            "bureau_score": bureau_score.round(0),
            "verification": verification,
        }
    )
    approved = current_policy(df).to_numpy()
    test_cell = (~approved) & (rng.random(n) < test_cell_rate)
    df["decision"] = np.where(approved, "approve", np.where(test_cell, "approve (test cell)", "decline"))
    df["arrears_3m"] = np.where(approved | test_cell, arrears, np.nan)
    return df
