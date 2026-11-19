"""Build the evidence pack: run the analyses, write findings, render one HTML file."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Template  # noqa: E402

from .analysis import (  # noqa: E402
    better_alternative,
    borderline_declines,
    check_depth,
    fit_arrears_model,
    group_outcomes,
    headroom_curve,
    policy_frontier,
    threshold_evidence,
)

NAVY, BLUE, GOLD, RED, GREEN, GREY = "#0E284B", "#3A85BD", "#B88E3A", "#B23A3A", "#2E7D5B", "#8A94A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY})
PCT = matplotlib.ticker.PercentFormatter(1.0, decimals=0)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def headroom_chart(curve, ev):
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    mid = (curve["headroom_from"] + curve["headroom_to"]) / 2
    ax.errorbar(mid, curve["arrears_rate"], yerr=[curve["arrears_rate"] - curve["ci_low"], curve["ci_high"] - curve["arrears_rate"]], fmt="o", ms=4, color=BLUE, capsize=2, label="Observed (95% credible interval)")
    ax.step(curve["headroom_from"], curve["fitted"], where="post", color=NAVY, lw=2, label="Monotone fit")
    ax.axvline(ev["threshold"], color=GOLD, lw=2, ls="--", label=f"Current threshold £{ev['threshold']:.0f} (shaded: test cell only)")
    ax.axvspan(curve["headroom_from"].min(), ev["threshold"], color=GOLD, alpha=0.07)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Monthly headroom after the new repayment (£)")
    ax.set_ylabel("3-month arrears rate")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Where does affordability risk rise?", loc="left", color=NAVY, weight="bold")
    return _png(fig)


def frontier_chart(frontier, current, alt):
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.scatter(frontier["approval_rate"], frontier["expected_arrears"], s=8, color="#C9D2DE", label="Policies tested")
    eff = frontier[frontier["efficient"]].sort_values("approval_rate")
    ax.plot(eff["approval_rate"], eff["expected_arrears"], color=NAVY, lw=2, label="Efficient frontier")
    ax.scatter([current["approval_rate"]], [current["expected_arrears"]], s=90, color=GOLD, zorder=5, label="Current policy")
    if alt:
        ax.scatter([alt["approval_rate"]], [alt["expected_arrears"]], s=90, marker="D", color=GREEN, zorder=5, label="Same approvals, fewer arrears")
        ax.annotate("", xy=(alt["approval_rate"], alt["expected_arrears"]), xytext=(current["approval_rate"], current["expected_arrears"]), arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Approval rate")
    ax.set_ylabel("Expected 3-month arrears")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("What would a policy change cost?", loc="left", color=NAVY, weight="bold")
    return _png(fig)


@dataclass
class EvidencePack:
    depth: pd.DataFrame
    curve: pd.DataFrame
    threshold: dict
    groups: pd.DataFrame
    frontier: pd.DataFrame
    current: dict
    alternative: dict | None
    borderline: pd.DataFrame
    findings: list[str]
    html: str


def build_pack(df: pd.DataFrame, min_headroom: float, min_score: float, test_cell_rate: float, lender="Example lender", synthetic=True) -> EvidencePack:
    depth = check_depth(df)
    curve = headroom_curve(df)
    ev = threshold_evidence(curve, min_headroom)
    groups = group_outcomes(df, ["age_band", "income_type", "vulnerable"])
    model = fit_arrears_model(df, 1.0 / test_cell_rate)
    frontier = policy_frontier(df, model)
    cur_row = frontier[(frontier["min_headroom"] == min_headroom) & (frontier["min_score"] == min_score)].iloc[0]
    current = cur_row.to_dict()
    alt = better_alternative(frontier, current)
    border = borderline_declines(df, min_headroom, min_score)

    f = []
    small, large = depth.iloc[0], depth.iloc[-1]
    f.append(
        f"Checks scale with the amount: {small['open_banking_share']:.0%} of applications in the {small['amount_band']} band are verified with open banking, "
        f"rising to {large['open_banking_share']:.0%} {large['amount_band']}."
    )
    f.append(
        ("Risk rises below the threshold" if ev["arrears_below"] > ev["arrears_above"] else "Risk does not clearly rise below the threshold, so it may be stricter than it needs to be")
        + f": arrears are {ev['arrears_below']:.1%} for headroom between £{min_headroom - 150:.0f} and £{min_headroom:.0f} "
        f"(test cell, {ev['loans_below']} loans) against {ev['arrears_above']:.1%} between £{min_headroom:.0f} and £{min_headroom + 150:.0f}. The fitted curve levels off from about £{ev['risk_levels_off_at']:.0f}."
    )
    flagged = groups[groups["flag"]]
    if flagged.empty:
        f.append("No customer group shows a statistically significant and material gap in approval or arrears outcomes.")
    else:
        parts = [f"{r.group} ({'approval' if r.metric == 'approval_rate' else 'arrears'} {r.rate:.1%} vs {r.rest:.1%})" for r in flagged.itertuples()]
        f.append("Groups with significantly worse outcomes, to review for fair value and support: " + "; ".join(parts) + ".")
    if alt:
        f.append(
            f"A different mix of rules (headroom ≥ £{alt['min_headroom']:.0f}, score ≥ {alt['min_score']:.0f}) is estimated to keep approvals at "
            f"{alt['approval_rate']:.1%} (now {current['approval_rate']:.1%}) with arrears of {alt['expected_arrears']:.2%} (now {current['expected_arrears']:.2%})."
        )
    else:
        f.append("The current policy sits on the efficient frontier: no tested policy gives the same approvals with fewer arrears.")
    f.append(f"{len(border)} borderline declines are listed for manual review, sampled across age bands.")

    show_groups = groups.sort_values(["flag", "q_value"], ascending=[False, True]).head(12)
    html = TEMPLATE.render(
        lender=lender,
        synthetic=synthetic,
        today=date.today().isoformat(),
        n=f"{len(df):,}",
        approval=f"{(df['decision'] == 'approve').mean():.1%}",
        findings=f,
        depth=depth.to_dict("records"),
        headroom_png=headroom_chart(curve, ev),
        groups=show_groups.to_dict("records"),
        frontier_png=frontier_chart(frontier, current, alt),
        border_n=len(border),
        border=border.head(8)[["application_id", "age_band", "headroom", "bureau_score", "amount"]].to_dict("records"),
        min_headroom=min_headroom,
        min_score=min_score,
        test_cell=f"{test_cell_rate:.0%}",
    )
    return EvidencePack(depth, curve, ev, groups, frontier, current, alt, border, f, html)


TEMPLATE = Template(
    """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Affordability Evidence Pack</title>
<style>
body{font-family:Inter,Segoe UI,Helvetica,Arial,sans-serif;color:#1B1F24;max-width:900px;margin:32px auto;padding:0 24px;line-height:1.5}
header{border-top:8px solid #B88E3A;padding-top:16px;display:flex;justify-content:space-between;align-items:baseline}
h1{color:#0E284B;margin:0;font-size:28px} h2{color:#0E284B;margin-top:36px;font-size:20px}
.brand{color:#0E284B;font-weight:700;letter-spacing:.08em;font-size:12px}
.note{background:#F4F6F9;border-left:4px solid #3A85BD;padding:10px 14px;font-size:14px}
table{border-collapse:collapse;width:100%;font-size:14px} td,th{padding:7px;border-bottom:1px solid #E3E7EE;text-align:left}
.flag{color:#B23A3A;font-weight:700} img{width:100%;max-width:740px} li{margin-bottom:6px}
footer{margin-top:40px;color:#5B6670;font-size:12px}
</style></head><body>
<header><h1>Affordability Evidence Pack</h1><span class="brand">SKOPOS CREDIT ANALYTICS</span></header>
<p>{{ lender }} · {{ n }} applications · approval rate {{ approval }} · policy: headroom ≥ £{{ '%.0f'|format(min_headroom) }} and bureau score ≥ {{ '%.0f'|format(min_score) }} · generated {{ today }}</p>
{% if synthetic %}<p class="note">Demonstration on simulated data. No real lender or customer data is used.</p>{% endif %}
<h2>Key findings</h2><ol>{% for x in findings %}<li>{{ x }}</li>{% endfor %}</ol>
<h2>1. Are the checks proportionate?</h2>
<p>Deeper verification should be used where more is at stake.</p>
<table><tr><th>Amount</th><th>Applications</th><th>Verified with open banking</th><th>Approval rate</th><th>3-month arrears</th></tr>
{% for r in depth %}<tr><td>{{ r.amount_band }}</td><td>{{ '{:,}'.format(r.applications) }}</td><td>{{ '%.0f%%'|format(r.open_banking_share*100) }}</td><td>{{ '%.1f%%'|format(r.approval_rate*100) }}</td><td>{{ '%.1f%%'|format(r.arrears_rate*100) }}</td></tr>{% endfor %}</table>
<h2>2. Is the threshold where risk rises?</h2>
<p>Outcomes below the threshold come from a {{ test_cell }} random test cell of applications the policy would have declined. Without a test cell, nobody can see what happens below the line.</p>
<img src="data:image/png;base64,{{ headroom_png }}">
<h2>3. Do outcomes differ across customer groups?</h2>
<p>Each group compared with everyone else. Flagged only if the gap is statistically significant after correcting for multiple comparisons <i>and</i> larger than 2 points.</p>
<table><tr><th>Group</th><th>Measure</th><th>Group rate</th><th>Everyone else</th><th>q-value</th><th></th></tr>
{% for r in groups %}<tr><td>{{ r.group }}</td><td>{{ 'Approval' if r.metric == 'approval_rate' else 'Arrears' }}</td><td>{{ '%.1f%%'|format(r.rate*100) }}</td><td>{{ '%.1f%%'|format(r.rest*100) }}</td><td>{{ '%.3f'|format(r.q_value) }}</td><td>{% if r.flag %}<span class="flag">Review</span>{% endif %}</td></tr>{% endfor %}</table>
<h2>4. What would a policy change cost?</h2>
<p>Every combination of headroom and score threshold on a grid, scored with an arrears model trained on approved loans and the reweighted test cell. The frontier shows the lowest expected arrears achievable at each approval rate.</p>
<img src="data:image/png;base64,{{ frontier_png }}">
<h2>5. Borderline declines for manual review</h2>
<p>{{ border_n }} declined applications within £50 of the headroom threshold or 20 points of the score cut-off, sampled across age bands. First rows:</p>
<table><tr><th>Application</th><th>Age band</th><th>Headroom (£)</th><th>Bureau score</th><th>Amount (£)</th></tr>
{% for r in border %}<tr><td>{{ r.application_id }}</td><td>{{ r.age_band }}</td><td>{{ '%.0f'|format(r.headroom) }}</td><td>{{ '%.0f'|format(r.bureau_score) }}</td><td>{{ '%.0f'|format(r.amount) }}</td></tr>{% endfor %}</table>
<footer>Method notes: intervals are Bayesian (Jeffreys prior); group comparisons use two-proportion z-tests with Benjamini-Hochberg correction; the headroom curve is an isotonic (monotone) fit; the arrears model is a spline logistic regression with the test cell weighted by the inverse of its sampling rate. This pack supports a firm's own assessment of its affordability approach. It is not legal or regulatory advice and does not confirm compliance.</footer>
</body></html>"""
)
