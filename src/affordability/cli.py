"""Command line entry point.

    python -m affordability.cli --demo
    python -m affordability.cli --data applications.csv --min-headroom 150 --min-score 560 --test-cell-rate 0.03

The CSV needs: application_id, decision ('approve', 'approve (test cell)', 'decline'), arrears_3m
(blank when not lent), headroom, bureau_score, income_volatility, vulnerable, amount, verification,
age_band and income_type.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .pack import build_pack
from .simulate import simulate_applications


def main(argv=None):
    ap = argparse.ArgumentParser(description="Affordability evidence pack")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--min-headroom", type=float, default=150.0)
    ap.add_argument("--min-score", type=float, default=560.0)
    ap.add_argument("--test-cell-rate", type=float, default=0.05)
    ap.add_argument("--out", default="examples/affordability_evidence_pack.html")
    a = ap.parse_args(argv)
    if a.demo:
        df = simulate_applications(40000, seed=7, test_cell_rate=a.test_cell_rate)
        lender = "Simulated BNPL lender"
        synthetic = True
    elif a.data:
        df, lender, synthetic = pd.read_csv(a.data), Path(a.data).stem, False
    else:
        ap.error("give --demo or --data")
    pack = build_pack(df, a.min_headroom, a.min_score, a.test_cell_rate, lender, synthetic)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pack.html, encoding="utf-8")
    pack.borderline.to_csv(out.with_name("borderline_declines.csv"), index=False)
    for i, x in enumerate(pack.findings, 1):
        print(f"{i}. {x}")
    print(f"Pack written to {out}")


if __name__ == "__main__":
    main()
