#!/usr/bin/env python3
# summarize_icc_ranges.py 
# Summarize mode-wise ICC(3,1) into a few coarse mode ranges...

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icc-csv", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    icc_path = Path(args.icc_csv)
    if not icc_path.exists():
        raise FileNotFoundError(f"no file: {icc_path}")

    outdir = Path(args.outdir) if args.outdir else icc_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(icc_path)
    if df.empty:
        print("[warn] empty icc csv")
        out_csv = outdir / "icc_mode_ranges_summary.csv"
        pd.DataFrame(
            columns=[
                "task", "condition", "method", "range_label",
                "k_start", "k_end", "n_modes",
                "ICC_mean_range", "ICC_lo_mean", "ICC_hi_mean",
            ]
        ).to_csv(out_csv, index=False)
        print("[wrote]", out_csv)
        return

    # inclusive ranges
    mode_ranges = [
        ("k1_3", 1, 3),
        ("k4_8", 4, 8),
        ("k9_15", 9, 15),
        ("k16_24", 16, 24),
    ]

    rows = []
    for (task, cond, method), g in df.groupby(["task", "condition", "method"]):
        for label, k0, k1 in mode_ranges:
            sub = g[(g["k"] >= k0) & (g["k"] <= k1)]
            if sub.empty:
                continue
            rows.append(
                dict(
                    task=task,
                    condition=cond,
                    method=method,
                    range_label=label,
                    k_start=int(k0),
                    k_end=int(k1),
                    n_modes=int(sub["k"].nunique()),
                    ICC_mean_range=float(sub["ICC_mean"].mean()),
                    ICC_lo_mean=float(sub["ICC_lo"].mean()),
                    ICC_hi_mean=float(sub["ICC_hi"].mean()),
                )
            )

    out_csv = outdir / "icc_mode_ranges_summary.csv"
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["task", "condition", "method", "k_start", "k_end"])
    else:
        out = pd.DataFrame(
            columns=[
                "task", "condition", "method", "range_label",
                "k_start", "k_end", "n_modes",
                "ICC_mean_range", "ICC_lo_mean", "ICC_hi_mean",
            ]
        )

    out.to_csv(out_csv, index=False)
    print("[wrote]", out_csv)


if __name__ == "__main__":
    main()
