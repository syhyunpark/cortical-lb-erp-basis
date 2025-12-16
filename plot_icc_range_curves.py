#!/usr/bin/env python3
# plot_icc_range_curves.py 
# Plot ICC(3,1) vs mode-range for LB/SPH using summarize_icc_ranges output.

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def slugify(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icc-ranges-csv", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--min-icc", type=float, default=-0.1)
    ap.add_argument("--max-icc", type=float, default=1.0)
    ap.add_argument("--omit-k1", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    path = Path(args.icc_ranges_csv)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")

    outdir = Path(args.outdir) if args.outdir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("empty CSV")

    if args.tasks:
        df = df[df["task"].isin(args.tasks)]
        if df.empty:
            raise RuntimeError("no rows after task filter")

    if args.omit_k1:
        df = df[df["range_label"] != "k1"]

    # this matches what we used elsewhere; anything unknown goes to end
    range_order = ["k2_4", "k5_9", "k10_16", "k17_25"]
    df["range_order_idx"] = df["range_label"].apply(
        lambda r: range_order.index(r) if r in range_order else len(range_order)
    )

    for task in sorted(df["task"].unique()):
        df_t = df[df["task"] == task]
        for cond in sorted(df_t["condition"].unique()):
            df_tc = df_t[df_t["condition"] == cond]
            if df_tc.empty:
                continue

            plt.figure(figsize=(6, 4))

            for method in sorted(df_tc["method"].unique()):
                sub = df_tc[df_tc["method"] == method].copy()
                sub = sub.sort_values("range_order_idx")

                x = np.arange(len(sub))
                labels = sub["range_label"].values
                mu = sub["ICC_mean_range"].values
                lo = sub["ICC_lo_mean"].values
                hi = sub["ICC_hi_mean"].values

                plt.plot(x, mu, marker="o", label=method)

                ok = np.isfinite(lo) & np.isfinite(hi)
                if ok.any():
                    plt.fill_between(x[ok], lo[ok], hi[ok], alpha=0.2)

            plt.axhline(0.0, color="k", linestyle="--", linewidth=0.5)

            # labels from last "sub" loop are fine here since ranges are shared
            plt.xticks(np.arange(len(labels)), labels, rotation=45)
            plt.xlabel("Mode range")
            plt.ylabel("ICC(3,1)")
            plt.ylim(args.min_icc, args.max_icc)
            plt.title(f"{task} - {cond}")
            plt.legend(loc="best")
            plt.tight_layout()

            if args.save:
                fname = outdir / f"icc_ranges_{task}_{slugify(cond)}.png"
                plt.savefig(fname, dpi=150)
                print("[saved]", fname)

            plt.close()

    print("\n[done]")


if __name__ == "__main__":
    main()
