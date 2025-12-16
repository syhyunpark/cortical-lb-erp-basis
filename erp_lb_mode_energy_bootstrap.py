#!/usr/bin/env python3
"""
erp_lb_mode_energy_bootstrap.py

Bootstrap CIs for mode-wise TF energy concentration metrics (K70/K90 and
energy in the first m modes) for each (task, method).... 
Writes mode_energy_bootstrap.csv 
"""

import os
import argparse
import numpy as np
import pandas as pd


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def bootstrap_mean_ci(vals, B=2000, alpha=0.05, rng=None):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, (np.nan, np.nan)

    if rng is None:
        rng = np.random.default_rng(0)

    n = vals.size
    boots = np.array([np.mean(vals[rng.integers(0, n, n)]) for _ in range(B)], dtype=float)

    mu = float(np.mean(vals))
    lo = float(np.percentile(boots, 100 * alpha / 2.0))
    hi = float(np.percentile(boots, 100 * (1.0 - alpha / 2.0)))
    return mu, (lo, hi)


def main():
    ap = argparse.ArgumentParser(description="Bootstrap CIs for LB/SPH mode-energy metrics.")
    ap.add_argument(
        "--input-csv",
        required=True,
        help="CSV with columns: subject, task, method, mode_index, energy",
    )
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha for percentile CI (default: 0.05 -> 95%% CI)",
    )
    ap.add_argument("--B", type=int, default=2000, help="Number of bootstrap resamples")
    args = ap.parse_args()

    outdir = ensure_dir(os.path.expanduser(args.outdir))
    csv_path = os.path.expanduser(args.input_csv)
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    need = {"subject", "task", "method", "mode_index", "energy"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise SystemExit(f"Input CSV missing columns: {missing}")

    df["mode_index"] = df["mode_index"].astype(int)
    df["energy"] = df["energy"].astype(float)
    df["subject"] = df["subject"].astype(str)
    df["task"] = df["task"].astype(str)
    df["method"] = df["method"].astype(str)

    print(f"[info] Loaded {df.shape[0]} rows: {csv_path}")
    print(f"[info] Tasks:   {sorted(df['task'].unique())}")
    print(f"[info] Methods: {sorted(df['method'].unique())}")

    rng = np.random.default_rng(0)
    rows_out = []
    top_m_list = [3, 5, 10, 15, 20]

    for (task, method), g_tm in df.groupby(["task", "method"]):
        n_subj_raw = g_tm["subject"].nunique()
        print(f"\n[info] task={task}, method={method}: {n_subj_raw} subjects")

        per_subj = []
        for subj, g_s in g_tm.groupby("subject"):
            g_s = g_s.sort_values("mode_index")
            energies = g_s["energy"].to_numpy()

            total = energies.sum()
            if total <= 0 or not np.isfinite(total):
                print(f"  [warn] subject {subj}: bad total energy; skipping")
                continue

            E_tilde = energies / total
            K_max = len(E_tilde)
            C = np.cumsum(E_tilde)

            def first_K(cum, thr):
                idx = np.where(cum >= thr)[0]
                return int(idx[0] + 1) if idx.size else np.nan

            K70 = first_K(C, 0.70)
            K90 = first_K(C, 0.90)

            out = {"subject": subj, "K70": K70, "K90": K90}
            for m in top_m_list:
                m_eff = min(m, K_max)
                out[f"energy_top_{m}"] = float(E_tilde[:m_eff].sum())
            per_subj.append(out)

        if not per_subj:
            print("  [warn] no usable subjects; skipping")
            continue

        subj_df = pd.DataFrame(per_subj)
        N = int(subj_df.shape[0])
        print(f"  [info] Using {N} subjects")

        metrics = ["K70", "K90"] + [f"energy_top_{m}" for m in top_m_list]
        for metric in metrics:
            vals = subj_df[metric].to_numpy()
            mu, (lo, hi) = bootstrap_mean_ci(vals, B=args.B, alpha=args.alpha, rng=rng)
            rows_out.append(
                dict(
                    task=task,
                    method=method,
                    metric=metric,
                    n_subjects=N,
                    mean=mu,
                    ci_low=lo,
                    ci_high=hi,
                    B=int(args.B),
                    alpha=float(args.alpha),
                )
            )

    if rows_out:
        out_df = pd.DataFrame(rows_out).sort_values(["task", "method", "metric"])
    else:
        print("[warn] No metrics computed; writing empty output.")
        out_df = pd.DataFrame(
            columns=["task", "method", "metric", "n_subjects", "mean", "ci_low", "ci_high", "B", "alpha"]
        )

    out_csv = os.path.join(outdir, "mode_energy_bootstrap.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\n[wrote] {out_csv} (rows={out_df.shape[0]})")


if __name__ == "__main__":
    main()
