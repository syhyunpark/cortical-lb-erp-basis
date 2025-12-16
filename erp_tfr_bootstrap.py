#!/usr/bin/env python3
# erp_tfr_bootstrap.py
#
# Bootstrap group summaries from ERP-TFR span outputs (resample subjects).

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
    boots = np.array([np.mean(vals[rng.integers(0, n, n)]) for _ in range(B)], float)
    mu = float(np.mean(vals))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return mu, (lo, hi)


def bootstrap_prop_ci(flags, B=2000, alpha=0.05, rng=None):
    flags = np.asarray(flags, float)
    flags = flags[np.isfinite(flags)]
    if flags.size == 0:
        return np.nan, (np.nan, np.nan)

    if rng is None:
        rng = np.random.default_rng(0)

    n = flags.size
    boots = np.array([np.mean(flags[rng.integers(0, n, n)]) for _ in range(B)], float)
    p = float(flags.mean())
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return p, (lo, hi)


def main():
    ap = argparse.ArgumentParser(description="ERP-TFR span bootstrap (group CIs)")
    ap.add_argument("--root", required=True, help="Root folder with tables/ from span run")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    tbl_dir = os.path.join(root, "tables")
    out_dir = ensure_dir(os.path.join(root, "group"))

    auc_csv = os.path.join(tbl_dir, "AUC_curves.csv")
    thr_csv = os.path.join(tbl_dir, "K_thresh.csv")

    if not (os.path.isfile(auc_csv) and os.path.isfile(thr_csv)):
        raise SystemExit(f"Missing AUC_curves.csv or K_thresh.csv under {tbl_dir}")

    AUC = pd.read_csv(auc_csv)
    THR = pd.read_csv(thr_csv)
    if AUC.empty or THR.empty:
        print("[warn] AUC or K_thresh empty")
        return

    rng = np.random.default_rng(0)

    # ---- AUC(K) ----
    auc_rows = []
    for (task, cond, method, K), g in AUC.groupby(["task", "condition", "method", "K"]):
        subj_vals = g.groupby("subject")["AUC"].mean().to_numpy()
        mu, (lo, hi) = bootstrap_mean_ci(subj_vals, B=args.B, alpha=args.alpha, rng=rng)
        auc_rows.append(
            dict(
                task=task, condition=cond, method=method, K=int(K),
                AUC_mean=mu, AUC_lo=lo, AUC_hi=hi,
                N=int(subj_vals.size), B=int(args.B), alpha=float(args.alpha),
            )
        )

    auc_out = pd.DataFrame(auc_rows).sort_values(["task", "condition", "method", "K"])
    out_auc = os.path.join(out_dir, "group_AUC_curves.csv")
    auc_out.to_csv(out_auc, index=False)
    print("[group] wrote", out_auc)

    # ---- thresholds ----
    thr_rows = []
    for (task, cond, method), g in THR.groupby(["task", "condition", "method"]):
        subj_grp = (
            g.groupby("subject")
            .agg({"K70": "mean", "K90": "mean", "attain_70": "max", "attain_90": "max"})
            .reset_index()
        )

        vals70 = subj_grp["K70"].to_numpy()
        vals90 = subj_grp["K90"].to_numpy()
        flags70 = subj_grp["attain_70"].to_numpy().astype(float)
        flags90 = subj_grp["attain_90"].to_numpy().astype(float)

        p70, (p70_lo, p70_hi) = bootstrap_prop_ci(flags70, B=args.B, alpha=args.alpha, rng=rng)
        p90, (p90_lo, p90_hi) = bootstrap_prop_ci(flags90, B=args.B, alpha=args.alpha, rng=rng)

        thr_rows.append(
            dict(task=task, condition=cond, method=method,
                 metric="attain_70", value=p70, lo=p70_lo, hi=p70_hi,
                 N=int(flags70.size), B=int(args.B), alpha=float(args.alpha))
        )
        thr_rows.append(
            dict(task=task, condition=cond, method=method,
                 metric="attain_90", value=p90, lo=p90_lo, hi=p90_hi,
                 N=int(flags90.size), B=int(args.B), alpha=float(args.alpha))
        )

        if (flags70 > 0).any():
            v70 = vals70[flags70 > 0]
            mu70, (lo70, hi70) = bootstrap_mean_ci(v70, B=args.B, alpha=args.alpha, rng=rng)
            thr_rows.append(
                dict(task=task, condition=cond, method=method,
                     metric="K70_att_mean", value=mu70, lo=lo70, hi=hi70,
                     N=int(v70.size), B=int(args.B), alpha=float(args.alpha))
            )

        if (flags90 > 0).any():
            v90 = vals90[flags90 > 0]
            mu90, (lo90, hi90) = bootstrap_mean_ci(v90, B=args.B, alpha=args.alpha, rng=rng)
            thr_rows.append(
                dict(task=task, condition=cond, method=method,
                     metric="K90_att_mean", value=mu90, lo=lo90, hi=hi90,
                     N=int(v90.size), B=int(args.B), alpha=float(args.alpha))
            )

    thr_out = pd.DataFrame(thr_rows).sort_values(["task", "condition", "method", "metric"])
    out_thr = os.path.join(out_dir, "group_thresholds.csv")
    thr_out.to_csv(out_thr, index=False)
    print("[group] wrote", out_thr)


if __name__ == "__main__":
    main()
