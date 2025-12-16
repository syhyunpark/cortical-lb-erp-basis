#!/usr/bin/env python3
# plot_group_auc.py
#
# Plot group mean R^2(K) curves (and bootstrap CI bands) from  group_AUC_curves.csv

import os
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def safe_tag(s):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--methods", nargs="+", default=["LB", "SPH", "PCA", "ICA"])
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    group_dir = os.path.join(root, "group")
    auc_csv = os.path.join(group_dir, "group_AUC_curves.csv")
    thr_csv = os.path.join(group_dir, "group_thresholds.csv")

    if not os.path.isfile(auc_csv):
        raise SystemExit(f"missing {auc_csv} (run erp_tfr_bootstrap.py first)")

    outdir = os.path.expanduser(args.outdir) if args.outdir else os.path.join(group_dir, "figures")
    ensure_dir(outdir)

    auc = pd.read_csv(auc_csv)
    if auc.empty:
        raise SystemExit("group_AUC_curves.csv is empty")

    methods = [m.upper() for m in args.methods]
    auc["method"] = auc["method"].astype(str).str.upper()
    auc = auc[auc["method"].isin(methods)]
    if auc.empty:
        raise SystemExit("no rows left after method filtering")

    if args.tasks:
        tasks_set = {t.upper() for t in args.tasks}
        auc["task"] = auc["task"].astype(str)
        auc = auc[auc["task"].str.upper().isin(tasks_set)]

    if args.conditions:
        cond_set = set(args.conditions)
        auc["condition"] = auc["condition"].astype(str)
        auc = auc[auc["condition"].isin(cond_set)]

    if auc.empty:
        raise SystemExit("no rows left after task/condition filtering")

    # optional, not really used right now
    if os.path.isfile(thr_csv):
        try:
            _ = pd.read_csv(thr_csv)
        except Exception as e:
            print("[warn] couldnt read group_thresholds.csv:", e)

    # keep this super simple; matplotlib default colors are fine
    marker_map = {"LB": "o", "SPH": "s", "PCA": "^", "ICA": "D"}

    auc["task"] = auc["task"].astype(str)
    auc["condition"] = auc["condition"].astype(str)

    for (task, cond), g_tc in auc.groupby(["task", "condition"]):
        plt.figure(figsize=(6, 4))

        for method in methods:
            g_m = g_tc[g_tc["method"] == method]
            if g_m.empty:
                continue
            g_m = g_m.sort_values("K")

            K = g_m["K"].to_numpy()
            mu = g_m["AUC_mean"].to_numpy() if "AUC_mean" in g_m.columns else g_m["AUC"].to_numpy()

            plt.plot(K, mu, marker=marker_map.get(method, "o"), label=method)

            if ("AUC_lo" in g_m.columns) and ("AUC_hi" in g_m.columns):
                lo = g_m["AUC_lo"].to_numpy()
                hi = g_m["AUC_hi"].to_numpy()
                plt.fill_between(K, lo, hi, alpha=0.2)

        plt.ylim(0.0, 1.0)
        plt.xlabel("Number of modes K")
        plt.ylabel("Mean R^2")
        plt.title(f"{task} — {cond}")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.2)
        plt.tight_layout()

        fname = f"R2_group_task-{safe_tag(task)}_cond-{safe_tag(cond)}.png"
        fpath = os.path.join(outdir, fname)
        plt.savefig(fpath, dpi=150)
        plt.close()

        print("[wrote]", fpath)

    print("\n[done] figures in", outdir)


if __name__ == "__main__":
    main()
