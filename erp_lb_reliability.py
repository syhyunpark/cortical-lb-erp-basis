#!/usr/bin/env python3
"""
erp_lb_reliability.py

Split-half ICC(3,1) for LB/SPH mode scores from ERP-CORE trial-level TFR.
Writes  icc_modewise.csv
"""

import argparse
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


COMP_WINDOWS = {
    "N170": (0.110, 0.150),
    "N2PC": (0.200, 0.275),
    "N400": (0.300, 0.500),
    "P3":   (0.300, 0.600),
    "ERN":  (0.000, 0.100),
    "LRP":  (-0.100, 0.000),
    "MMN":  (0.125, 0.225),
}


def load_lb_sph_dict(dict_path):
    d = np.load(str(dict_path), allow_pickle=True)
    for k in ("channels", "D_lb", "B_sph"):
        if k not in d:
            raise KeyError(f"{dict_path} missing key {k} (has {list(d.keys())})")

    ch = np.array(d["channels"]).astype(str)
    D = np.asarray(d["D_lb"], float)
    B = np.asarray(d["B_sph"], float)
    if D.shape[0] != len(ch) or B.shape[0] != len(ch):
        raise ValueError("dict rows != len(channels)")
    return ch, D, B


def icc_3_1(values):
    values = np.asarray(values, float)
    n, k = values.shape
    if k != 2:
        raise ValueError(f"icc_3_1 expects 2 columns, got {k}")
    if n < 2:
        return np.nan

    mean_subj = values.mean(axis=1, keepdims=True)
    mean_rater = values.mean(axis=0, keepdims=True)
    grand = values.mean()

    ss_bs = k * np.sum((mean_subj - grand) ** 2)
    ss_err = np.sum((values - mean_subj - mean_rater + grand) ** 2)

    df_bs = n - 1
    df_err = (n - 1) * (k - 1)
    if df_bs <= 0 or df_err <= 0:
        return np.nan

    ms_bs = ss_bs / df_bs
    ms_err = ss_err / df_err
    denom = ms_bs + (k - 1) * ms_err
    if denom <= 0:
        return np.nan

    return float((ms_bs - ms_err) / denom)


def bootstrap_icc(A, Bv, B=2000, alpha=0.05, rng=None):
    A = np.asarray(A, float)
    Bv = np.asarray(Bv, float)
    ok = np.isfinite(A) & np.isfinite(Bv)
    A = A[ok]
    Bv = Bv[ok]
    n = A.size
    if n < 2:
        return np.nan, np.nan, np.nan, 0

    if rng is None:
        rng = np.random.default_rng(0)

    data = np.column_stack([A, Bv])
    icc_full = icc_3_1(data)

    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        v = icc_3_1(data[idx])
        if np.isfinite(v):
            boots.append(float(v))

    if not boots:
        return float(icc_full), np.nan, np.nan, n

    boots = np.asarray(boots, float)
    lo = float(np.percentile(boots, 100 * alpha / 2.0))
    hi = float(np.percentile(boots, 100 * (1.0 - alpha / 2.0)))
    return float(icc_full), lo, hi, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfr-root", required=True)
    ap.add_argument("--dict-npz", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--min-trials", type=int, default=20)
    ap.add_argument("--min-subjects", type=int, default=10)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    tfr_root = Path(os.path.expanduser(args.tfr_root))
    outdir = Path(os.path.expanduser(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    dict_channels, D_lb_full, B_sph_full = load_lb_sph_dict(Path(os.path.expanduser(args.dict_npz)))
    K_lb = D_lb_full.shape[1]
    K_sph = B_sph_full.shape[1]

    pat = re.compile(r"^sub-(?P<sub>[^_]+)_task-(?P<task>[^_]+)_Y_tfr\.npz$")
    files_by_task = defaultdict(list)

    all_npz = sorted(tfr_root.rglob("*_Y_tfr.npz"))
    if not all_npz:
        raise RuntimeError(f"no '*_Y_tfr.npz' under {tfr_root}")

    for f in all_npz:
        m = pat.match(f.name)
        if not m:
            continue
        subj = m.group("sub")
        task = m.group("task")
        if args.tasks and task not in args.tasks:
            continue
        files_by_task[task].append((subj, f))

    if not files_by_task:
        raise RuntimeError("no subject×task files after filtering")

    rng = np.random.default_rng(0)
    rows = []

    for task, entries in sorted(files_by_task.items()):
        print(f"\n[task] {task}  (files={len(entries)})")

        cond_names = set()
        for _, npz_path in entries:
            ds_tmp = np.load(str(npz_path), allow_pickle=True)
            for k in ds_tmp.files:
                if k.startswith("Y_tfr_trials__"):
                    cond_names.add(k.replace("Y_tfr_trials__", ""))
        cond_names = sorted(cond_names)
        if not cond_names:
            print("  no Y_tfr_trials__* keys, skip task")
            continue

        tname = task.upper()
        w0, w1 = COMP_WINDOWS.get(tname, (None, None))

        for cond in cond_names:
            print(f"  [cond] {cond}")

            scores_A = {"LB": [], "SPH": []}
            scores_B = {"LB": [], "SPH": []}

            for subj, npz_path in sorted(entries, key=lambda x: x[0]):
                ds = np.load(str(npz_path), allow_pickle=True)
                key = f"Y_tfr_trials__{cond}"
                if key not in ds:
                    continue

                Y = ds[key]
                if getattr(Y, "ndim", 0) != 4:
                    print(f"    weird shape {Y.shape} for {npz_path.name}:{key}")
                    continue

                N, T, F, M_subj = Y.shape
                if N < args.min_trials or N < 2:
                    continue

                times = np.asarray(ds["times"], float)
                freqs = np.asarray(ds["freqs"], float)
                canonical = np.array(ds["canonical_channels"]).astype(str)
                subj_mask = np.asarray(ds["subject_mask"], bool)

                if M_subj != int(subj_mask.sum()):
                    continue

                if w0 is not None:
                    tmask = (times >= w0) & (times <= w1)
                    if not tmask.any():
                        tmask = np.ones_like(times, bool)
                else:
                    tmask = np.ones_like(times, bool)

                fmask = np.ones_like(freqs, bool)

                canon_idx = [i for i, m in enumerate(subj_mask) if m]
                subj_ch = [canonical[i] for i in canon_idx]

                subj_map = {c.lower(): j for j, c in enumerate(subj_ch)}
                dict_map = {c.lower(): i for i, c in enumerate(dict_channels)}

                common = []
                for c_low, i_d in dict_map.items():
                    j_s = subj_map.get(c_low)
                    if j_s is not None:
                        common.append((i_d, j_s))

                if len(common) < 8:
                    continue

                dict_keep = [p[0] for p in common]
                subj_keep = [p[1] for p in common]

                D_lb = D_lb_full[dict_keep, :]
                B_sph = B_sph_full[dict_keep, :]

                half = N // 2
                if half < 1 or (N - half) < 1:
                    continue

                P_A = Y[:half].mean(axis=0)
                P_B = Y[half:].mean(axis=0)

                P_A = P_A[tmask][:, fmask][:, :, subj_keep]
                P_B = P_B[tmask][:, fmask][:, :, subj_keep]

                T_sel, F_sel, M_common = P_A.shape
                R = T_sel * F_sel
                if R <= 0:
                    continue

                X_A = P_A.reshape(R, M_common)
                X_B = P_B.reshape(R, M_common)

                C_lb_A = X_A @ D_lb
                C_lb_B = X_B @ D_lb
                C_sph_A = X_A @ B_sph
                C_sph_B = X_B @ B_sph

                scores_A["LB"].append(C_lb_A.mean(axis=0))
                scores_B["LB"].append(C_lb_B.mean(axis=0))
                scores_A["SPH"].append(C_sph_A.mean(axis=0))
                scores_B["SPH"].append(C_sph_B.mean(axis=0))

            for method, K in (("LB", K_lb), ("SPH", K_sph)):
                if not scores_A[method]:
                    for k in range(K):
                        rows.append(
                            dict(
                                task=task, condition=cond, method=method, k=k + 1,
                                ICC_mean=np.nan, ICC_lo=np.nan, ICC_hi=np.nan,
                                n_subjects=0, B=int(args.B), alpha=float(args.alpha),
                            )
                        )
                    continue

                A_mat = np.vstack(scores_A[method])
                B_mat = np.vstack(scores_B[method])

                for k in range(K):
                    icc_m, icc_lo, icc_hi, n_eff = bootstrap_icc(
                        A_mat[:, k], B_mat[:, k], B=args.B, alpha=args.alpha, rng=rng
                    )
                    if n_eff < args.min_subjects:
                        icc_m, icc_lo, icc_hi = np.nan, np.nan, np.nan

                    rows.append(
                        dict(
                            task=task, condition=cond, method=method, k=k + 1,
                            ICC_mean=icc_m, ICC_lo=icc_lo, ICC_hi=icc_hi,
                            n_subjects=int(n_eff), B=int(args.B), alpha=float(args.alpha),
                        )
                    )

    out_csv = outdir / "icc_modewise.csv"
    df = pd.DataFrame(rows).sort_values(["task", "condition", "method", "k"])
    df.to_csv(out_csv, index=False)
    print("\n[wrote]", out_csv)


if __name__ == "__main__":
    main()
