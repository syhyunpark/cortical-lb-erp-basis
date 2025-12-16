#!/usr/bin/env python3
# erp_tfr_span.py
#
# ERP-TFR span analysis with global LB/SPH/PCA/ICA bases...

import os
import re
import argparse
import warnings

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning


COMP_WINDOWS = {
    "N170": (0.110, 0.150),
    "N2PC": (0.200, 0.275),
    "N400": (0.300, 0.500),
    "P3":   (0.300, 0.600),
    "ERN":  (0.000, 0.100),
    "LRP":  (-0.100, 0.000),
    "MMN":  (0.125, 0.225),
}


def component_time_indices(task, times, eval_t_step):
    tname = str(task).upper()
    if tname in COMP_WINDOWS:
        w0, w1 = COMP_WINDOWS[tname]
    else:
        w0, w1 = float(times[0]), float(times[-1])
    tmask = (times >= w0) & (times <= w1)
    idx = np.where(tmask)[0]
    if idx.size == 0:
        idx = np.arange(times.size)
    return idx[::max(1, int(eval_t_step))]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def safe_tag(s):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", str(s))


def zscore_rows(A, eps=1e-12):
    A = np.asarray(A, float)
    mu = A.mean(axis=1, keepdims=True)
    sd = A.std(axis=1, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (A - mu) / (sd + eps)


def qr_no_pivot(B):
    if B.size == 0:
        return np.empty((B.shape[0], 0))
    Q, R = np.linalg.qr(B, mode="reduced")
    if R.size == 0:
        return np.empty((B.shape[0], 0))
    d = np.abs(np.diag(R))
    keep = d > (1e-12 * max(1.0, d.max()))
    return Q[:, keep] if keep.any() else np.empty((B.shape[0], 0))


def r2_rows(Y, Yhat):
    num = np.sum((Y - Yhat) ** 2, axis=1)
    den = np.sum(Y ** 2, axis=1) + 1e-12
    return 1.0 - (num / den)


def first_K_at_threshold(v, thr):
    idx = np.where(v >= thr)[0]
    return (idx[0] + 1) if idx.size else np.nan


def build_positions_std1005(ch_names):
    montage = mne.channels.make_standard_montage("standard_1005")
    pos_dict = montage.get_positions()["ch_pos"]
    pos_lower = {name.lower(): coord for name, coord in pos_dict.items()}
    coords = []
    for ch in ch_names:
        coord = pos_lower.get(str(ch).lower(), np.array([np.nan, np.nan, np.nan], float))
        coords.append(coord)
    P = np.array(coords, float)
    norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    return P / norms


def sph_Y(l, m, theta, phi):
    try:
        from scipy.special import sph_harm_y
        return sph_harm_y(l, m, theta, phi)
    except Exception:
        from scipy.special import sph_harm
        return sph_harm(m, l, theta, phi)


def sph_design_degree_major(pos_unit, Kbuild):
    if pos_unit.size == 0:
        return np.empty((0, 0))
    z = np.clip(pos_unit[:, 2], -1.0, 1.0)
    theta = np.arccos(z)
    phi = np.mod(np.arctan2(pos_unit[:, 1], pos_unit[:, 0]), 2 * np.pi)

    cols = []
    l = 1
    while len(cols) < Kbuild:
        for m in range(-l, l + 1):
            Ylm = sph_Y(l, m, theta, phi)
            if m < 0:
                v = np.sqrt(2.0) * np.imag(Ylm)
            elif m == 0:
                v = np.real(Ylm)
            else:
                v = np.sqrt(2.0) * np.real(Ylm)
            cols.append(np.asarray(v, float))
            if len(cols) >= Kbuild:
                break
        l += 1
        if l > 64:
            break
    return np.column_stack(cols) if cols else np.empty((pos_unit.shape[0], 0))


def pca_Q_full(Y, sv_cut=1e-8):
    Yc = Y - Y.mean(axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(Yc, full_matrices=False)
    if S.size == 0:
        return np.empty((Y.shape[1], 0))
    keep = S >= (sv_cut * S.max())
    Q = VT.T[:, keep]
    if Q.size:
        Q, _ = np.linalg.qr(Q, mode="reduced")
    return Q


def ica_mixing(Y, ncomp, tol=1e-5, max_iter=10000, seed=0):
    Yc = Y - Y.mean(axis=0, keepdims=True)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            ica = FastICA(
                n_components=ncomp,
                whiten="unit-variance",
                tol=tol,
                max_iter=max_iter,
                fun="logcosh",
                random_state=seed,
            )
            _ = ica.fit_transform(Yc)
            A = ica.mixing_
    except Exception:
        A = None
    return A


def load_dictionary(dict_npz):
    d = np.load(os.path.expanduser(dict_npz), allow_pickle=True)
    if ("D" not in d) or ("channels" not in d):
        raise KeyError(f"{dict_npz} must contain D and channels")
    D = np.asarray(d["D"], float)
    channels = [str(x) for x in d["channels"].tolist()]
    Kmax_dict = int(d["K"][0]) if "K" in d else D.shape[1]
    return D, channels, Kmax_dict


def main():
    ap = argparse.ArgumentParser(description="ERP-TFR span (global LB/SPH/PCA/ICA)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dict-npz", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--methods", nargs="+", default=["LB", "SPH", "PCA", "ICA"])
    ap.add_argument("--mode", choices=["trials", "avg", "bands"], default="trials")
    ap.add_argument("--K-list", default="1,2,3,4,5,7,9,10,15,16,20,25,30")
    ap.add_argument("--eval-t-step", type=int, default=2)
    ap.add_argument("--eval-f-step", type=int, default=1)
    ap.add_argument("--chunk-rows", type=int, default=100000)
    ap.add_argument("--prefix-qr", action="store_true")
    args = ap.parse_args()

    try:
        K_list = sorted({int(k) for k in args.K_list.split(",") if str(k).strip()})
    except Exception:
        K_list = [1, 2, 3, 4, 5, 7, 9, 10, 15, 16, 20, 25, 30]

    out_root = ensure_dir(os.path.expanduser(args.outdir))
    tbl_dir = ensure_dir(os.path.join(out_root, "tables"))
    fig_dir = ensure_dir(os.path.join(out_root, "figures"))

    D_full, dict_ch, Kmax_dict = load_dictionary(args.dict_npz)
    Kmax_global = min(max(K_list), Kmax_dict)

    man = pd.read_csv(os.path.expanduser(args.manifest))
    if man.empty:
        print("[warn] empty manifest", args.manifest)
        return

    # ---- group channels ----
    first_npz_path = man.iloc[0]["npz_path"]
    z0 = np.load(first_npz_path, allow_pickle=True)
    canon0 = [str(x) for x in z0["canonical_channels"].tolist()]
    mask_group = z0["subject_mask"].astype(bool).copy()

    for _, rec in man.iterrows():
        npz_path = rec.get("npz_path", None)
        if not isinstance(npz_path, str) or not os.path.isfile(npz_path):
            continue
        z = np.load(npz_path, allow_pickle=True)
        canon = [str(x) for x in z["canonical_channels"].tolist()]
        if canon != canon0:
            print("[warn] canonical order differs; using first subject order")
        mask_group &= z["subject_mask"].astype(bool)

    dict_idx_lower = {c.lower(): i for i, c in enumerate(dict_ch)}
    candidate = [c for c, m in zip(canon0, mask_group) if m and (c.lower() in dict_idx_lower)]

    pos_all = build_positions_std1005(candidate)
    ok = np.all(np.isfinite(pos_all), axis=1)

    group_channels = [c for c, good in zip(candidate, ok) if good]
    dict_rows = [dict_idx_lower[c.lower()] for c in group_channels]

    if len(group_channels) < 8:
        raise SystemExit(f"group channels too small ({len(group_channels)})")

    print("[info] group channels:", len(group_channels))

    D_group = D_full[dict_rows, :][:, :Kmax_global]
    M_group = D_group.shape[0]

    if "SPH" in args.methods:
        pos_group = pos_all[ok]
        B_sph_group = sph_design_degree_major(pos_group, Kmax_global)
    else:
        B_sph_group = None

    # ---- train PCA/ICA on Y_tfr_avg ----
    Y_blocks = []
    for _, rec in man.iterrows():
        sid = str(rec["subject"])
        task = str(rec["task"])
        npz_path = rec.get("npz_path", None)
        if not isinstance(npz_path, str) or not os.path.isfile(npz_path):
            continue

        z = np.load(npz_path, allow_pickle=True)
        times = z["times"].astype(float)
        freqs = z["freqs"].astype(float)
        conds = [str(x) for x in z["conditions"].tolist()]
        ch_used = [str(x) for x in z["channels"].tolist()]

        used_idx_lower = {c.lower(): i for i,  c in enumerate(ch_used)}
        try:
            cols = [used_idx_lower[c.lower()] for c in group_channels]
        except KeyError:
            continue

        t_comp = component_time_indices(task, times, args.eval_t_step)
        f_eval = np.arange(len(freqs))[::max(1, args.eval_f_step)]

        for li in range(len(conds)):
            arr = z["Y_tfr_avg"][li]
            arr = arr[t_comp][:, f_eval][:, :, cols]
            T_eval = arr.shape[0]
            F_eval_len = arr.shape[1]
            Ytmp = arr.reshape(T_eval * F_eval_len,  M_group)
            Ytmp = zscore_rows(Ytmp)
            Y_blocks.append(Ytmp)

    if not Y_blocks:
        raise SystemExit("no data for PCA/ICA training")

    Y_group = np.vstack(Y_blocks)
    print("[info] PCA/ICA training Y_group:",  Y_group.shape)

    Q_pca_group = pca_Q_full(Y_group)

    A_ica_group = None
    try:
        _, S, _ = np.linalg.svd(Y_group - Y_group.mean(axis=0, keepdims=True), full_matrices=False)
        r = int((S >= (1e-12 * (S[0] if S.size else 1.0))).sum())
        ncomp = max(1, min(max(K_list), r))
        A = ica_mixing(Y_group, ncomp=ncomp)
        if A is not None and A.size:
            A_ica_group = A
        else:
            print("[warn] ICA failed; will fallback to PCA for ICA")
    except Exception as e:
        print("[warn] ICA training failed:", e)
        A_ica_group = None

    rows_AUC, rows_THR = [], []

    # ---- span evaluation ----
    for _, rec in man.iterrows():
        sid = str(rec["subject"])
        task = str(rec["task"])
        npz_path = rec.get("npz_path", None)
        if not isinstance(npz_path, str) or not os.path.isfile(npz_path):
            continue

        z = np.load(npz_path, allow_pickle=True)
        times = z["times"].astype(float)
        freqs = z["freqs"].astype(float)
        conds = [str(x) for x in z["conditions"].tolist()]
        ch_used = [str(x) for x in z["channels"].tolist()]

        used_idx_lower = {c.lower(): i for i,  c in enumerate(ch_used)}
        try:
            cols = [used_idx_lower[c.lower()] for c in group_channels]
        except KeyError:
            continue

        t_comp = component_time_indices(task,  times, args.eval_t_step)
        f_eval = np.arange(len(freqs))[::max(1, args.eval_f_step)]

        for li, cond in enumerate(conds):
            mode = args.mode

            if mode == "trials":
                key = f"Y_tfr_trials__{cond}"
                if key not in z:
                    continue
                arr = z[key]
                arr = arr[:, t_comp][:, :, f_eval][:, :, :, cols]
                N, T_eval, F_eval_len, _ = arr.shape
                Y = zscore_rows(arr.reshape(N * T_eval * F_eval_len, M_group))
            elif mode == "bands":
                arr = z["Y_tfr_band_avg"][li]
                bands = [str(x) for x in z["bands"].tolist()]
                b_idx = np.arange(len(bands))[::max(1, args.eval_f_step)]
                arr = arr[t_comp][:, b_idx][:, :, cols]
                T_eval = arr.shape[0]
                B_eval_len = arr.shape[1]
                Y = zscore_rows(arr.reshape(T_eval * B_eval_len, M_group))
            else:
                arr = z["Y_tfr_avg"][li]
                arr = arr[t_comp][:, f_eval][:, :, cols]
                T_eval = arr.shape[0]
                F_eval_len = arr.shape[1]
                Y = zscore_rows(arr.reshape(T_eval * F_eval_len, M_group))

            n_rows = Y.shape[0]

            methods = []
            if "LB" in args.methods:
                methods.append(("LB", D_group, "native"))
            if "SPH" in args.methods and B_sph_group is not None:
                methods.append(("SPH", B_sph_group, "native"))
            if "PCA" in args.methods and Q_pca_group is not None and Q_pca_group.size:
                methods.append(("PCA", Q_pca_group, "orth"))
            if "ICA" in args.methods:
                if A_ica_group is not None and A_ica_group.size:
                    methods.append(("ICA", A_ica_group, "native"))
                elif Q_pca_group is not None and Q_pca_group.size:
                    methods.append(("ICA", Q_pca_group, "orth"))

            for mname, basis, basis_type in methods:
                basis = np.asarray(basis, float)
                Ks_eff = [k for k in K_list if k <= min(M_group, basis.shape[1])]
                if not Ks_eff:
                    continue

                auc_sum = {k: 0.0 for k in Ks_eff}
                auc_cnt = {k: 0 for k in Ks_eff}
                proj_mats = {}

                for start in range(0, n_rows, args.chunk_rows):
                    end = min(n_rows, start + args.chunk_rows)
                    Yb = Y[start:end, :]

                    for k in Ks_eff:
                        if args.prefix_qr:
                            Bk = basis[:, :k]
                            Qk = qr_no_pivot(Bk)
                            if Qk.size == 0:
                                continue
                            Yhat = (Yb @ Qk) @ Qk.T
                        else:
                            if basis_type == "orth":
                                Qk = basis[:, :k]
                                if Qk.size == 0:
                                    continue
                                Yhat = (Yb @ Qk) @ Qk.T
                            else:
                                if k not in proj_mats:
                                    Bk = basis[:, :k]
                                    BtB = Bk.T @ Bk
                                    BtB_inv = np.linalg.pinv(BtB)
                                    proj_mats[k] = Bk @ BtB_inv @ Bk.T
                                Pk = proj_mats[k]
                                Yhat = Yb @ Pk

                        r2b = r2_rows(Yb, Yhat)
                        auc_sum[k] += float(r2b.sum())
                        auc_cnt[k] += int(r2b.size)

                auc_vals = []
                for k in Ks_eff:
                    av = auc_sum[k] / max(1, auc_cnt[k])
                    auc_vals.append(av)
                    rows_AUC.append(dict(subject=sid, task=task,  condition=cond, method=mname, K=k, AUC=float(av)))

                v = np.array(auc_vals)
                idx70 = first_K_at_threshold(v, 0.70)
                idx90 = first_K_at_threshold(v, 0.90)
                K70 = Ks_eff[idx70 - 1] if idx70 == idx70 else np.nan
                K90 = Ks_eff[idx90 - 1] if idx90 == idx90 else np.nan

                rows_THR.append(
                    dict(
                        subject=sid, task=task, condition=cond, method=mname,
                        Kcap=int(M_group),
                        K70=float(K70) if K70 == K70 else np.nan,
                        K90=float(K90) if K90 == K90 else np.nan,
                        attain_70=int(K70 == K70),
                        attain_90=int(K90 == K90),
                    )
                )

    auc_df = pd.DataFrame(rows_AUC)
    thr_df = pd.DataFrame(rows_THR)

    auc_csv = os.path.join(tbl_dir, "AUC_curves.csv")
    thr_csv = os.path.join(tbl_dir, "K_thresh.csv")

    if not auc_df.empty:
        auc_df.sort_values(["task", "method", "subject", "condition", "K"]).to_csv(auc_csv, index=False)
    else:
        pd.DataFrame(columns=["subject", "task", "condition", "method", "K", "AUC"]).to_csv(auc_csv, index=False)

    if not thr_df.empty:
        thr_df.sort_values(["task", "method", "subject", "condition"]).to_csv(thr_csv, index=False)
    else:
        pd.DataFrame(
            columns=["subject", "task", "condition", "method", "Kcap",  "K70", "K90", "attain_70", "attain_90"]
        ).to_csv(thr_csv, index=False)

    print("\n[wrote]")
    print(" ", auc_csv, f"(rows={len(auc_df)})")
    print(" ", thr_csv, f"(rows={len(thr_df)})")

    if not auc_df.empty:
        for (sid, task, cond, method), g in auc_df.groupby(["subject", "task",   "condition", "method"]):
            g = g.sort_values("K")
            plt.figure()
            plt.plot(g["K"], g["AUC"], marker="o")
            plt.ylim(0.0, 1.0)
            plt.xlabel("K")
            plt.ylabel("AUC (mean R² over component window)")
            plt.title(f"sub-{sid} {task} {cond} {method}")

            fname = (
                f"AUC_sub-{safe_tag(sid)}_task-{safe_tag(task)}_cond-{safe_tag(cond)}_meth-{safe_tag(method)}.png"
            )
            fpath = os.path.join(fig_dir, fname)
            plt.tight_layout()
            plt.savefig(fpath, dpi=150)
            plt.close()

        print(" [figures] AUC plots ->", fig_dir)


if __name__ == "__main__":
    main()
