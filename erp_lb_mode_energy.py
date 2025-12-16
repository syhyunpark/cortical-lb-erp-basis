#!/usr/bin/env python3
"""
erp_lb_mode_energy.py
.
Compute LB/SPH mode energy from ERP-CORE TFR derivatives...

Sources: 
  trials : window-avg mode×freq energy from trial-level TFR
  avg    : window-avg mode×freq energy from trial-avg TFR
  peak   : peak-latency contrast (one timepoint)

Writes erp_core_mode_energy_lb_sph.npz
  
"""

import argparse
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import mne


COMP_WINDOWS = {
    "N170": (0.110, 0.150),
    "N2PC": (0.200, 0.275),
    "N400": (0.300, 0.500),
    "P3":   (0.300, 0.600),
    "ERN":  (0.000, 0.100),
    "LRP":  (-0.100, 0.000),
    "MMN":  (0.125, 0.225),
}

ERP_CONTRASTS = {
    "MMN":  ("stimulus/standard",       "stimulus/deviant"),
    "N170": ("stimulus/car/normal",     "stimulus/face/normal"),
    "N2pc": ("stimulus/left",           "stimulus/right"),   # we use right-left
    "N400": ("stimulus/target/related", "stimulus/target/unrelated"),
    "P3":   ("stimulus/non-target",     "stimulus/target"),
    "LRP":  ("response/left",           "response/right"),
    "ERN":  ("response/correct",        "response/incorrect"),
}


def zscore_rows(A, eps=1e-12):
    A = np.asarray(A, float)
    mu = A.mean(axis=1, keepdims=True)
    sd = A.std(axis=1, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (A - mu) / (sd + eps)


def load_lb_sph_dict(dict_path):
    d = np.load(os.path.expanduser(dict_path), allow_pickle=True)
    if "channels" not in d:
        raise KeyError(f"{dict_path} missing 'channels'")

    dict_ch = np.array(d["channels"]).astype(str)

    if "D_lb" in d:
        D_lb = np.asarray(d["D_lb"], float)
    elif "D" in d:
        D_lb = np.asarray(d["D"], float)
    else:
        raise KeyError(f"{dict_path} needs 'D_lb' or 'D'")

    if D_lb.shape[0] != len(dict_ch):
        raise ValueError("LB dict rows != len(channels)")

    B_sph = None
    if "B_sph" in d:
        B_sph = np.asarray(d["B_sph"], float)
        if B_sph.shape[0] != len(dict_ch):
            raise ValueError("SPH dict rows != len(channels)")

    return dict_ch, D_lb, B_sph


def get_time_mask(task, times):
    times = np.asarray(times, float)
    tname = str(task).upper()
    if tname in COMP_WINDOWS:
        t0, t1 = COMP_WINDOWS[tname]
        mask = (times >= t0) & (times <= t1)
        if not mask.any():
            mask[:] = True
        return mask
    return np.ones_like(times, dtype=bool)


def main():
    ap = argparse.ArgumentParser(description="Compute LB/SPH mode energy spectra from ERP-CORE TFR.")
    ap.add_argument("--tfr-root", required=True)
    ap.add_argument("--dict-npz", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--source", choices=["trials", "avg", "peak"],  default="trials")
    ap.add_argument("--fmin", type=float, default=None)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--max-k-lb", type=int, default=None)
    ap.add_argument("--max-k-sph", type=int, default=None)
    ap.add_argument("--per-subject-csv", action="store_true")
    args = ap.parse_args()

    tfr_root = Path(os.path.expanduser(args.tfr_root))
    outdir = Path(os.path.expanduser(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    dict_channels, D_lb_full, B_sph_full =  load_lb_sph_dict(args.dict_npz)
    C_full = len(dict_channels)

    if args.max_k_lb is not None:
        D_lb_full = D_lb_full[:, :args.max_k_lb]
    K_lb = D_lb_full.shape[1]

    K_sph = 0
    if B_sph_full is not None:
        K_sph_target = args.max_k_sph if args.max_k_sph is not None else K_lb
        B_sph_full = B_sph_full[:, :K_sph_target]
        K_sph = B_sph_full.shape[1]

    pat =  re.compile(r"^sub-(?P<sub>[^_]+)_task-(?P<task>[^_]+)_Y_tfr\.npz$")
    files_by_task = defaultdict(list)
    all_npz = sorted(tfr_root.rglob("*_Y_tfr.npz"))
    if not all_npz:
        raise SystemExit(f"no '*_Y_tfr.npz' under {tfr_root}")

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
        raise SystemExit(f"no subject×task files after filtering (tasks={args.tasks})")

    subj_rows = []

    for task, entries in sorted(files_by_task.items()):
        print(f"\n[info] task={task} n={len(entries)} source={args.source}")

        if args.source == "peak" and task not in ERP_CONTRASTS:
            print("  [skip] no ERP_CONTRASTS for this task")
            continue

        lb_power_sum = None   #   (F_sel,K_lb) accumulated across subjects
        sph_power_sum = None  #  (F_sel,K_sph)
        freqs_sel = None
        n_subj = 0

        for subj, npz_path in entries:
            ds = np.load(str(npz_path), allow_pickle=True)

            times = np.asarray(ds["times"], float)
            freqs = np.asarray(ds["freqs"], float)
            canonical = np.array(ds["canonical_channels"]).astype(str)
            subj_mask = np.asarray(ds["subject_mask"], bool)

            tmask = get_time_mask(task, times)

            fmask = np.ones_like(freqs, dtype=bool)
            if args.fmin is not None:
                fmask &= (freqs >= args.fmin)
            if args.fmax is not None:
                fmask &= (freqs <= args.fmax)
            if not fmask.any():
                print(f"  [skip] sub={subj} no freqs in [{args.fmin},{args.fmax}]")
                continue

            # map subject channel list (masked canonical) -> dict channels (case-insensitive)
            subj_idx_all = [i for i, m in enumerate(subj_mask) if m]
            subj_ch = [canonical[i] for i in subj_idx_all]

            subj_map = {ch.lower(): j for j, ch in enumerate(subj_ch)}
            dict_map = {ch.lower(): i for i, ch in enumerate(dict_channels)}

            common_pairs = []
            for ch_low, i_d in dict_map.items():
                j_s = subj_map.get(ch_low)
                if j_s is not None:
                    common_pairs.append((i_d, j_s))

            if len(common_pairs) < 8:
                print(f"  [skip] sub={subj} only {len(common_pairs)} common channels")
                continue

            dict_keep = [p[0] for p in common_pairs]
            subj_keep = [p[1] for p in common_pairs]

            D_lb_sub = D_lb_full[dict_keep, :]
            B_sph_sub = B_sph_full[dict_keep, :] if B_sph_full is not None else None
            M_common = D_lb_sub.shape[0]

            conds = [str(x) for x in ds["conditions"].tolist()]

            if args.source in ("trials", "avg"):
                if args.source == "trials":
                    Y_list = []
                    for k in ds.files:
                        if not k.startswith("Y_tfr_trials__"):
                            continue
                        Yk = ds[k]
                        if getattr(Yk, "ndim", 0) != 4:
                            continue
                        Y_list.append(Yk)

                    if not Y_list:
                        print(f"  [skip] sub={subj} no trial-level arrays")
                        continue

                    Y_all = np.concatenate(Y_list, axis=0)  # (N,T,F,M_subj)
                    Y_all = Y_all[:, tmask][:, :, fmask][:, :, :, subj_keep]
                    N, T_sel, F_sel, _ = Y_all.shape
                    Y = zscore_rows(Y_all.reshape(N * T_sel * F_sel, M_common))
                else:
                    Y_avgs = []
                    T_sel = None
                    F_sel = None
                    for li in range(len(conds)):
                        arr = ds["Y_tfr_avg"][li]  # (T,F,M_subj)
                        arr = arr[tmask][:, fmask][:, :, subj_keep]
                        T_sel, F_sel = arr.shape[0], arr.shape[1]
                        Y_avgs.append(zscore_rows(arr.reshape(T_sel * F_sel, M_common)))

                    if not Y_avgs or T_sel is None or F_sel is None:
                        print(f"  [skip] sub={subj} couldn't build avg arrays")
                        continue
                    Y = np.vstack(Y_avgs)

                Z_lb = Y @ D_lb_sub
                Z_sph = (Y @ B_sph_sub) if B_sph_sub is not None else None

                n_tf = T_sel * F_sel
                n_blocks = Z_lb.shape[0] // n_tf
                if n_blocks < 1:
                    print(f"  [skip] sub={subj} reshape fail")
                    continue

                Z_lb_tf = Z_lb.reshape(n_blocks, T_sel, F_sel, K_lb)
                P_lb = (Z_lb_tf ** 2).mean(axis=(0, 1))  # (F_sel,K_lb)

                P_sph = None
                if Z_sph is not None:
                    Z_sph_tf = Z_sph.reshape(n_blocks, T_sel, F_sel, K_sph)
                    P_sph = (Z_sph_tf ** 2).mean(axis=(0, 1))

                if lb_power_sum is None:
                    lb_power_sum = P_lb.copy()
                    sph_power_sum = P_sph.copy() if P_sph is not None else None
                    freqs_sel = freqs[fmask]
                else:
                    if P_lb.shape != lb_power_sum.shape:
                        raise ValueError(f"[{task}] LB shape mismatch {P_lb.shape} vs {lb_power_sum.shape}")
                    lb_power_sum += P_lb
                    if P_sph is not None:
                        if sph_power_sum is None:
                            sph_power_sum = P_sph.copy()
                        else:
                            if P_sph.shape != sph_power_sum.shape:
                                raise ValueError(f"[{task}] SPH shape mismatch {P_sph.shape} vs {sph_power_sum.shape}")
                            sph_power_sum += P_sph

                n_subj += 1

                if args.per_subject_csv:
                    E_lb = np.sum(P_lb, axis=0)
                    for k in range(K_lb):
                        subj_rows.append(dict(subject=subj, task=task, method="LB", mode_index=k + 1, energy=float(E_lb[k])))
                    if P_sph is not None:
                        E_sph = np.sum(P_sph, axis=0)
                        for k in range(K_sph):
                            subj_rows.append(dict(subject=subj, task=task, method="SPH", mode_index=k + 1, energy=float(E_sph[k])))

            else:
                # peak
                cond0, cond1 = ERP_CONTRASTS[task]
                try:
                    if task == "N2pc":
                        i0 = conds.index(cond0)
                        i1 = conds.index(cond1)
                        X = ds["Y_tfr_avg"][i1] - ds["Y_tfr_avg"][i0]  # right-left
                    else:
                        i0 = conds.index(cond0)
                        i1 = conds.index(cond1)
                        X = ds["Y_tfr_avg"][i1] - ds["Y_tfr_avg"][i0]
                except ValueError:
                    print(f"  [skip] sub={subj} missing contrast conds")
                    continue

                X_win = X[tmask][:, fmask][:, :, subj_keep]  # (T_sel,F_sel,M_common)
                T_sel, F_sel = X_win.shape[0], X_win.shape[1]
                if T_sel < 1 or F_sel < 1:
                    print(f"  [skip] sub={subj} empty TF window")
                    continue

                energy_t = np.sum(X_win ** 2, axis=(1, 2))
                t_peak = int(np.argmax(energy_t))

                X_peak = X_win[t_peak]    #    (F_sel,M_common)
                X_peak_z = zscore_rows(X_peak)

                C_lb = X_peak_z @ D_lb_sub
                C_sph = (X_peak_z @ B_sph_sub) if B_sph_sub is not None else None

                P_lb = (C_lb ** 2)
                P_sph = (C_sph ** 2) if C_sph is not None else None

                if lb_power_sum is None:
                    lb_power_sum = P_lb.copy()
                    sph_power_sum = P_sph.copy() if P_sph is not None else None
                    freqs_sel = freqs[fmask]
                else:
                    if P_lb.shape != lb_power_sum.shape:
                        raise ValueError(f"[{task}] LB shape mismatch {P_lb.shape} vs {lb_power_sum.shape}")
                    lb_power_sum += P_lb
                    if P_sph is not None:
                        if sph_power_sum is None:
                            sph_power_sum = P_sph.copy()
                        else:
                            if P_sph.shape != sph_power_sum.shape:
                                raise ValueError(f"[{task}] SPH shape mismatch {P_sph.shape} vs {sph_power_sum.shape}")
                            sph_power_sum += P_sph

                n_subj += 1

                if args.per_subject_csv:
                    E_lb = np.sum(P_lb, axis=0)
                    for k in range(K_lb):
                        subj_rows.append(dict(subject=subj, task=task, method="LB", mode_index=k + 1, energy=float(E_lb[k])))
                    if P_sph is not None:
                        E_sph = np.sum(P_sph, axis=0)
                        for k in range(K_sph):
                            subj_rows.append(dict(subject=subj, task=task, method="SPH", mode_index=k + 1, energy=float(E_sph[k])))

        if n_subj == 0 or lb_power_sum is None:
            print(f"[warn] task={task} no usable subjects")
            continue

        lb_power_group = (lb_power_sum / float(n_subj)).T  # (K_lb,F_sel)
        if sph_power_sum is not None:
            sph_power_group = (sph_power_sum / float(n_subj)).T
        else:
            sph_power_group = np.zeros((0, lb_power_group.shape[1]), float)

        lb_mode_energy = lb_power_group.mean(axis=1)
        sph_mode_energy = sph_power_group.mean(axis=1) if sph_power_group.size else np.zeros(0, float)

        out_path = outdir / f"erp_core_{task}_mode_energy_lb_sph.npz"
        np.savez_compressed(
            out_path,
            task=np.array(task),
            freqs=freqs_sel.astype(float),
            lb_power=lb_power_group.astype(np.float32),
            sph_power=sph_power_group.astype(np.float32),
            lb_mode_energy=lb_mode_energy.astype(np.float32),
            sph_mode_energy=sph_mode_energy.astype(np.float32),
            lb_patterns=D_lb_full.T.astype(np.float32),  #  (K_lb,   C_full)
            sph_patterns=(B_sph_full.T.astype(np.float32) if B_sph_full is not None else np.empty((0, C_full), np.float32)),
            dict_channels=np.array(dict_channels, dtype=object),
        )
        print(f"[info] wrote {out_path} (n_subj={n_subj})")

    if args.per_subject_csv and subj_rows:
        import pandas as pd
        subj_csv = outdir / f"mode_energy_subject_{args.source}.csv"
        pd.DataFrame(subj_rows).to_csv(subj_csv, index=False)
        print(f"[info] wrote {subj_csv} (rows={len(subj_rows)})")


if __name__ == "__main__":
    main()
