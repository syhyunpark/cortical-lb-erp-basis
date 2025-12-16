#!/usr/bin/env python3
"""
make_lb_sph_dict.py

Take an LB-only dict NPZ (keys: D, channels) and write a combined LB+SPH NPZ:
  channels, D_lb, Q_sph (QR-orth), B_sph (native degree-major, no DC)
"""

import argparse
import os
import numpy as np
import mne


def qr_no_pivot(B):
    if B.size == 0:
        return np.empty((B.shape[0], 0))
    Q, R = np.linalg.qr(B, mode="reduced")
    if R.size == 0:
        return np.empty((B.shape[0], 0))
    d = np.abs(np.diag(R))
    keep = d > (1e-12 * max(1.0, d.max()))
    return Q[:, keep] if keep.any() else np.empty((B.shape[0], 0))


def build_positions_std1005(ch_names):
    info = mne.create_info(ch_names=list(ch_names), sfreq=250.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
    pos = info.get_montage().get_positions()["ch_pos"]
    P = np.array([pos.get(ch, np.array([np.nan, np.nan, np.nan])) for ch in ch_names], float)
    nrm = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    return P / nrm


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


def main():
    ap = argparse.ArgumentParser(description="Make LB+SPH dict NPZ from LB-only dict NPZ")
    ap.add_argument("--lb-dict-npz", required=True, help="LB dict NPZ with keys D, channels")
    ap.add_argument("--out-npz", required=True, help="Output NPZ")
    ap.add_argument("--max-k-lb", type=int, default=None)
    ap.add_argument("--max-k-sph", type=int, default=None)
    args = ap.parse_args()

    lb_path = os.path.expanduser(args.lb_dict_npz)
    d = np.load(lb_path, allow_pickle=True)
    if ("D" not in d) or ("channels" not in d):
        raise KeyError(f"{lb_path} must contain D and channels")

    D_lb = np.asarray(d["D"], float)
    channels = [str(x) for x in d["channels"].tolist()]

    if args.max_k_lb is not None:
        D_lb = D_lb[:, :args.max_k_lb]

    M, K_lb = D_lb.shape

    pos = build_positions_std1005(channels)
    K_sph_target = args.max_k_sph if args.max_k_sph is not None else K_lb
    B_sph = sph_design_degree_major(pos, K_sph_target)
    Q_sph = qr_no_pivot(B_sph)

    out_path = os.path.expanduser(args.out_npz)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        out_path,
        channels=np.array(channels, dtype=object),
        D_lb=D_lb.astype(np.float32),
        Q_sph=Q_sph.astype(np.float32),
        B_sph=B_sph.astype(np.float32),
    )

    print("[info] wrote", out_path)
    print("       channels:", M, "K_lb:", K_lb, "K_sph:", int(Q_sph.shape[1]))


if __name__ == "__main__":
    main()
