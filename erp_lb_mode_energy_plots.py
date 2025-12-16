#!/usr/bin/env python3
"""
erp_lb_mode_energy_plots.py
 
"""

import os
import re
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import mne


def safe_tag(s):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode-energy-dir", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--top-n-modes", type=int, default=3)
    args = ap.parse_args()

    mode_dir = os.path.expanduser(args.mode_energy_dir)
    outdir = os.path.expanduser(args.outdir) if args.outdir else mode_dir
    os.makedirs(outdir, exist_ok=True)

    pat = os.path.join(mode_dir, "erp_core_*_mode_energy_lb_sph.npz")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise SystemExit(f"no files matching {pat}")

    files = []
    for p in paths:
        base = os.path.basename(p)
        m = re.match(r"^erp_core_(.+)_mode_energy_lb_sph\.npz$", base)
        if not m:
            continue
        task = m.group(1)
        if args.tasks and task not in set(args.tasks):
            continue
        files.append((task, p))

    if not files:
        raise SystemExit("no tasks left after filtering")

    print("[info] tasks:", ", ".join([t for t, _ in files]))

    for task, path in files:
        print("\n---", task, "---")
        ds = np.load(path, allow_pickle=True)

        freqs = np.asarray(ds["freqs"], float)
        lb_power = np.asarray(ds["lb_power"], float)
        sph_power = np.asarray(ds["sph_power"], float)
        lb_mode_energy = np.asarray(ds["lb_mode_energy"], float)
        sph_mode_energy = np.asarray(ds["sph_mode_energy"], float)
        lb_patterns_raw = np.asarray(ds["lb_patterns"], float)
        dict_channels = [str(x) for x in ds["dict_channels"].tolist()]

        K_lb, F = lb_power.shape
        K_sph, F2 = sph_power.shape
        if F2 != F:
            raise ValueError(f"[{task}] freq dim mismatch: LB={F}, SPH={F2}")

        C_full = len(dict_channels)
        if lb_patterns_raw.shape == (K_lb, C_full):
            lb_patterns = lb_patterns_raw
        elif lb_patterns_raw.shape == (C_full, K_lb):
            lb_patterns = lb_patterns_raw.T
        else:
            raise ValueError(f"[{task}] weird lb_patterns shape: {lb_patterns_raw.shape}")

        # ---- heatmaps ----
        lb_power_norm = lb_power / (lb_power.max() + 1e-12)
        sph_power_norm = sph_power / (sph_power.max() + 1e-12)

        plt.figure(figsize=(6, 4))
        im = plt.imshow(
            lb_power_norm,
            aspect="auto",
            origin="lower",
            extent=[freqs[0], freqs[-1], 1, K_lb],
            vmin=0.0,
            vmax=1.0,
        )
        plt.colorbar(im, label="Normalized mode×frequency power")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("LB mode index $k$")
        plt.title(f"{task}: LB mode×frequency power")
        plt.tight_layout()
        fname = os.path.join(outdir, f"lb_power_{safe_tag(task)}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print("  wrote", os.path.basename(fname))

        plt.figure(figsize=(6, 4))
        im = plt.imshow(
            sph_power_norm,
            aspect="auto",
            origin="lower",
            extent=[freqs[0], freqs[-1], 1, K_sph],
            vmin=0.0,
            vmax=1.0,
        )
        plt.colorbar(im, label="Normalized mode×frequency power")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("SPH mode index $k$")
        plt.title(f"{task}: SPH mode×frequency power")
        plt.tight_layout()
        fname = os.path.join(outdir, f"sph_power_{safe_tag(task)}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print("  wrote", os.path.basename(fname))

        # ---- top LB topos ----
        n_show = min(args.top_n_modes, K_lb)
        order = np.argsort(-lb_mode_energy)
        top_modes = order[:n_show]
        print("  top LB modes:", top_modes.tolist())

        info = mne.create_info(ch_names=list(dict_channels), sfreq=250.0, ch_types="eeg")
        montage = mne.channels.make_standard_montage("standard_1005")
        info.set_montage(montage, on_missing="ignore")

        fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 3))
        if n_show == 1:
            axes = [axes]

        for ax, idx in zip(axes, top_modes):
            x = lb_patterns[idx, :].astype(float)
            x = x - x.mean()
            vmax = np.max(np.abs(x))
            if vmax == 0:
                vmax = 1.0

            mne.viz.plot_topomap(
                x,
                info,
                axes=ax,
                show=False,
                vlim=(-vmax, vmax),
                contours=6,
            )
            ax.set_title(f"LB mode {idx+1}\nenergy={lb_mode_energy[idx]:.3f}")

        fig.suptitle(f"{task}: top {n_show} LB modes by energy", y=0.95)
        plt.tight_layout()
        fname = os.path.join(outdir, f"lb_topos_{safe_tag(task)}.png")
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        print("  wrote", os.path.basename(fname))

        # ---- cumulative energy ----
        lb_frac = lb_mode_energy / (lb_mode_energy.sum() + 1e-12)
        sph_frac = sph_mode_energy / (sph_mode_energy.sum() + 1e-12)

        K_len = min(25, len(lb_frac), len(sph_frac))
        k_axis = np.arange(1, K_len + 1)

        lb_cum = np.cumsum(lb_frac[:K_len])
        sph_cum = np.cumsum(sph_frac[:K_len])

        plt.figure(figsize=(5, 3))
        plt.plot(k_axis, lb_cum, marker="o", label="LB")
        plt.plot(k_axis, sph_cum, marker="s", label="SPH")
        plt.axhline(0.7, color="gray", linestyle="--", linewidth=1)
        plt.axhline(0.9, color="gray", linestyle="--", linewidth=1)
        plt.ylim(0.0, 1.01)
        plt.xlabel("Mode index $k$")
        plt.ylabel("Cumulative normalized energy")
        plt.title(f"{task}: cumulative mode energy")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        fname = os.path.join(outdir, f"mode_energy_cum_{safe_tag(task)}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print("  wrote", os.path.basename(fname))

    print("\n[done] plots ->", outdir)


if __name__ == "__main__":
    main()
