#!/usr/bin/env python3
import os, glob, argparse, collections
import numpy as np
import mne


def std1005_map():
    mon = mne.channels.make_standard_montage("standard_1005")
    return {nm.lower(): nm for nm in mon.ch_names}


def map_to_std1005_case(names, lut):
    out = []
    for n in names:
        out.append(lut.get(str(n).lower(), str(n)))
    return out


def sort_std_order(chs):
    info = mne.create_info(ch_names=list(chs), sfreq=250., ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
    pos = info.get_montage().get_positions()["ch_pos"]

    def key(c):
        xyz = pos.get(c, np.array([0.0, 0.0, 0.0]))
        return (-xyz[2], -xyz[1], -xyz[0], c)

    return sorted(chs, key=key)


def main():
    ap = argparse.ArgumentParser(description="Derive canonical_30.txt from Stage-1 NPZ files")
    ap.add_argument("--npz-root", required=True, help=".../derivatives/Y_erp_core")
    ap.add_argument("--out", default="canonical_30.txt")
    ap.add_argument("--min-presence", type=float, default=1.0,
                    help="fraction of files a channel must appear in (default 1.0)")
    args = ap.parse_args()

    npz_root = os.path.expanduser(args.npz_root)
    files = sorted(glob.glob(os.path.join(npz_root, "sub-*", "sub-*_task-*_Y_erp.npz")))
    if not files:
        raise SystemExit(f"no NPZs under {npz_root}")

    lut = std1005_map()
    counts = collections.Counter()
    file_count = 0

    eog_like = {"HEOG", "VEOG", "EOG", "ECG"}

    for f in files:
        d = np.load(f, allow_pickle=True)
        ch = [str(x) for x in d["channels"].tolist()]
        ch = map_to_std1005_case(ch, lut)
        ch = [c for c in ch if c not in eog_like]
        s = set(ch)
        for c in s:
            counts[c] += 1
        file_count += 1

    thresh = int(np.ceil(args.min_presence * file_count))
    kept = [c for c, n in counts.items() if n >= thresh]

    if len(kept) < 20 and args.min_presence == 1.0:
        thresh = int(np.ceil(0.95 * file_count))
        kept = [c for c, n in counts.items() if n >= thresh]

    kept = sort_std_order(kept)

    out_path = os.path.expanduser(args.out)
    with open(out_path, "w") as fo:
        for c in kept:
            fo.write(c + "\n")

    print(f"[canonical] wrote {len(kept)} channels -> {os.path.abspath(out_path)}")
    print("preview:", kept[:10], "...")

if __name__ == "__main__":
    main()
