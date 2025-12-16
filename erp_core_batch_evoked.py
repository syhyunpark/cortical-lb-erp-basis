#!/usr/bin/env python
# erp_core_batch_evoked.py

"""
Batch ERP-CORE ERP/TFR extractor...

Outputs per subject/task:
- trial-averaged TFR (per condition) + band-averaged summaries
- optionally trial-level TFR arrays
- manifest.csv under derivatives/Y_tfr_core/  
"""


import os
import glob
import argparse
import zipfile
import textwrap
import re

import numpy as np
import pandas as pd
import requests
import mne
from mne_bids import BIDSPath, read_raw_bids


ERP_CORE_BIDS_OSF_URL = "https://osf.io/3zk6n/download?version=2"


def http_download(url, dst_path, chunk=2**20):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return dst_path
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    f.write(c)
    return dst_path


def find_bids_root(root_dir):
    for this_root, _, files in os.walk(root_dir):
        if "dataset_description.json" in files:
            return this_root
    raise RuntimeError(f"dataset_description.json not found under {root_dir}")


def ensure_erp_core_bids(bids_root):
    try:
        return find_bids_root(bids_root)
    except RuntimeError:
        pass

    print(f"[info] No BIDS dataset under {bids_root}. Downloading ERP-CORE BIDS...")
    zip_path = os.path.join(bids_root, "erp_core_bids_osf_3zk6n.zip")
    http_download(ERP_CORE_BIDS_OSF_URL, zip_path)

    print(f"[info] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(bids_root)

    bids_root_found = find_bids_root(bids_root)
    print(f"[info] Found BIDS root at: {bids_root_found}")
    return bids_root_found


def load_canonical_list(path):
    with open(path, "r") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    seen = set()
    out = []
    for ch in names:
        if ch not in seen:
            out.append(ch)
            seen.add(ch)
    return out


def make_subject_mask(canonical_list, subject_ch_names):
    subj_list = list(subject_ch_names)
    subj_set = set(subj_list)
    mask = np.array([ch in subj_set for ch in canonical_list], dtype=bool)
    idx = [subj_list.index(ch) for ch in np.array(canonical_list)[mask]]
    return mask, idx


def zscore_rows(Y):
    Y = np.asarray(Y, float)
    mu = Y.mean(axis=1, keepdims=True)
    sd = Y.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (Y - mu) / sd


def get_task_config(task):
    task = task.upper()

    if task == "MMN":
        rename_events = {"stimulus/70": "stimulus/deviant", "stimulus/80": "stimulus/standard"}
        return dict(
            session="MMN",
            task="MMN",
            rename_events=rename_events,
            eeg_reference=["P9", "P10"],
            tmin=-0.2,
            tmax=0.8,
            baseline_erp=(None, 0.0),
            tfr_baseline=(-0.2, 0.0),
            conditions={"stimulus/standard": "stimulus/standard", "stimulus/deviant": "stimulus/deviant"},
            is_response_locked=False,
        )

    if task == "N170":
        rename_events = {"response/201": "response/correct", "response/202": "response/error"}
        for i in range(1, 181):
            if 1 <= i <= 40:
                rename_events[f"stimulus/{i}"] = "stimulus/face/normal"
            elif 41 <= i <= 80:
                rename_events[f"stimulus/{i}"] = "stimulus/car/normal"
            elif 101 <= i <= 140:
                rename_events[f"stimulus/{i}"] = "stimulus/face/scrambled"
            elif 141 <= i <= 180:
                rename_events[f"stimulus/{i}"] = "stimulus/car/scrambled"
        return dict(
            session="N170",
            task="N170",
            rename_events=rename_events,
            eeg_reference="average",
            tmin=-0.2,
            tmax=0.8,
            baseline_erp=(None, 0.0),
            tfr_baseline=(-0.2, 0.0),
            conditions={"stimulus/face/normal": "stimulus/face/normal", "stimulus/car/normal": "stimulus/car/normal"},
            is_response_locked=False,
        )

    if task == "N2PC":
        rename_events = {
            "response/201": "response/correct",
            "response/202": "response/error",
            "stimulus/111": "stimulus/left",
            "stimulus/112": "stimulus/left",
            "stimulus/121": "stimulus/right",
            "stimulus/122": "stimulus/right",
            "stimulus/211": "stimulus/left",
            "stimulus/212": "stimulus/left",
            "stimulus/221": "stimulus/right",
            "stimulus/222": "stimulus/right",
        }
        return dict(
            session="N2pc",
            task="N2pc",
            rename_events=rename_events,
            eeg_reference=["P9", "P10"],
            tmin=-0.2,
            tmax=0.8,
            baseline_erp=(None, 0.0),
            tfr_baseline=(-0.2, 0.0),
            conditions={"stimulus/left": "stimulus/left", "stimulus/right": "stimulus/right"},
            is_response_locked=False,
        )

    if task == "N400":
        rename_events = {
            "response/201": "response/correct",
            "response/202": "response/error",
            "stimulus/111": "stimulus/prime/related",
            "stimulus/112": "stimulus/prime/related",
            "stimulus/121": "stimulus/prime/unrelated",
            "stimulus/122": "stimulus/prime/unrelated",
            "stimulus/211": "stimulus/target/related",
            "stimulus/212": "stimulus/target/related",
            "stimulus/221": "stimulus/target/unrelated",
            "stimulus/222": "stimulus/target/unrelated",
        }
        return dict(
            session="N400",
            task="N400",
            rename_events=rename_events,
            eeg_reference=["P9", "P10"],
            tmin=-0.2,
            tmax=0.8,
            baseline_erp=(None, 0.0),
            tfr_baseline=(-0.2, 0.0),
            conditions={
                "stimulus/target/related": "stimulus/target/related",
                "stimulus/target/unrelated": "stimulus/target/unrelated",
            },
            is_response_locked=False,
        )

    if task == "P3":
        rename_events = {
            "response/201": "response/correct",
            "response/202": "response/incorrect",
            "stimulus/11": "stimulus/target/11",
            "stimulus/22": "stimulus/target/22",
            "stimulus/33": "stimulus/target/33",
            "stimulus/44": "stimulus/target/44",
            "stimulus/55": "stimulus/target/55",
            "stimulus/21": "stimulus/non-target/21",
            "stimulus/31": "stimulus/non-target/31",
            "stimulus/41": "stimulus/non-target/41",
            "stimulus/51": "stimulus/non-target/51",
            "stimulus/12": "stimulus/non-target/12",
            "stimulus/32": "stimulus/non-target/32",
            "stimulus/42": "stimulus/non-target/42",
            "stimulus/52": "stimulus/non-target/52",
            "stimulus/13": "stimulus/non-target/13",
            "stimulus/23": "stimulus/non-target/23",
            "stimulus/43": "stimulus/non-target/43",
            "stimulus/53": "stimulus/non-target/53",
            "stimulus/14": "stimulus/non-target/14",
            "stimulus/24": "stimulus/non-target/24",
            "stimulus/34": "stimulus/non-target/34",
            "stimulus/54": "stimulus/non-target/54",
            "stimulus/15": "stimulus/non-target/15",
            "stimulus/25": "stimulus/non-target/25",
            "stimulus/35": "stimulus/non-target/35",
            "stimulus/45": "stimulus/non-target/45",
        }
        return dict(
            session="P3",
            task="P3",
            rename_events=rename_events,
            eeg_reference=["P9", "P10"],
            tmin=-0.2,
            tmax=0.8,
            baseline_erp=(None, 0.0),
            tfr_baseline=(-0.2, 0.0),
            conditions={"stimulus/target": "stimulus/target", "stimulus/non-target": "stimulus/non-target"},
            is_response_locked=False,
        )

    if task == "LRP":
        rename_events = {
            "stimulus/11": "compatible/left",
            "stimulus/12": "compatible/right",
            "stimulus/21": "incompatible/left",
            "stimulus/22": "incompatible/right",
            "response/111": "response/left/correct",
            "response/112": "response/left/incorrect",
            "response/121": "response/left/correct",
            "response/122": "response/left/incorrect",
            "response/211": "response/right/incorrect",
            "response/212": "response/right/correct",
            "response/221": "response/right/incorrect",
            "response/222": "response/right/correct",
        }
        return dict(
            session="LRP",
            task="LRP",
            rename_events=rename_events,
            eeg_reference=["P9", "P10"],
            tmin=-0.8,
            tmax=0.2,
            baseline_erp=(None, -0.6),
            tfr_baseline=(-0.4, -0.2),
            conditions={"response/left": "response/left", "response/right": "response/right"},
            is_response_locked=True,
        )

    if task == "ERN":
        rename_events = {
            "stimulus/11": "compatible/left",
            "stimulus/12": "compatible/right",
            "stimulus/21": "incompatible/left",
            "stimulus/22": "incompatible/right",
            "response/111": "response/correct",
            "response/112": "response/incorrect",
            "response/121": "response/correct",
            "response/122": "response/incorrect",
            "response/211": "response/incorrect",
            "response/212": "response/correct",
            "response/221": "response/incorrect",
            "response/222": "response/correct",
        }
        return dict(
            session="ERN",
            task="ERN",
            rename_events=rename_events,
            eeg_reference=["P9", "P10"],
            tmin=-0.6,
            tmax=0.4,
            baseline_erp=(-0.4, -0.2),
            tfr_baseline=(-0.4, -0.2),
            conditions={"response/correct": "response/correct", "response/incorrect": "response/incorrect"},
            is_response_locked=True,
        )

    raise ValueError(f"Unsupported task {task!r}")


def rename_annotations(raw, rename_events):
    anns = raw.annotations
    if anns is None or len(anns) == 0:
        return
    new_desc = [rename_events.get(d, d) for d in anns.description]
    raw.set_annotations(
        mne.Annotations(onset=anns.onset, duration=anns.duration, description=new_desc, orig_time=anns.orig_time)
    )


def find_subjects(bids_root):
    subs = []
    for p in glob.glob(os.path.join(bids_root, "sub-*")):
        if os.path.isdir(p):
            subs.append(os.path.basename(p).split("-")[1])
    return sorted(subs)


def read_raw_for_task(bids_root, subject, session, task):
    bids_path = BIDSPath(
        root=bids_root, subject=subject, session=session, task=task, datatype="eeg", suffix="eeg", extension=".set"
    )
    if not bids_path.fpath.exists():
        raise FileNotFoundError(f"No EEG file found for {bids_path}")
    return read_raw_bids(bids_path, verbose="ERROR")


def _make_bipolar_eog(raw):
    chs = set(raw.ch_names)
    try:
        if {"HEOG_left", "HEOG_right"}.issubset(chs):
            mne.set_bipolar_reference(
                raw, anode="HEOG_left", cathode="HEOG_right", ch_name="HEOG", drop_refs=False, copy=False
            )
        if {"VEOG_lower", "FP2"}.issubset(chs):
            mne.set_bipolar_reference(
                raw, anode="VEOG_lower", cathode="FP2", ch_name="VEOG", drop_refs=False, copy=False
            )
        drop = [c for c in ("HEOG_left", "HEOG_right", "VEOG_lower") if c in raw.ch_names]
        if drop:
            raw.drop_channels(drop)
        set_types = {c: "eog" for c in ("HEOG", "VEOG") if c in raw.ch_names}
        if set_types:
            raw.set_channel_types(set_types)
    except Exception as e:
        print(f"[warn] EOG bipolar creation skipped: {e}")


def _run_ica_cleaning(raw, ica_reject, ica_decim=2, eog_threshold=2.0, random_state=97):
    from mne.preprocessing import ICA

    ica = ICA(max_iter=1000, random_state=random_state)
    raw_for_ica = raw.copy().filter(1.0, None, fir_design="firwin", verbose=False)
    ica.fit(raw_for_ica, decim=ica_decim, reject=ica_reject)

    exclude = []
    for eog_ch in ("VEOG", "HEOG"):
        if eog_ch in raw.ch_names:
            inds, _ = ica.find_bads_eog(raw, ch_name=eog_ch, threshold=eog_threshold)
            exclude.extend(inds)

    ica.exclude = sorted(set(exclude))
    if ica.exclude:
        print(f"  [info] Excluding ICA components: {ica.exclude}")
        ica.apply(raw)
    else:
        print("  [info] No ICA components marked for exclusion.")
    return raw


def _build_code_to_label(event_id):
    return {code: label for label, code in event_id.items()}


def _extract_event_table(events, code_to_label, sfreq):
    onsets = events[:, 0] / sfreq
    codes = events[:, 2]
    labels = [code_to_label.get(c, f"code/{c}") for c in codes]
    return pd.DataFrame({"onset": onsets, "code": codes, "label": labels})


def _mark_n400_correct_trials(
    epochs,
    events,
    event_id,
    target_prefixes,
    resp_correct_prefix="response/correct",
    resp_error_prefix="response/error",
    max_latency=1.0,
):
    sfreq = epochs.info["sfreq"]
    code_to_label = _build_code_to_label(event_id)
    df = _extract_event_table(events, code_to_label, sfreq)
    df_resp = df[df["label"].str.startswith("response/")].copy()

    meta_rows = []
    for (ev_samp, _, ev_code) in epochs.events:
        t0 = ev_samp / sfreq
        epoch_label = code_to_label.get(ev_code, "")
        is_target = any(epoch_label.startswith(pref) for pref in target_prefixes)

        if not is_target:
            meta_rows.append(dict(is_target=False, correct=None))
            continue

        df_win = df_resp[(df_resp["onset"] >= t0) & (df_resp["onset"] <= t0 + max_latency)]
        if df_win.empty:
            meta_rows.append(dict(is_target=True, correct=False))
            continue

        if (df_win["label"].str.startswith(resp_correct_prefix)).any():
            meta_rows.append(dict(is_target=True, correct=True))
        elif (df_win["label"].str.startswith(resp_error_prefix)).any():
            meta_rows.append(dict(is_target=True, correct=False))
        else:
            meta_rows.append(dict(is_target=True, correct=False))

    epochs.metadata = pd.DataFrame(meta_rows)
    return epochs


def compute_tfr_trials(epochs, freqs, cycles_rule, decim, baseline, db=True, safety=0.90):
    sfreq = float(epochs.info["sfreq"])
    n_times = int(epochs.get_data(copy=False).shape[-1])

    if cycles_rule == "const":
        base = np.full_like(freqs, 7.0, dtype=float)
    else:
        base = np.clip(freqs / 2.0, 3.0, 10.0)

    n_cycles_max = safety * n_times * (np.pi / 7.0) * (freqs / sfreq)
    n_cycles = np.minimum(base, n_cycles_max)
    n_cycles = np.maximum(n_cycles - 1e-6, 0.1)

    tfr = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        return_itc=False,
        average=False,
        decim=decim,
        picks="eeg",
        use_fft=True,
        verbose=False,
    )

    tfr.apply_baseline(baseline=baseline, mode="logratio")
    data = tfr.data
    if db:
        data *= 10.0
    return data, tfr.times


def row_zscore_tf_topos(arr_tf_ch):
    arr_tf_ch = np.asarray(arr_tf_ch, float)
    n_trials, T, F, M = arr_tf_ch.shape
    X = arr_tf_ch.reshape(-1, M)
    Xzs = zscore_rows(X)
    return Xzs.reshape(n_trials, T, F, M)


def process_subject_task(
    subject,
    task,
    bids_root,
    out_root,
    canonical_list,
    decim=1,
    l_freq=0.1,
    h_freq=None,
    notch_freq=60.0,
    raw_resample=128.0,
    use_autoreject=True,
    n400_correct_only=False,
    n400_max_latency=1.0,
    tfr_fmin=2.0,
    tfr_fmax=30.0,
    tfr_n_freqs=25,
    tfr_cycles="constQ",
    tfr_decim=2,
    band_defs="theta:4-8,alpha:8-13,beta:13-30",
    save_trials=True,
):
    cfg = get_task_config(task)
    session = cfg["session"]
    task_name = cfg["task"]

    print(f"  [subject {subject}] [{task_name}] reading raw...")
    raw = read_raw_for_task(bids_root, subject, session, task_name)
    raw.load_data()

    picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=False, stim=True, misc=True)
    raw.pick(picks)

    if raw.info.get("dig", None) is None:
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore")

    _make_bipolar_eog(raw)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    if notch_freq and notch_freq > 0:
        raw.notch_filter(freqs=[notch_freq], verbose=False)
    if raw_resample and raw_resample > 0:
        raw.resample(raw_resample)

    want_ref = cfg["eeg_reference"]
    if isinstance(want_ref, (list, tuple)):
        if all(ch in raw.ch_names for ch in want_ref):
            raw.set_eeg_reference(want_ref, verbose=False)
        else:
            print(f"  [info] Reference {want_ref} missing; using average.")
            raw.set_eeg_reference("average", verbose=False)
    else:
        raw.set_eeg_reference(want_ref, verbose=False)

    ica_reject = dict(eeg=350e-6, eog=500e-6)
    raw = _run_ica_cleaning(raw, ica_reject=ica_reject, ica_decim=2, eog_threshold=2.0)

    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, stim=False, misc=False)
    raw_eeg = raw.copy().pick(picks_eeg)

    subj_ch = list(raw_eeg.ch_names)
    mask, idx = make_subject_mask(canonical_list, subj_ch)
    if mask.sum() == 0:
        raise RuntimeError(f"No overlap between canonical and subject {subject} channels.")

    raw_eeg.pick(idx)
    raw_eeg.reorder_channels(list(np.array(canonical_list)[mask]))
    used_channels = list(raw_eeg.ch_names)

    rename_annotations(raw, cfg["rename_events"])
    events, event_id = mne.events_from_annotations(raw, event_id=None)

    epochs_all = mne.Epochs(
        raw_eeg,
        events,
        event_id=event_id,
        tmin=cfg["tmin"],
        tmax=cfg["tmax"],
        baseline=cfg["baseline_erp"],
        preload=True,
        detrend=None,
    )

    if use_autoreject:
        try:
            from autoreject import AutoReject

            ar = AutoReject(method="bayesian_optimization", random_state=97, verbose=False)
            epochs_all = ar.fit_transform(epochs_all)
            print("  [info] Applied autoreject (global).")
        except Exception as e:
            print(f"  [warn] autoreject unavailable ({e}); using amplitude threshold.")
            epochs_all.drop_bad(reject=dict(eeg=150e-6))
    else:
        epochs_all.drop_bad(reject=dict(eeg=150e-6))

    if task_name == "N400" and n400_correct_only:
        target_prefixes = ["stimulus/target/related", "stimulus/target/unrelated"]
        epochs_all = _mark_n400_correct_trials(
            epochs_all,
            events,
            event_id,
            target_prefixes=target_prefixes,
            resp_correct_prefix="response/correct",
            resp_error_prefix="response/error",
            max_latency=n400_max_latency,
        )

    if decim and decim > 1:
        epochs_all.decimate(decim)

    cond_order = sorted(cfg["conditions"].keys())
    code_to_label = {code: lbl for lbl, code in epochs_all.event_id.items()}
    ep_codes = epochs_all.events[:, 2]

    freqs = np.logspace(np.log10(tfr_fmin), np.log10(tfr_fmax), tfr_n_freqs)

    band_spec = {}
    for token in band_defs.split(","):
        token = token.strip()
        if not token:
            continue
        m = re.match(r"^([A-Za-z0-9_]+)\s*:\s*([0-9.]+)\s*-\s*([0-9.]+)$", token)
        if not m:
            raise ValueError(f"Bad band spec: {token}")
        band_spec[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    band_names = list(band_spec.keys())

    Y_tfr_avg_list = []
    Y_tfr_band_avg_list = []
    cond_list = []
    n_epochs_by_cond = {}
    trial_level_payload = {}

    for cond in cond_order:
        prefix = cfg["conditions"][cond]
        codes = [code for code, label in code_to_label.items() if str(label).startswith(prefix)]
        if not codes:
            print(f"  [warn] no events for {cond!r} (prefix {prefix!r}); skip.")
            continue

        keep = np.isin(ep_codes, np.array(codes, int))
        if task_name == "N400" and n400_correct_only and epochs_all.metadata is not None:
            if prefix.startswith("stimulus/target/") and "correct" in epochs_all.metadata:
                correct_vec = epochs_all.metadata["correct"].fillna(False).to_numpy()
                keep = keep & correct_vec

        if not keep.any():
            print(f"  [warn] {cond!r} has 0 epochs after cleaning; skip.")
            continue

        ep_cond = epochs_all[keep]
        n_epochs_by_cond[cond] = int(len(ep_cond))

        tfr_baseline = cfg.get("tfr_baseline", (-0.2, 0.0))
        data_trials, tfr_times = compute_tfr_trials(
            ep_cond, freqs=freqs, cycles_rule=tfr_cycles, decim=tfr_decim, baseline=tfr_baseline, db=True, safety=0.90
        )

        data_trials = np.transpose(data_trials, (0, 3, 2, 1))  # (n_trials, T, F, M)
        data_trials = row_zscore_tf_topos(data_trials)
        avg_tf = data_trials.mean(axis=0)  # (T, F, M)

        band_cube = []
        for b in band_names:
            lo, hi = band_spec[b]
            fmask = (freqs >= lo) & (freqs <= hi)
            if not np.any(fmask):
                band_topo = np.full((avg_tf.shape[0], avg_tf.shape[2]), np.nan)
            else:
                band_topo = np.nanmean(avg_tf[:, fmask, :], axis=1)
            band_topo = zscore_rows(band_topo)
            band_cube.append(band_topo[..., None])

        if band_cube:
            Y_band = np.concatenate(band_cube, axis=2)  # (T, M, B)
            Y_band = np.transpose(Y_band, (0, 2, 1))    # (T, B, M)
        else:
            Y_band = np.empty((avg_tf.shape[0], 0, avg_tf.shape[2]))

        Y_tfr_avg_list.append(avg_tf)
        Y_tfr_band_avg_list.append(Y_band)
        cond_list.append(cond)

        if save_trials:
            trial_level_payload[f"Y_tfr_trials__{cond}"] = data_trials

    if not cond_list:
        raise RuntimeError(f"No non-empty conditions for subject {subject}, task {task_name}.")

    Y_tfr_avg = np.stack(Y_tfr_avg_list, axis=0)       # (L, T, F, M)
    Y_tfr_band_avg = np.stack(Y_tfr_band_avg_list, axis=0)  # (L, T, B, M)
    cond_array = np.array(cond_list, dtype=object)

    subj_deriv = os.path.join(out_root, "derivatives", "Y_tfr_core", f"sub-{subject}")
    os.makedirs(subj_deriv, exist_ok=True)
    out_npz = os.path.join(subj_deriv, f"sub-{subject}_task-{task_name}_Y_tfr.npz")

    n_epochs_items = np.array(list(n_epochs_by_cond.items()), dtype=object)

    payload = dict(
        canonical_channels=np.array(canonical_list, dtype=object),
        subject_mask=np.array(mask, dtype=bool),
        channels=np.array(used_channels, dtype=object),
        times=np.array(tfr_times, dtype=float),
        freqs=np.array(freqs, dtype=float),
        bands=np.array([*band_spec.keys()], dtype=object),
        task=np.array([task_name], dtype=object),
        session=np.array([session], dtype=object),
        conditions=cond_array,
        n_epochs_by_cond=n_epochs_items,
        Y_tfr_avg=Y_tfr_avg.astype(np.float32),
        Y_tfr_band_avg=Y_tfr_band_avg.astype(np.float32),
        tfr_fmin=np.array([tfr_fmin], dtype=float),
        tfr_fmax=np.array([tfr_fmax], dtype=float),
        tfr_n_freqs=np.array([tfr_n_freqs], dtype=int),
        tfr_cycles=np.array([tfr_cycles], dtype=object),
        tfr_decim=np.array([tfr_decim], dtype=int),
        tfr_baseline=np.array([cfg.get("tfr_baseline", (-0.2, 0.0))], dtype=object),
        l_freq=np.array([l_freq if l_freq is not None else np.nan], dtype=float),
        h_freq=np.array([h_freq if h_freq is not None else np.nan], dtype=float),
        notch_freq=np.array([notch_freq if notch_freq is not None else np.nan], dtype=float),
        raw_resample=np.array([raw_resample], dtype=float),
        use_autoreject=np.array([bool(use_autoreject)], dtype=bool),
        n400_correct_only=np.array([bool(n400_correct_only)], dtype=bool),
        n400_max_latency=np.array([n400_max_latency], dtype=float),
    )

    if save_trials:
        for k, v in trial_level_payload.items():
            payload[k] = v.astype(np.float32)

    np.savez_compressed(out_npz, **payload)

    return dict(
        subject=subject,
        task=task_name,
        session=session,
        n_channels=int(mask.sum()),
        n_canonical=len(canonical_list),
        n_conditions_saved=int(len(cond_list)),
        n_epochs_total=int(len(epochs_all)),
        tfr_T=int(len(tfr_times)),
        tfr_F=int(Y_tfr_avg.shape[2]),
        npz_path=out_npz,
    )


def main():
    ap = argparse.ArgumentParser(description="ERP-CORE batch ERP-TFR extractor (trial-avg + optional trial-level)")
    ap.add_argument(
        "--bids-root",
        default=os.path.expanduser("~/mne_data/ERP_CORE"),
        help="Where the ERP-CORE BIDS dataset lives or should be downloaded.",
    )
    ap.add_argument("--outdir", default=os.path.expanduser("~/erpcore_work"), help="Output root directory")
    ap.add_argument(
        "--canonical-file",
        required=False,
        help="Text file listing canonical EEG channels (one per line). Use your canonical_30.txt here.",
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["P3", "N2pc", "ERN"],
        help="Subset of tasks to process (MMN, N170, N2pc, N400, P3, LRP, ERN).",
    )
    ap.add_argument("--subjects", nargs="*", default=None, help="Optional list of subject IDs (e.g., 015 016 ...).")

    ap.add_argument("--decim", type=int, default=1, help="Epoch decimation factor (after N400 correctness).")
    ap.add_argument("--l-freq", type=float, default=0.1, help="High-pass cutoff (Hz). None for no HP.")
    ap.add_argument("--h-freq", type=float, default=None, help="Low-pass cutoff (Hz). None for no LP.")
    ap.add_argument("--notch-freq", type=float, default=60.0, help="Notch filter frequency (Hz). <=0 disables.")
    ap.add_argument(
        "--raw-resample", type=float, default=128.0, help="Resample frequency for raw prior to ICA/epoching."
    )
    ap.add_argument("--no-autoreject", action="store_true", help="Disable autoreject; use amplitude threshold instead.")
    ap.add_argument("--n400-correct-only", action="store_true", help="N400: keep only correct target epochs.")
    ap.add_argument(
        "--n400-max-latency",
        type=float,
        default=1.0,
        help="Max response latency (s) after stimulus to count as correct (N400).",
    )

    ap.add_argument("--tfr-fmin", type=float, default=2.0)
    ap.add_argument("--tfr-fmax", type=float, default=30.0)
    ap.add_argument("--tfr-n-freqs", type=int, default=25)
    ap.add_argument(
        "--tfr-cycles",
        choices=["const", "constQ"],
        default="constQ",
        help='"const" -> n_cycles=7; "constQ" -> n_cycles(f)=clip(f/2,3,10)',
    )
    ap.add_argument("--tfr-decim", type=int, default=2)

    ap.add_argument("--bands", default="theta:4-8,alpha:8-13,beta:13-30")
    ap.add_argument("--no-save-trials", action="store_true", help="Do not save trial-level TFR arrays.")

    args = ap.parse_args()

    bids_root_input = os.path.expanduser(args.bids_root)
    os.makedirs(bids_root_input, exist_ok=True)
    bids_root = ensure_erp_core_bids(bids_root_input)

    out_root = os.path.expanduser(args.outdir)
    os.makedirs(out_root, exist_ok=True)

    all_subs = find_subjects(bids_root)
    if not all_subs:
        raise SystemExit(f"No sub-* directories under {bids_root}.")

    if args.subjects:
        subjects = [s for s in args.subjects if s in all_subs]
        missing = set(args.subjects) - set(subjects)
        if missing:
            print(f"[warn] Requested subjects not found: {sorted(missing)}")
    else:
        subjects = all_subs

    if not subjects:
        raise SystemExit("No valid subjects to process after filtering.")

    if args.canonical_file:
        canonical_list = load_canonical_list(args.canonical_file)
        print(f"[info] Loaded canonical list ({len(canonical_list)} ch) from {args.canonical_file}")
    else:
        first_sub = subjects[0]
        first_task = args.tasks[0]
        cfg0 = get_task_config(first_task)
        raw0 = read_raw_for_task(bids_root, first_sub, cfg0["session"], cfg0["task"])
        raw0.pick(mne.pick_types(raw0.info, eeg=True, eog=False, ecg=False, stim=False, misc=False))
        canonical_list = list(raw0.ch_names)
        print(
            "[warn] --canonical-file not provided. "
            f"Derived canonical list of {len(canonical_list)} channels from sub-{first_sub}, task {first_task}."
        )

    notch = args.notch_freq if (args.notch_freq and args.notch_freq > 0) else None
    use_autoreject = not args.no_autoreject

    print(
        textwrap.dedent(
            f"""
        [info] ERP-CORE BIDS root: {bids_root}
        [info] Output root:        {out_root}
        [info] Subjects:           {', '.join(subjects)}
        [info] Tasks:              {', '.join(args.tasks)}
        [info] Raw resample:       {args.raw_resample} Hz
        [info] Epoch decim:        {args.decim} (after N400 correctness)
        [info] Filters:            {args.l_freq}–{args.h_freq} Hz, notch={notch}
        [info] Autoreject:         {'on' if use_autoreject else 'off'}
        [info] N400 correct-only:  {'on' if args.n400_correct_only else 'off'} (max_latency={args.n400_max_latency}s)
        [info] TFR freqs:          {args.tfr_fmin}–{args.tfr_fmax} Hz (n={args.tfr_n_freqs}), cycles={args.tfr_cycles}, decim={args.tfr_decim}
        [info] Bands:              {args.bands}
        [info] Save trials:        {'on' if not args.no_save_trials else 'off'}
    """
        ).strip()
    )

    manifest_rows = []
    for subj in subjects:
        for task in args.tasks:
            try:
                print(f"\n=== sub-{subj} / task-{task} ===")
                row = process_subject_task(
                    subject=subj,
                    task=task,
                    bids_root=bids_root,
                    out_root=out_root,
                    canonical_list=canonical_list,
                    decim=args.decim,
                    l_freq=args.l_freq if args.l_freq and args.l_freq > 0 else None,
                    h_freq=args.h_freq if (args.h_freq is None or args.h_freq > 0) else None,
                    notch_freq=notch,
                    raw_resample=args.raw_resample,
                    use_autoreject=use_autoreject,
                    n400_correct_only=args.n400_correct_only,
                    n400_max_latency=args.n400_max_latency,
                    tfr_fmin=args.tfr_fmin,
                    tfr_fmax=args.tfr_fmax,
                    tfr_n_freqs=args.tfr_n_freqs,
                    tfr_cycles=args.tfr_cycles,
                    tfr_decim=args.tfr_decim,
                    band_defs=args.bands,
                    save_trials=(not args.no_save_trials),
                )
                manifest_rows.append(row)
            except FileNotFoundError as e:
                print(f"  [skip] {e}")
            except Exception as e:
                print(f"  [ERROR] sub-{subj}, task-{task}: {e}")

    manifest_dir = os.path.join(out_root, "derivatives", "Y_tfr_core")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_csv = os.path.join(manifest_dir, "manifest.csv")

    if manifest_rows:
        df = pd.DataFrame(manifest_rows).sort_values(["subject", "task"])
    else:
        df = pd.DataFrame(
            columns=[
                "subject",
                "task",
                "session",
                "n_channels",
                "n_canonical",
                "n_conditions_saved",
                "n_epochs_total",
                "tfr_T",
                "tfr_F",
                "npz_path",
            ]
        )

    df.to_csv(manifest_csv, index=False)
    print(f"\n[info] Wrote manifest: {manifest_csv} (rows={len(df)})")




if __name__ == "__main__":
    main()
