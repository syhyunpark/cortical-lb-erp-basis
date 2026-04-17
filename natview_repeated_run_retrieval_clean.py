#!/usr/bin/env python3
"""
natview_repeated_run_retrieval_clean.py

Repeated-run retrieval / matching on NatView EEG 

A "stimulus" means the identity of the audiovisual clip, e.g.
    - The Present
    - Despicable Me (English)
    - Despicable Me (Hungarian)
    - Monkey 1
    - Monkey 2
    - Monkey 5

A repeated-run pair means the same stimulus appears twice within the same subject and session, typically as run-01 and run-02.

Example:
    query      = The Present run-01
    candidates = all other repeated runs from the same subject/session
    success    = nearest neighbor is The Present run-02

Main outputs
------------
- query_results.csv
- session_results.csv
- summary_metrics.csv
- pairwise_tests.csv
- session_meta.csv
- analysis_meta.json
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

 
# Basic helpers 

def load_summary(summary_csv: str) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    return df


def load_lb_sph(lb_sph_npz: str):
    d = np.load(lb_sph_npz, allow_pickle=True)
    channels =  np.array(d["channels"]).astype(str)

    if "D_lb" in d:
        D_lb = np.asarray(d["D_lb"], float)
    elif "D" in d:
        D_lb = np.asarray(d["D"], float)
    else:
        raise KeyError(f"{lb_sph_npz} missing 'D_lb' or 'D'")

    if "B_sph" in d:
        B_sph = np.asarray(d["B_sph"], float)
    else:
        raise KeyError(f"{lb_sph_npz} missing 'B_sph'")

    return channels, D_lb, B_sph




def load_pca(pca_npz: str):
    d = np.load(pca_npz, allow_pickle=True)
    channels = np.array(d["channels"]).astype(str)
    mu = np.asarray(d["mu_master"], float)
    U = np.asarray(d["U_pca_master"], float)
    return channels, mu, U


def cosine_similarity(x, y, eps=1e-12):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx < eps or ny < eps:
        return np.nan
    return float(np.dot(x, y) / (nx * ny))


def starts_to_ticks(starts_sec, scale=1000):
    return np.rint(np.asarray(starts_sec, float) * scale).astype(np.int64)


def parse_bin_edges(edge_str):
    if edge_str is None or edge_str == "":
        return None
    vals = [float(x.strip()) for x in edge_str.split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError("freq-bin-edges must contain at least two comma-separated values")
    vals = sorted(vals)
    return vals



def normalize_columns(B, eps=1e-12):
    """
    Normalize columns of B to unit L2 norm.
    """
    B = np.asarray(B, float)
    norms = np.linalg.norm(B, axis=0)
    norms = np.where(norms < eps, 1.0, norms)
    return B / norms[None, :]

 
# Frequency reduction 

def reduce_frequency_axis(X, freqs, fmin=None, fmax=None, freq_decim=1, bin_edges=None):
    """
    Reduce frequency axis of X.

    X: (..., F, M)
    freqs: (F,)

    If bin_edges is given:
        average within each [edge_i, edge_{i+1}) bin
        with the final bin inclusive of upper edge.

    Else:
        restrict to [fmin, fmax] and decimate by freq_decim.

    Returns
    -------
    X_red : (..., F_red, M)
    freq_repr : (F_red,)
    """
    freqs = np.asarray(freqs, float)
    X = np.asarray(X, float)

    if bin_edges is not None:
        pieces = []
        centers = []
        for i in range(len(bin_edges) - 1):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            if i < len(bin_edges) - 2:
                mask = (freqs >= lo) & (freqs < hi)
            else:
                mask = (freqs >= lo) & (freqs <= hi)
            if not np.any(mask):
                continue
            pieces.append(np.nanmean(X[..., mask, :], axis=-2))
            centers.append(0.5 * (lo + hi))
        if not pieces:
            raise RuntimeError("No frequency bins contained any dense PSD frequencies.")
        X_red = np.stack(pieces, axis=-2)
        freq_repr = np.array(centers, float)
        return X_red, freq_repr

    mask = np.ones_like(freqs, dtype=bool)
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax
    if not np.any(mask):
        raise RuntimeError("No dense PSD frequencies after fmin/fmax restriction.")

    X_sel = X[..., mask, :]
    f_sel = freqs[mask]
    if freq_decim > 1:
        X_sel = X_sel[..., ::freq_decim, :]
        f_sel = f_sel[::freq_decim]
    return X_sel, f_sel

 
# Repeated-run inventory 

def build_repeated_inventory(df,
                             min_windows=20,
                             min_present_channels=40,
                             min_pairs_per_session=2):
    keep = df.copy()

    if "n_kept_windows" in keep.columns:
        keep = keep[keep["n_kept_windows"] >= min_windows]
    if "n_present_channels" in keep.columns:
        keep = keep[keep["n_present_channels"] >= min_present_channels]

    keep["run_id"] = (
        keep["subject"].astype(str) + "|" +
        keep["session"].astype(str) + "|" +
        keep["task"].astype(str)
    )


    pair_rows = []
    for (subj, sess, stim), g in keep.groupby(["subject", "session", "stimulus_label"]):
        if g.shape[0] == 2:
            g = g.sort_values("task")
            pair_rows.append({
                "subject": subj,
                "session": sess,
                "stimulus_label": stim,
                "run1_id": g.iloc[0]["run_id"],
                "run2_id": g.iloc[1]["run_id"],
            })

    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        return keep.iloc[0:0].copy(), pair_df

    #### Keep only subject-sessions with at least min_pairs_per_session repeated pairs
    sess_counts = pair_df.groupby(["subject", "session"]).size().reset_index(name="n_pairs")
    good_sessions = sess_counts[sess_counts["n_pairs"] >= min_pairs_per_session][["subject", "session"]]
    if good_sessions.empty:
        return keep.iloc[0:0].copy(), pair_df.iloc[0:0].copy()

    good_sessions["keep"] = 1
    pair_df = pair_df.merge(good_sessions, on=["subject", "session"], how="inner").drop(columns="keep")
    keep = keep.merge(good_sessions, on=["subject", "session"], how="inner").drop(columns="keep")

    repeated_run_ids = set(pair_df["run1_id"]).union(set(pair_df["run2_id"]))
    run_df = keep[keep["run_id"].isin(repeated_run_ids)].copy()

    return run_df, pair_df


def build_session_run_dict(run_df):
    out = {}
    for (subj, sess), g in run_df.groupby(["subject", "session"]):
        out[(subj, sess)] = g.copy().reset_index(drop=True)
    return out


def build_session_meta(pair_df):
    """
    For each subject-session:
      n_pairs
      n_runs = 2*n_pairs
      chance_top1 = 1 / (n_runs - 1)
    """
    rows = []
    for (subj, sess), g in pair_df.groupby(["subject", "session"]):
        n_pairs = int(g.shape[0])
        n_runs = 2 * n_pairs
        chance_top1 = 1.0 / (n_runs - 1)
        rows.append({
            "subject": subj,
            "session": sess,
            "n_pairs": n_pairs,
            "n_runs": n_runs,
            "chance_top1": chance_top1,
        })
    meta_df = pd.DataFrame(rows)
    meta_dict = {
        (row["subject"], row["session"]): {
            "n_pairs": int(row["n_pairs"]),
            "n_runs": int(row["n_runs"]),
            "chance_top1": float(row["chance_top1"]),
        }
        for _, row in meta_df.iterrows()
    }
    return meta_df, meta_dict


def get_matched_run_id(query_row, session_runs):
    stim = query_row["stimulus_label"]
    same = session_runs[
        (session_runs["stimulus_label"] == stim) &
        (session_runs["run_id"] != query_row["run_id"])
    ]
    if same.shape[0] != 1:
        return None
    return same.iloc[0]["run_id"]

 
# Shared masked basis cache 

def build_basis_cache_for_mask(mask,
                               D_lb_master,
                               B_sph_master,
                               mu_master,
                               U_master,
                               K_values,
                               ridge_alpha=0.1):
    """
    Build shared masked basis transforms with column normalization and ridge coefficients.

    For each basis and each K:
        T_K = B_norm[:, :K] @ (B_norm[:, :K]^T B_norm[:, :K] + alpha I)^{-1}

    coefficients are then:
        c = y @ T_K

    Returns
    -------
    cache["LB"]["T_by_K"][K], cache["SPH"]["T_by_K"][K], cache["PCA"]["T_by_K"][K]
    """
    mask = np.asarray(mask, bool)
    Kmax = max(K_values)

    basis_master = {
        "LB": D_lb_master[mask, :Kmax],
        "SPH": B_sph_master[mask, :Kmax],
        "PCA": U_master[mask, :Kmax],
    }
    mu_mask = mu_master[mask]

    cache = {}
    for basis_name, B in basis_master.items():
        B_norm = normalize_columns(B)
        T_by_K = {}

        for K in K_values:
            Bk = B_norm[:, :K]              # M_r x K
            G = Bk.T @ Bk                   # K x K
            A = np.linalg.solve(G + ridge_alpha * np.eye(K), np.eye(K))
            T = Bk @ A                      # M_r x K
            T_by_K[K] = T

        cache[basis_name] = {
            "T_by_K": T_by_K,
            "mu_mask": mu_mask if basis_name == "PCA" else None,
        }

    return cache

 
# Dense trajectory extraction 

def extract_dense_window_feature_trajectories(dense_file,
                                              expected_master_channels,
                                              basis_cache_for_mask,
                                              K_values,
                                              include_raw=True,
                                              fmin=2.0,
                                              fmax=30.0,
                                              freq_decim=1,
                                              freq_bin_edges=None):
    """
    Build per-window signed dense-PSD feature trajectories.

    Returns
    -------
    starts_ticks : (W,)
    trajs : dict
        ("LB", K)   -> (W, F_red*K)
        ("SPH", K)  -> (W, F_red*K)
        ("PCA", K)  -> (W, F_red*K)
        ("RAW",None)-> (W, F_red*61)
    n_windows : int
    n_present : int
    n_freq_repr : int
    freq_repr : (F_red,)
    """
    d = np.load(dense_file, allow_pickle=True)
    file_channels = np.array(d["master_channels"]).astype(str)
    if not np.array_equal(file_channels, expected_master_channels):
        raise RuntimeError(f"master_channels mismatch in {dense_file}")

    present_mask = d["present_mask"].astype(bool)
    starts_sec = np.asarray(d["window_start_sec"], float)
    starts_ticks = starts_to_ticks(starts_sec)

    freqs = np.asarray(d["freqs"], float)
    X_master = np.asarray(d["dense_logpsd_z_master"], float)   # (W, F, 61)

    X_red_master, freq_repr = reduce_frequency_axis(
        X_master, freqs,
        fmin=fmin, fmax=fmax,
        freq_decim=freq_decim,
        bin_edges=freq_bin_edges
    )

    X_red = X_red_master[:, :, present_mask]   # (W, F_red, M_r)
    W, F_red, M_r = X_red.shape

    trajs = {}

    for basis_name in ["LB", "SPH", "PCA"]:
        mu_mask = basis_cache_for_mask[basis_name]["mu_mask"]

        if basis_name == "PCA":
            X_use = X_red - mu_mask[None, None, :]
        else:
            X_use = X_red

        X2 = X_use.reshape(W * F_red, M_r)

        for K in K_values:
            T = basis_cache_for_mask[basis_name]["T_by_K"][K]   # M_r x K
            C = X2 @ T                                          # (W*F_red) x K
            C = C.reshape(W, F_red, K)
            trajs[(basis_name, K)] = C.reshape(W, F_red * K)

    if include_raw:
        X_raw = np.array(X_red_master, copy=True)
        X_raw[~np.isfinite(X_raw)] = 0.0
        trajs[("RAW", None)] = X_raw.reshape(W, F_red * X_red_master.shape[2])

    return starts_ticks, trajs, int(W), int(present_mask.sum()), int(F_red), freq_repr
 

# Aligned-window similarity 

def aligned_window_similarity(starts_a, F_a, starts_b, F_b, min_common_windows=5):
    common = np.intersect1d(starts_a, starts_b)
    if common.size < min_common_windows:
        return np.nan, int(common.size)

    idx_a = {t: i for i, t in enumerate(starts_a)}
    idx_b = {t: i for i, t in enumerate(starts_b)}

    sims = []
    for t in common:
        va = F_a[idx_a[t]]
        vb = F_b[idx_b[t]]
        sim = cosine_similarity(va, vb)
        if np.isfinite(sim):
            sims.append(sim)

    if len(sims) < min_common_windows:
        return np.nan, len(sims)

    return float(np.mean(sims)), int(len(sims))

 


# Retrieval task 

def compute_query_results(run_df,
                          session_runs_dict,
                          session_meta_dict,
                          feature_cache,
                          K_values,
                          include_raw=True,
                          min_common_windows=5):
    rows = []

    method_keys = [("LB", K) for K in K_values] + \
                  [("SPH", K) for K in K_values] + \
                  [("PCA", K) for K in K_values]
    if include_raw:
        method_keys += [("RAW", None)]

    for _, query_row in run_df.iterrows():
        subj = query_row["subject"]
        sess = query_row["session"]
        run_id = query_row["run_id"]
        stim = query_row["stimulus_label"]

        session_runs = session_runs_dict[(subj, sess)]
        session_meta = session_meta_dict[(subj, sess)]
        chance_top1 = session_meta["chance_top1"]

        candidates = session_runs[session_runs["run_id"] != run_id].copy()
        matched_run_id = get_matched_run_id(query_row, session_runs)
        if matched_run_id is None:
            continue

        for basis_name, K in method_keys:
            q_data = feature_cache[(run_id, basis_name, K)]
            q_starts = q_data["starts"]
            q_F = q_data["F"]

            sims = []
            for _, cand_row in candidates.iterrows():
                cand_id = cand_row["run_id"]
                c_data = feature_cache[(cand_id, basis_name, K)]

                sim_mean, n_common = aligned_window_similarity(
                    q_starts, q_F,
                    c_data["starts"], c_data["F"],
                    min_common_windows=min_common_windows
                )

                sims.append({
                    "cand_run_id": cand_id,
                    "cand_stimulus_label": cand_row["stimulus_label"],
                    "sim": sim_mean,
                    "n_common_windows": n_common
                })

            sims_df = pd.DataFrame(sims)
            sims_df = sims_df[np.isfinite(sims_df["sim"])].copy()
            if sims_df.empty:
                continue

            sims_df = sims_df.sort_values("sim", ascending=False).reset_index(drop=True)

            match_rows = sims_df[sims_df["cand_run_id"] == matched_run_id]
            if match_rows.empty:
                continue

            matched_rank = int(match_rows.index[0]) + 1
            reciprocal_rank = 1.0 / matched_rank

            sim_match = float(match_rows["sim"].iloc[0])
            n_common_match = int(match_rows["n_common_windows"].iloc[0])

            nonmatch = sims_df[sims_df["cand_run_id"] != matched_run_id]["sim"].values
            sim_nonmatch_mean = float(np.mean(nonmatch)) if len(nonmatch) > 0 else np.nan
            margin = sim_match - sim_nonmatch_mean if np.isfinite(sim_nonmatch_mean) else np.nan

            top1_correct = int(matched_rank == 1)

            rows.append({
                "subject": subj,
                "session": sess,
                "query_run_id": run_id,
                "query_task": query_row["task"],
                "stimulus_label": stim,
                "basis": basis_name,
                "K": K if K is not None else np.nan,
                "matched_run_id": matched_run_id,
                "chance_top1": chance_top1,
                "top1_correct": top1_correct,
                "matched_rank": matched_rank,
                "reciprocal_rank": reciprocal_rank,
                "sim_match": sim_match,
                "sim_nonmatch_mean": sim_nonmatch_mean,
                "margin": margin,
                "n_valid_candidates": int(sims_df.shape[0]),
                "n_common_windows_match": n_common_match,
            })

    return pd.DataFrame(rows)


def aggregate_session_results(query_df):
    sess_df = (
        query_df
        .groupby(["subject", "session", "basis", "K"], dropna=False)
        .agg(
            n_queries=("top1_correct", "size"),
            chance_top1=("chance_top1", "first"),
            mean_top1_accuracy=("top1_correct", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            mean_matched_rank=("matched_rank", "mean"),
            mean_margin=("margin", "mean"),
            mean_common_windows_match=("n_common_windows_match", "mean")
        )
        .reset_index()
    )

    sess_df["chance_adjusted_top1_accuracy"] = (
        sess_df["mean_top1_accuracy"] - sess_df["chance_top1"]
    )

    return sess_df

 
# Bootstrap summaries and paired tests 

def bootstrap_mean(x, B=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(B, float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = np.mean(x[idx])
    return float(np.mean(x)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def make_summary_metrics(sess_df, B=2000, alpha=0.05):
    out = []
    for (basis, K), g in sess_df.groupby(["basis", "K"], dropna=False):
        top1_mean, top1_lo, top1_hi = bootstrap_mean(g["mean_top1_accuracy"].values, B=B, alpha=alpha, seed=11)
        adj_mean, adj_lo, adj_hi = bootstrap_mean(g["chance_adjusted_top1_accuracy"].values, B=B, alpha=alpha, seed=12)
        mrr_mean, mrr_lo, mrr_hi = bootstrap_mean(g["mean_reciprocal_rank"].values, B=B, alpha=alpha, seed=13)
        rank_mean, rank_lo, rank_hi = bootstrap_mean(g["mean_matched_rank"].values, B=B, alpha=alpha, seed=14)
        mar_mean, mar_lo, mar_hi = bootstrap_mean(g["mean_margin"].values, B=B, alpha=alpha, seed=15)

        out.append({
            "basis": basis,
            "K": K,
            "n_sessions": g.shape[0],

            "mean_top1_accuracy": top1_mean,
            "top1_ci_low": top1_lo,
            "top1_ci_high": top1_hi,

            "mean_chance_adjusted_top1_accuracy": adj_mean,
            "adj_top1_ci_low": adj_lo,
            "adj_top1_ci_high": adj_hi,

            "mean_reciprocal_rank": mrr_mean,
            "mrr_ci_low": mrr_lo,
            "mrr_ci_high": mrr_hi,

            "mean_matched_rank": rank_mean,
            "rank_ci_low": rank_lo,
            "rank_ci_high": rank_hi,

            "mean_margin": mar_mean,
            "margin_ci_low": mar_lo,
            "margin_ci_high": mar_hi,
        })

    return pd.DataFrame(out).sort_values(["basis", "K"], na_position="last")


def paired_tests_at_k(sess_df, K_compare=10):
    out = []

    def compare(df_a, df_b, label_a, label_b, metric):
        merged = df_a.merge(df_b, on=["subject", "session"], suffixes=("_a", "_b"))
        if merged.shape[0] < 5:
            return None
        x = merged[f"{metric}_a"].values
        y = merged[f"{metric}_b"].values
        diffs = x - y
        try:
            stat, p = wilcoxon(diffs)
        except Exception:
            stat, p = np.nan, np.nan
        return {
            "comparison": f"{label_a} - {label_b}",
            "metric": metric,
            "n_sessions": merged.shape[0],
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "wilcoxon_stat": stat,
            "p_value": p,
        }

    keep_cols = ["subject", "session", "mean_top1_accuracy", "chance_adjusted_top1_accuracy", "mean_reciprocal_rank", "mean_matched_rank", "mean_margin"]

    lb = sess_df[(sess_df["basis"] == "LB") & (sess_df["K"] == K_compare)][keep_cols]
    sph = sess_df[(sess_df["basis"] == "SPH") & (sess_df["K"] == K_compare)][keep_cols]
    pca = sess_df[(sess_df["basis"] == "PCA") & (sess_df["K"] == K_compare)][keep_cols]
    raw = sess_df[(sess_df["basis"] == "RAW")][keep_cols]

    metrics = [
        "mean_top1_accuracy",
        "chance_adjusted_top1_accuracy",
        "mean_reciprocal_rank",
        "mean_matched_rank",
        "mean_margin",
    ]

    for metric in metrics:
        comparisons = [
            compare(lb, sph, "LB", "SPH", metric),
            compare(lb, pca, "LB", "PCA", metric),
        ]
        if not raw.empty:
            comparisons.append(compare(lb, raw, "LB", "RAW", metric))

        for res in comparisons:
            if res is not None:
                out.append(res)

    return pd.DataFrame(out)

 
# 	Main 

def main():
    ap = argparse.ArgumentParser(description="Repeated-run retrieval on NatView EEG using dense PSD, flexible frequency bins, and shared masked ridge-basis coefficients.")
    ap.add_argument("--summary-csv", required=True,
                    help="feature_export_summary.csv from export_natview_eeg_features_masked.py")
    ap.add_argument("--lb-sph-npz", required=True,
                    help="NPZ with D_lb (or D) and B_sph on master61 montage")
    ap.add_argument("--pca-npz", required=True,
                    help="master61 PCA NPZ built from dense PSD")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 15, 20])

    ap.add_argument("--ridge-alpha", type=float, default=0.1)

    ap.add_argument("--min-windows", type=int, default=20)
    ap.add_argument("--min-present-channels", type=int, default=40)
    ap.add_argument("--min-pairs-per-session", type=int, default=2)
    ap.add_argument("--min-common-windows", type=int, default=5)
    ap.add_argument("--include-raw-baseline", action="store_true")

    ap.add_argument("--bootstrap-B", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--k-compare", type=int, default=10)

    # Flexible dense-frequency handling
    ap.add_argument("--fmin", type=float, default=2.0)
    ap.add_argument("--fmax", type=float, default=30.0)
    ap.add_argument("--freq-decim", type=int, default=1)
    ap.add_argument("--freq-bin-edges", type=str, default="2,4,6,8,10,13,16,20,24,30",
                    help="Comma-separated dense PSD bin edges; set to 2,4,8,13,30 for canonical 4 bands")

    args = ap.parse_args()
    freq_bin_edges = parse_bin_edges(args.freq_bin_edges)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load summary and bases
    df = load_summary(args.summary_csv)
    channels_lb, D_lb_master, B_sph_master = load_lb_sph(args.lb_sph_npz)
    channels_pca, mu_master, U_master = load_pca(args.pca_npz)

    if not np.array_equal(channels_lb, channels_pca):
        raise RuntimeError("LB/SPH and PCA channel orders do not match....")

    # Build repeated-run inventory
    run_df, pair_df = build_repeated_inventory(
        df,
        min_windows=args.min_windows,
        min_present_channels=args.min_present_channels,
        min_pairs_per_session=args.min_pairs_per_session
    )
    if run_df.empty or pair_df.empty:
        raise RuntimeError("No usable repeated-run sessions after filtering......")

    run_df.to_csv(outdir / "run_inventory_repeated.csv", index=False)
    pair_df.to_csv(outdir / "pair_inventory_repeated.csv", index=False)

    session_meta_df, session_meta_dict = build_session_meta(pair_df)
    session_meta_df.to_csv(outdir / "session_meta.csv", index=False)

    # Cache masked basis transforms by unique run-specific channel mask
    basis_cache = {}
    feature_cache = {}

    freq_repr_saved = None
    n_freq_repr = None

    for _, row in run_df.iterrows():
        run_id = row["run_id"]
        dense_file = row["dense_file"]

        feat = np.load(dense_file, allow_pickle=True)
        file_channels = np.array(feat["master_channels"]).astype(str)
        if not np.array_equal(file_channels, channels_lb):
            raise RuntimeError(f"master_channels mismatch in {dense_file}")

        mask = feat["present_mask"].astype(bool)
        mask_key = (mask.astype(np.uint8).tobytes(), float(args.ridge_alpha), tuple(args.k_values))

        if mask_key not in basis_cache:
            basis_cache[mask_key] = build_basis_cache_for_mask(
                mask=mask,
                D_lb_master=D_lb_master,
                B_sph_master=B_sph_master,
                mu_master=mu_master,
                U_master=U_master,
                K_values=args.k_values,
                ridge_alpha=args.ridge_alpha
            )

        starts_ticks, trajs, n_windows, n_present, n_freq_repr, freq_repr = extract_dense_window_feature_trajectories(
            dense_file=dense_file,
            expected_master_channels=channels_lb,
            basis_cache_for_mask=basis_cache[mask_key],
            K_values=args.k_values,
            include_raw=args.include_raw_baseline,
            fmin=args.fmin,
            fmax=args.fmax,
            freq_decim=args.freq_decim,
            freq_bin_edges=freq_bin_edges
        )

        if freq_repr_saved is None:
            freq_repr_saved = freq_repr

        for (basis_name, K), F in trajs.items():
            feature_cache[(run_id, basis_name, K)] = {
                "starts": starts_ticks,
                "F": F,
            }

    # Query-level retrieval
    session_runs_dict = build_session_run_dict(run_df)
    query_df = compute_query_results(
        run_df=run_df,
        session_runs_dict=session_runs_dict,
        session_meta_dict=session_meta_dict,
        feature_cache=feature_cache,
        K_values=args.k_values,
        include_raw=args.include_raw_baseline,
        min_common_windows=args.min_common_widows if hasattr(args, "min_common_widows") else args.min_common_windows
    )
    if query_df.empty:
        raise RuntimeError("No query results were produced.")

    query_df.to_csv(outdir / "query_results.csv", index=False)

     # Session-level aggregation
    sess_df = aggregate_session_results(query_df)
    sess_df.to_csv(outdir / "session_results.csv", index=False)

     #  Summaries and paired tests
    summary_df = make_summary_metrics(sess_df, B=args.bootstrap_B, alpha=args.alpha)
    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)

    tests_df = paired_tests_at_k(sess_df, K_compare=args.k_compare)
    tests_df.to_csv(outdir / "pairwise_tests.csv", index=False)

    meta = {
        "summary_csv": args.summary_csv,
        "lb_sph_npz": args.lb_sph_npz,
          "pca_npz": args.pca_npz,
        "k_values": args.k_values,
        "ridge_alpha": args.ridge_alpha,
        "min_windows": args.min_windows,
        "min_present_channels": args.min_present_channels,
          "min_pairs_per_session": args.min_pairs_per_session,
        "min_common_windows": args.min_common_windows,
        "include_raw_baseline": bool(args.include_raw_baseline),
        "fmin": args.fmin,
        "fmax": args.fmax,
         "freq_decim": args.freq_decim,
         "freq_bin_edges": args.freq_bin_edges,
        "n_runs": int(run_df.shape[0]),
        "n_pairs": int(pair_df.shape[0]),
        "n_subject_sessions": int(run_df[["subject", "session"]].drop_duplicates().shape[0]),
        "n_freq_repr": int(n_freq_repr) if n_freq_repr is not None else None,
         "freq_repr": freq_repr_saved.tolist() if freq_repr_saved is not None else None,
    }
    with open(outdir / "analysis_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[info] Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()