# cortical-lb-erp-basis
Cortex-anchored sensor space basis for analysis of event-related potentials (ERP)

# Cortex-anchored sensor-space bases for evoked EEG

This repository provides code to:

1. **Build a cortex-anchored sensor-space dictionary** by forward-projecting Laplace–Beltrami (LB) eigenmodes on the fsaverage cortical template through an MNE BEM head model, for **any EEG montage** that can be mapped to the `standard_1005` system.
2. **Add a spherical harmonics (SPH) comparator basis** on the same channels.
3. **Reproduce the ERP-CORE analyses** from the manuscript: trial-averaged time–frequency (TF) span tests, mode-wise TF energy spectra, and ERP contrast alignment.

The core idea is that instead of treating each electrode as an unrelated axis, we analyze evoked EEG in a **geometry-aligned basis** whose columns are the scalp projections of cortical LB eigenmodes (and SPH harmonics for comparison).

---

## 0. Requirements

- Python ≥ 3.9
- Packages:
  - `mne`
  - `numpy`
  - `scipy`
  - `nibabel`
  - `lapy` (for Laplace–Beltrami eigenmodes)
  - `pandas`
  - `matplotlib`
- FreeSurfer `fsaverage` (fetched automatically via `mne.datasets.fetch_fsaverage()`)

Install dependencies e.g.:

```bash
pip install mne lapy nibabel scipy numpy pandas matplotlib
```

## 1. Derive a canonical montage for your dataset

If you have a BIDS-like ERP dataset with per-subject NPZ files, you can derive a canonical channel list as the intersection of channels across subjects.

For the ERP-CORE dataset, we used:
```bash
python canonical_30.py \
  --npz-root /path/to/derivatives/Y_erp_core \
  --out canonical_30.txt \
  --min-presence 1.0
```
This writes one channel name per line to canonical_30.txt

## 2. Build the fsaverage LB dictionary for your montage

Given a channel list (e.g., canonical_30.txt), you can build the LB sensor dictionary:
```bash
python make_fsaverage_lb_dictionary.py \
  --canonical-file canonical_30.txt \
  --outdir /path/to/dict_out \
  --K 60 \
  --spacing ico4 \
  --bem-ico 4
```
This script:
	1.	Creates an mne.Info with the channels in canonical_30.txt
	
	2.	Attaches the standard_1005 montage (ignoring missing labels)
	
	3.	Sets up an fsaverage source space at the chosen spacing (e.g., ico4)
	
	4.	Builds a 3-layer BEM model (skin, skull, brain; 0.3/0.006/0.3 S/m)
	
	5.	Computes a fixed-orientation EEG forward matrix ($G \in \mathbb{R}^{M \times V_{\text{src}}}$)
	
	6.	Calls get_phi_fsaverage(...) to compute the first K symmetric LB eigenmodes ($\Phi_{\text{sym}} \in \mathbb{R}^{V_{\text{src}} \times K}$)
	
	7.	Forms the sensor dictionary ($D = G \Phi_{\text{sym}} \in \mathbb{R}^{M \times K}$)

It writes two NPZ files under outdir/derivatives/Dict:

	•	fsaverage_phi_sym_K{K}_{spacing}.npz  – cortical eigenmodes ($\Phi_{\text{sym}}$)
	
	•	fsaverage_D_sym_K{K}_M{M}_{spacing}.npz – LB sensor dictionary (D) with:
			- D        : array of shape (M, K)
			- channels : channel names
			-spacing, subject, K, combine metadata

For the ERP-CORE analysis we used K=60 and the 30-channel canonical montage, but you can choose any K and any montage (as long as labels map to standard_1005).

## 3. Add spherical harmonics (SPH) on the same channels

To build an SPH comparator basis on exactly the same channels, run:
```bash
python make_lb_sph_dict.py \
  --lb-dict-npz /path/to/dict_out/derivatives/Dict/fsaverage_D_sym_K60_M30_ico4.npz \
  --out-npz     /path/to/dict_out/derivatives/Dict/lb_sph_canonical30_ico4.npz \
  --max-k-lb    30 \
  --max-k-sph   30
```
Output NPZ contains:
	•	channels: same channel names
	•	D_lb    : LB dictionary (M, K_lb)
	•	B_sph   : native SPH basis (M, K_sph) (ℓ ≥ 1 only)
	•	Q_sph   : orthonormal SPH basis (M, K_sph)

This LB+SPH dictionary NPZ is what the downstream analyses use.

## 4. Reproducing the ERP-CORE pipeline 

### 4.1. Preprocess and compute TF representations
```bash
#Preprocess and compute TF representations
python erp_core_batch_evoked.py \
  --bids-root /path/to/ERP_CORE_BIDS \
  --outdir    /path/to/erpcore_work_full \
  --canonical-file canonical_30.txt \
  --tasks MMN N170 N2pc N400 P3 LRP ERN \
  --tfr-fmin 2 --tfr-fmax 30 --tfr-n-freqs 25 \
  --tfr-cycles constQ --tfr-decim 2
```


### 4.2. R²(K) for trial-averaged TF maps
```bash
python erp_tfr_span.py \
  --manifest /path/to/erpcore_work_full/derivatives/Y_tfr_core/manifest.csv \
  --dict-npz /path/to/dict_out/derivatives/Dict/lb_sph_canonical30_ico4.npz \
  --outdir   /path/to/erpcore_work_full/derivatives/Stage2_TFR_avg \
  --methods  LB SPH PCA ICA \
  --mode     avg \
  --K-list   1,5,10,15,20,25 \
  --eval-t-step 1 --eval-f-step 1 \
  --write-r2 full

#Bootstrap group summaries:
python erp_tfr_bootstrap.py \
  --root  /path/to/erpcore_work_full/derivatives/Stage2_TFR_avg \
  --B     2000 \
  --alpha 0.05

#Plot group R²(K) curves:
python plot_group_auc.py \
  --root    /path/to/erpcore_work_full/derivatives/Stage2_TFR_avg \
  --methods LB SPH PCA ICA
```

### 4.3. Mode-wise TF energy spectra
```bash
python erp_lb_mode_energy.py \
  --tfr-root /path/to/erpcore_work_full/derivatives/Y_tfr_core \
  --dict-npz /path/to/dict_out/derivatives/Dict/lb_sph_canonical30_ico4.npz \
  --outdir   /path/to/erpcore_work_full/derivatives/erp_mode_energy \
  --tasks    MMN N170 N2pc N400 P3 LRP ERN \
  --source   avg \
  --fmin     2.0 \
  --fmax     30.0 \
  --max-k-lb 30 \
  --max-k-sph 30 \
  --per-subject-csv

python erp_lb_mode_energy_bootstrap.py \
  --input-csv /path/to/erpcore_work_full/derivatives/erp_mode_energy/mode_energy_subject_avg.csv \
  --outdir    /path/to/erpcore_work_full/derivatives/erp_mode_energy \
  --B         2000 \
  --alpha     0.05

python erp_lb_mode_energy_plots.py \
  --mode-energy-dir /path/to/erpcore_work_full/derivatives/erp_mode_energy \
  --outdir          /path/to/erpcore_work_full/derivatives/erp_mode_energy/figs \
  --tasks           N170 N2pc N400 P3 ERN \
  --top-n-modes     3
```

### 4.4. Split-half reliability (ICC)
```bash
python erp_lb_reliability.py \
  --tfr-root /path/to/erpcore_work_full/derivatives/Y_tfr_core \
  --dict-npz /path/to/dict_out/derivatives/Dict/lb_sph_canonical30_ico4.npz \
  --outdir   /path/to/erpcore_work_full/derivatives/icc_reliability \
  --tasks    MMN N170 N2pc N400 P3 LRP ERN

python plot_icc_mode_curves.py \
  --icc-csv /path/to/erpcore_work_full/derivatives/icc_reliability/icc_modewise.csv \
  --outdir  /path/to/erpcore_work_full/derivatives/icc_reliability/fig_icc \
  --min-icc -0.2 \
  --max-icc  1.0 \
  --save

python summarize_icc_ranges.py \
  --icc-csv /path/to/erpcore_work_full/derivatives/icc_reliability/icc_modewise.csv \
  --outdir  /path/to/erpcore_work_full/derivatives/icc_reliability
```

### 4.5. ERP contrast alignment
```bash
python erp_alignment_gallery.py \
  --erp-root /path/to/erpcore_work_full/derivatives/Y_erp_core \
  --dict-npz /path/to/dict_out/derivatives/Dict/lb_sph_canonical30_ico4.npz \
  --outdir   /path/to/erpcore_work_full/derivatives/erp_lb_alignment \
  --tasks    MMN N170 N2pc N400 P3 LRP ERN \
  --gallery-tasks P3 N2pc N400 ERN \
  --K 15
```

## 5. Adapting to your own EEG dataset

To use this LB/SPH sensor dictionary for a different study:
	1.	Prepare a channel list for your montage: one label per line (e.g., my_montage.txt), using or mapping to standard_1005 names if possible.
	2.	Build the LB dictionary:
```bash
python make_fsaverage_lb_dictionary.py \
  --canonical-file my_montage.txt \
  --outdir /path/to/my_dict_out \
  --K 60 \
  --spacing ico4
```
	3.	Use the resulting NPZ (LB-only or LB+SPH) as the input --dict-npz for your own projection / span / energy analyses.

The geometry-anchored part (LB on fsaverage + BEM forward) stays the same; only the row selection (channels) changes with the montage. The LB dictionary builder is montage-agnostic: given a text file of channel names, it recomputes the fsaverage LB basis for that layout.
