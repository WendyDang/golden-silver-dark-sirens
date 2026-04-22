#!/usr/bin/env python
"""
SDS_bilby_maglim.py

Explores how different apparent-magnitude limits on the Uchuu galaxy catalog
affect the H0 posterior from silver gravitational-wave dark sirens.

Apparent magnitude cuts tested: MAG_LIMITS (AppMagStard_SDSSr column)
Events: first N_EVENTS from Ucchuu_silver_A# (sorted by injection number)

Output layout
-------------
maglim_study/
├── config_<timestamp>.txt          # top-level run config
├── run_<timestamp>.log             # full run log (also echoed to stdout)
├── sky_maps/                       # per-event sky maps (generated once, shared)
│   └── event_<N>_sky_map.png
├── mlim_19/
│   ├── H0_likelihoods.npz          # keys: H0_grid, joint_H0_posterior, event_<N>, ...
│   ├── H0_posteriors.png           # per-event + joint posterior plot
│   └── per_event_stats.csv         # per-event selection & host-survival statistics
├── mlim_21/
│   └── ...
├── mlim_23/
│   └── ...
└── comparison_H0_posteriors.png    # joint posteriors for all mag cuts overlaid
"""

import os
import re
import glob
import datetime
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
from astropy.table import Table
from ligo.skymap.postprocess import find_greedy_credible_levels
from bilby.core.result import read_in_result
from tqdm import tqdm

from find_gal_in_CI_varying_H0 import find_galaxies_in_sky_and_distance_CI_healpix
from H0_likelihood import H0_likelihood
from H0_posterior import H0_posterior

# ── Configuration ──────────────────────────────────────────────────────────────
CATALOG_DIR   = "/hildafs/projects/phy220048p/share/Uchuu/z_0.5_healpix_catalog"
UCHUU_NSIDE   = 64   # nside of the per-pixel parquet files

SILVER_CSV    = "/hildafs/projects/phy220048p/share/gwbench_dark_siren/bilby_result/Ucchuu_silver_injection_list_HsLsIs.csv"
SILVER_FOLDER = "/hildafs/projects/phy220048p/share/gwbench_dark_siren/bilby_result/Ucchuu_silver_A#"

MAG_LIMITS    = [19, 21, 23]        # apparent magnitude thresholds to test
N_EVENTS      = 10                  # use only the first N injection numbers
H0_GRID       = np.linspace(60, 80, 40)
CI_LEVEL      = 0.9
SELECTION_EFF = "HLI#S, SHELA, 0.5"
OUTPUT_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maglim_study")

APP_MAG_COL   = "AppMagStard_SDSSr"  # apparent SDSS-r magnitude (disk component)
ABS_MAG_COL   = "MagStar_SDSSr"      # absolute SDSS-r magnitude (total stellar; for diagnostics)

# ── Output directories ──────────────────────────────────────────────────────────
os.makedirs(OUTPUT_ROOT, exist_ok=True)
SKY_MAP_DIR = os.path.join(OUTPUT_ROOT, "sky_maps")
os.makedirs(SKY_MAP_DIR, exist_ok=True)
for _m in MAG_LIMITS:
    os.makedirs(os.path.join(OUTPUT_ROOT, f"mlim_{_m}"), exist_ok=True)

# ── Run identifier: timestamp + SLURM job ID if available ──────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
job_id    = os.environ.get("SLURM_JOB_ID", "local")
run_tag   = f"{timestamp}_job{job_id}"

# ── Logging: file + stdout ──────────────────────────────────────────────────────
log_path  = os.path.join(OUTPUT_ROOT, f"run_{run_tag}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Top-level config file ───────────────────────────────────────────────────────
config_path = os.path.join(OUTPUT_ROOT, f"config_{run_tag}.txt")
with open(config_path, "w") as _f:
    _f.write("Magnitude-cut study — run configuration\n")
    _f.write("=" * 60 + "\n")
    _f.write(f"Timestamp        : {datetime.datetime.now().isoformat()}\n")
    _f.write(f"SLURM job ID     : {job_id}\n")
    _f.write(f"Catalog          : Uchuu z<0.5 HEALPix ({CATALOG_DIR})\n")
    _f.write(f"Silver CSV       : {SILVER_CSV}\n")
    _f.write(f"Silver PE folder : {SILVER_FOLDER}\n")
    _f.write(f"N_EVENTS         : {N_EVENTS}\n")
    _f.write(f"MAG_LIMITS       : {MAG_LIMITS}  (column: {APP_MAG_COL})\n")
    _f.write(f"Abs mag column   : {ABS_MAG_COL}  (diagnostics only)\n")
    _f.write(f"Selection effects: {SELECTION_EFF}\n")
    _f.write(f"H0 grid          : [{H0_GRID[0]}, {H0_GRID[-1]}], n={len(H0_GRID)}\n")
    _f.write(f"CI level         : {CI_LEVEL}\n")
    _f.write(f"Output root      : {OUTPUT_ROOT}\n")
log.info(f"Config written to {config_path}")
log.info(f"Run log          : {log_path}")

# ── Load silver injection list ──────────────────────────────────────────────────
df_silver = pd.read_csv(SILVER_CSV)
log.info(f"Loaded silver injection list: {len(df_silver)} events")

# ── Find and sort Bilby result files ────────────────────────────────────────────
def extract_inj_number(path):
    match = re.search(r'inj_(\d+)', path)
    return int(match.group(1)) if match else -1

result_files = sorted(
    glob.glob(f"{SILVER_FOLDER}/inj_*/bilby_inj_*_result.json"),
    key=extract_inj_number,
)[:N_EVENTS]

log.info(f"Using {len(result_files)} Bilby result files (first {N_EVENTS} by injection number).")
for _f in result_files:
    log.info(f"  {_f}")

# ── Uchuu tile loader ───────────────────────────────────────────────────────────
def load_uchuu_for_event(ra_samples, dec_samples, ci_level=0.9):
    """Load Uchuu parquet tiles covering the sky CI of one event."""
    npix     = hp.nside2npix(2048)
    prob_map = np.zeros(npix)
    theta    = 0.5 * np.pi - dec_samples
    phi      = ra_samples
    np.add.at(prob_map, hp.ang2pix(2048, theta, phi), 1)
    prob_map /= prob_map.sum()

    cl          = find_greedy_credible_levels(prob_map)
    inside_high = np.where(cl <= ci_level)[0]

    theta_c, phi_c = hp.pix2ang(2048, inside_high)
    cat_pixels = np.unique(hp.ang2pix(UCHUU_NSIDE, theta_c, phi_c, nest=True))

    frames = []
    for pid in cat_pixels:
        path = os.path.join(CATALOG_DIR, f"healpix_{pid:06d}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
    if not frames:
        return None

    cat_df = pd.concat(frames, ignore_index=True)
    cat_df['stellar_mass'] = cat_df['MstarBulge'] + cat_df['MstarDisk']
    return Table.from_pandas(cat_df)

# ── Per-run data stores ──────────────────────────────────────────────────────────
H0_dicts   = {m: {} for m in MAG_LIMITS}   # {m_lim: {event_N: like_H0}}
stats_rows = {m: [] for m in MAG_LIMITS}   # {m_lim: [row_dict, ...]}

# ── Main loop: one catalog load + CI selection per event ─────────────────────────
for fname in tqdm(result_files, desc="Events"):
    inj_num = extract_inj_number(fname)
    log.info(f"\n{'='*60}")
    log.info(f"Event inj_{inj_num}: {fname}")

    row = df_silver.iloc[inj_num]

    result    = read_in_result(fname)
    posterior = result.posterior
    ra_samp   = posterior["ra"].values
    dec_samp  = posterior["dec"].values
    dL_samp   = posterior["luminosity_distance"].values

    # Load Uchuu tiles — done once per event, reused for all mag cuts
    log.info(f"  Loading Uchuu tiles...")
    catalog = load_uchuu_for_event(ra_samp, dec_samp, ci_level=CI_LEVEL)
    if catalog is None or len(catalog) == 0:
        log.warning(f"  No Uchuu tiles loaded for inj_{inj_num}, skipping.")
        for m in MAG_LIMITS:
            stats_rows[m].append(dict(
                inj_num=inj_num, DL_true=row['DL'], z_true=row['z'],
                skipped=True, skip_reason="no_tiles",
            ))
        continue
    log.info(f"  Catalog rows loaded: {len(catalog)}")

    host_id = row['HostHaloID']

    # CI galaxy selection — done once per event; sky maps saved to shared dir
    galaxies_in_CI, area_90 = find_galaxies_in_sky_and_distance_CI_healpix(
        ra_samp, dec_samp, dL_samp,
        catalog,
        host_halo_id=host_id,
        ci_level=CI_LEVEL,
        nside=1024,
        show_plot=True,
        event_id=inj_num,
        save_dir=SKY_MAP_DIR,
    )
    n_in_CI = len(galaxies_in_CI)
    log.info(f"  Galaxies in CI: {n_in_CI}  |  90% sky area: {area_90:.3f} deg²")

    # ── Inner loop: apply each magnitude cut independently ─────────────────────
    for m_lim in MAG_LIMITS:
        log.info(f"  --- mag cut: {APP_MAG_COL} <= {m_lim} ---")

        base_row = dict(
            inj_num     = inj_num,
            DL_true     = row['DL'],
            z_true      = row['z'],
            area_90     = area_90,
            n_in_CI     = n_in_CI,
            mag_limit   = m_lim,
            skipped     = False,
            skip_reason = "",
        )

        if n_in_CI == 0:
            stats_rows[m_lim].append({**base_row, **dict(
                n_after_cut=0, frac_kept=np.nan,
                med_app_mag=np.nan, med_abs_mag=np.nan,
                host_in_CI=False, host_survives_cut=False,
                host_app_mag=np.nan, host_abs_mag=np.nan,
            )})
            continue

        # Magnitude cut on the CI sample
        if APP_MAG_COL in galaxies_in_CI.colnames:
            mag_vals = np.array(galaxies_in_CI[APP_MAG_COL])
            mag_mask = mag_vals <= m_lim
            gals_cut = galaxies_in_CI[mag_mask]
        else:
            log.warning(f"  Column '{APP_MAG_COL}' not found — no mag cut applied.")
            gals_cut = galaxies_in_CI
            mag_mask = np.ones(n_in_CI, dtype=bool)

        n_cut     = len(gals_cut)
        frac_kept = n_cut / n_in_CI if n_in_CI > 0 else np.nan

        # Magnitude statistics on the cut sample
        med_app = float(np.median(gals_cut[APP_MAG_COL])) if n_cut > 0 and APP_MAG_COL in gals_cut.colnames else np.nan
        med_abs = float(np.median(gals_cut[ABS_MAG_COL])) if n_cut > 0 and ABS_MAG_COL in gals_cut.colnames else np.nan

        # Host galaxy survival through the magnitude cut
        ci_halo_ids      = np.array(galaxies_in_CI['HostHaloID'])
        host_in_CI       = bool(np.any(ci_halo_ids == host_id))
        host_surv_cut    = False
        host_app_mag     = np.nan
        host_abs_mag     = np.nan

        if host_in_CI and n_cut > 0:
            cut_halo_ids = np.array(gals_cut['HostHaloID'])
            host_match   = cut_halo_ids == host_id
            if np.any(host_match):
                host_surv_cut = True
                if APP_MAG_COL in gals_cut.colnames:
                    host_app_mag = float(gals_cut[APP_MAG_COL][host_match][0])
                if ABS_MAG_COL in gals_cut.colnames:
                    host_abs_mag = float(gals_cut[ABS_MAG_COL][host_match][0])

        log.info(
            f"    galaxies: {n_in_CI} -> {n_cut} ({frac_kept*100:.1f}% kept)"
            f" | host_in_CI={host_in_CI}, host_survives_cut={host_surv_cut}"
            f" | host: app_mag={host_app_mag:.2f}, abs_mag={host_abs_mag:.2f}"
        )

        stats_entry = {**base_row, **dict(
            n_after_cut       = n_cut,
            frac_kept         = frac_kept,
            med_app_mag       = med_app,
            med_abs_mag       = med_abs,
            host_in_CI        = host_in_CI,
            host_survives_cut = host_surv_cut,
            host_app_mag      = host_app_mag,
            host_abs_mag      = host_abs_mag,
        )}

        if n_cut == 0:
            log.info(f"    No galaxies after cut, skipping H0 computation.")
            stats_rows[m_lim].append(stats_entry)
            continue

        # H0 likelihood and posterior
        gw_likes = H0_likelihood(ra_samp, dec_samp, dL_samp, gals_cut, H0_GRID)
        like_H0  = H0_posterior(
            gw_likes, gals_cut, H0_GRID, df_silver,
            selection_label=SELECTION_EFF,
            luminosity_weight=False,
            self_chosen_luminosity_weight=False,
        )

        H0_dicts[m_lim][f"event_{inj_num}"] = like_H0
        stats_rows[m_lim].append(stats_entry)

log.info(f"\n{'='*60}")
log.info("All events processed. Saving results...")

# ── Save per-mag-limit results ──────────────────────────────────────────────────
for m_lim in MAG_LIMITS:
    out_dir  = os.path.join(OUTPUT_ROOT, f"mlim_{m_lim}")
    h0_dict  = H0_dicts[m_lim]

    # Joint posterior
    if h0_dict:
        joint = np.ones_like(H0_GRID)
        for like in h0_dict.values():
            joint *= like
        joint /= np.trapezoid(joint, H0_GRID)
        log.info(f"[mlim={m_lim}] {len(h0_dict)} events contributed to joint posterior.")
    else:
        joint = np.zeros_like(H0_GRID)
        log.warning(f"[mlim={m_lim}] No events contributed; joint posterior is zero.")

    # npz: H0_grid + joint + per-event posteriors
    npz_path = os.path.join(out_dir, "H0_likelihoods.npz")
    np.savez(npz_path, H0_grid=H0_GRID, joint_H0_posterior=joint, **h0_dict)
    log.info(f"[mlim={m_lim}] Saved {npz_path}")

    # Per-event statistics CSV
    csv_path = os.path.join(out_dir, "per_event_stats.csv")
    pd.DataFrame(stats_rows[m_lim]).to_csv(csv_path, index=False)
    log.info(f"[mlim={m_lim}] Saved {csv_path}")

    # H0 posterior plot for this magnitude cut
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, like in h0_dict.items():
        ax.plot(H0_GRID, like, alpha=0.4, lw=1, label=name)
    ax.plot(H0_GRID, joint, color="k", lw=2.5, label="Joint posterior")
    ax.axvline(70, color="gray", ls="--", lw=1, label=r"$H_0 = 70$")
    ax.set_xlabel(r"$H_0$ [km/s/Mpc]", fontsize=13)
    ax.set_ylabel(r"$p(H_0)$", fontsize=13)
    ax.set_title(
        f"Silver sirens — apparent mag cut: {APP_MAG_COL} $\leq$ {m_lim}"
        f"\n({len(h0_dict)} events, first {N_EVENTS} silver injections)",
        fontsize=12,
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "H0_posteriors.png"), dpi=300)
    plt.close(fig)
    log.info(f"[mlim={m_lim}] Saved H0_posteriors.png")

# ── Comparison plot: joint posteriors across all mag cuts ───────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["C0", "C1", "C2"]
for m_lim, color in zip(MAG_LIMITS, colors):
    npz   = np.load(os.path.join(OUTPUT_ROOT, f"mlim_{m_lim}", "H0_likelihoods.npz"))
    joint = npz["joint_H0_posterior"]
    n_ev  = len(H0_dicts[m_lim])
    ax.plot(H0_GRID, joint, color=color, lw=2,
            label=rf"$r \leq {m_lim}$ ({n_ev} events)")
ax.axvline(70, color="gray", ls="--", lw=1, label=r"$H_0 = 70$")
ax.set_xlabel(r"$H_0$ [km/s/Mpc]", fontsize=14)
ax.set_ylabel(r"$p(H_0)$", fontsize=14)
ax.set_title(
    f"Joint H0 posterior: apparent magnitude cut comparison\n"
    f"(first {N_EVENTS} silver injections, {APP_MAG_COL})",
    fontsize=13,
)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
fig.tight_layout()
cmp_path = os.path.join(OUTPUT_ROOT, "comparison_H0_posteriors.png")
fig.savefig(cmp_path, dpi=300)
plt.close(fig)
log.info(f"Saved comparison plot: {cmp_path}")
log.info("Done.")
