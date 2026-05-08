#!/usr/bin/env python
"""
SDS_bilby_maglim.py

Tests how the depth of a volume-limited host-galaxy sample affects the H0
posterior from GW dark sirens.

For each event the GW dL posterior provides dL_high_90 = 95th-percentile of
the dL samples (far edge of the 90% dL CI).  Each survey-depth label m_lim
is converted into a fixed absolute-magnitude threshold:

    M_lim = m_lim - mu(dL_high_90),  where  mu = 5*log10(dL_Mpc) + 25

Galaxies in the CI are cut on MagStar_SDSSr <= M_lim — a single fixed cut
defining a volume-limited subcatalog.  The cut is H0-independent because a
real GW analysis only has dL, not z.  Different m_lim labels correspond to
volume-limited samples of different depth; the test asks how that depth
choice biases or tightens H0.

The matching beta(H0) is loaded from
gwbench/beta_maglim_tests/test1_vs_events/mlim_<m_lim>/, which builds beta
under the same H0-independent fixed-M_lim prescription.  Bin centres are
Planck15 dL at uniform z bin centres over [0, 0.5]; the event's dL_high_90
is digitised into one of these bins to select the matching beta file.

Usage
-----
python SDS_bilby_maglim.py --siren-type silver --n-events 20
python SDS_bilby_maglim.py --siren-type golden --n-events 10

Output layout  (default root: maglim_study_<siren-type>/)
-----------------
├── config_<timestamp>_job<id>.txt
├── run_<timestamp>_job<id>.log
├── sky_maps/
├── mlim_19.5/  mlim_21.5/  mlim_23.5/   (depth label)
│   ├── H0_likelihoods.npz
│   ├── H0_posteriors.png
│   └── per_event_stats.csv             (includes dL_high_90, M_lim per event)
└── comparison_H0_posteriors.png
"""

import os
import re
import glob
import argparse
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

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

# ── Command-line arguments ──────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BILBY_ROOT = "/hildafs/projects/phy220048p/share/gwbench_dark_siren/bilby_result"

_SIREN_CONFIGS = {
    "silver": dict(
        csv    = f"{_BILBY_ROOT}/Ucchuu_silver_injection_list_HsLsIs.csv",
        folder = f"{_BILBY_ROOT}/Ucchuu_silver_A#",
        sel    = "HLI#S",
    ),
    "golden": dict(
        csv    = f"{_BILBY_ROOT}/Ucchuu_golden_injection_list_HsLsIs.csv",
        folder = f"{_BILBY_ROOT}/Ucchuu_golden_A#",
        sel    = "HLI#G",
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument("--siren-type", choices=["silver", "golden"], default="silver")
parser.add_argument("--n-events",   type=int, default=20)
parser.add_argument("--output-root", default=None,
                    help="Output directory (default: maglim_study_<siren-type>/)")
args = parser.parse_args()

_cfg          = _SIREN_CONFIGS[args.siren_type]
INJ_CSV       = _cfg["csv"]
PE_FOLDER     = _cfg["folder"]
SELECTION_EFF = _cfg["sel"]
N_EVENTS      = args.n_events
OUTPUT_ROOT   = args.output_root or os.path.join(_SCRIPT_DIR, f"maglim_study_{args.siren_type}")

# ── Fixed configuration ─────────────────────────────────────────────────────────
CATALOG_DIR   = "/hildafs/projects/phy220048p/share/Uchuu/z_0.5_healpix_catalog"
UCHUU_NSIDE   = 64

# Apparent magnitude thresholds — must match mlim_* folders in the beta source
APP_MAG_LIMITS = [19.5, 21.5, 23.5]

H0_GRID       = np.linspace(60, 80, 40)
CI_LEVEL      = 0.9

# Absolute magnitude column used for the cut
ABS_MAG_COL   = "MagStar_SDSSr"   # total-stellar absolute SDSS-r

# ── Output directories ──────────────────────────────────────────────────────────
os.makedirs(OUTPUT_ROOT, exist_ok=True)
SKY_MAP_DIR = os.path.join(OUTPUT_ROOT, "sky_maps")
os.makedirs(SKY_MAP_DIR, exist_ok=True)
for _m in APP_MAG_LIMITS:
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
    _f.write("Magnitude-cut study (volume-limited test) — run configuration\n")
    _f.write("=" * 60 + "\n")
    _f.write(f"Timestamp        : {datetime.datetime.now().isoformat()}\n")
    _f.write(f"SLURM job ID     : {job_id}\n")
    _f.write(f"Catalog          : Uchuu z<0.5 HEALPix ({CATALOG_DIR})\n")
    _f.write(f"Siren type       : {args.siren_type}\n")
    _f.write(f"Injection CSV    : {INJ_CSV}\n")
    _f.write(f"PE folder        : {PE_FOLDER}\n")
    _f.write(f"N_EVENTS         : {N_EVENTS}\n")
    _f.write(f"App mag thresholds: {APP_MAG_LIMITS}\n")
    _f.write(f"Cut method       : M_lim = m_lim - mu(dL_high_90), cut on {ABS_MAG_COL}\n")
    _f.write(f"Beta source      : H0_posterior._load_beta_from_disk (test1_vs_events, dL-binned)\n")
    _f.write(f"Selection effects: {SELECTION_EFF}\n")
    _f.write(f"H0 grid          : [{H0_GRID[0]}, {H0_GRID[-1]}], n={len(H0_GRID)}\n")
    _f.write(f"CI level         : {CI_LEVEL}\n")
    _f.write(f"Output root      : {OUTPUT_ROOT}\n")
log.info(f"Config written to {config_path}")
log.info(f"Run log          : {log_path}")

# ── Load injection list ─────────────────────────────────────────────────────────
df_inj = pd.read_csv(INJ_CSV)
log.info(f"Loaded {args.siren_type} injection list: {len(df_inj)} events")

# ── Find and sort Bilby result files ────────────────────────────────────────────
def extract_inj_number(path):
    match = re.search(r'inj_(\d+)', path)
    return int(match.group(1)) if match else -1

result_files = sorted(
    glob.glob(f"{PE_FOLDER}/inj_*/bilby_inj_*_result.json"),
    key=extract_inj_number,
)[:N_EVENTS]

log.info(f"Using {len(result_files)} Bilby result files (first {N_EVENTS} by injection number).")
for _f in result_files:
    log.info(f"  {_f}")

# ── Uchuu tile loader ───────────────────────────────────────────────────────────
def load_uchuu_for_event(ra_samples, dec_samples, ci_level=0.9):
    """Load Uchuu parquet tiles covering the sky CI of one event."""
    _TILE_NSIDE = 512
    npix     = hp.nside2npix(_TILE_NSIDE)
    prob_map = np.zeros(npix)
    theta    = 0.5 * np.pi - dec_samples
    phi      = ra_samples
    np.add.at(prob_map, hp.ang2pix(_TILE_NSIDE, theta, phi), 1)
    prob_map /= prob_map.sum()

    cl          = find_greedy_credible_levels(prob_map)
    inside_high = np.where(cl <= ci_level)[0]

    theta_c, phi_c = hp.pix2ang(_TILE_NSIDE, inside_high)
    cat_pixels = np.unique(hp.ang2pix(UCHUU_NSIDE, theta_c, phi_c, nest=True))

    paths = [os.path.join(CATALOG_DIR, f"healpix_{pid:06d}.parquet")
             for pid in cat_pixels]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return None

    # pyarrow releases the GIL during read, so threads parallelise effectively.
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as ex:
        frames = list(ex.map(pd.read_parquet, paths))

    cat_df = pd.concat(frames, ignore_index=True)
    cat_df['stellar_mass'] = cat_df['MstarBulge'] + cat_df['MstarDisk']
    return Table.from_pandas(cat_df)

# ── Per-run data stores ──────────────────────────────────────────────────────────
H0_dicts   = {m: {} for m in APP_MAG_LIMITS}
stats_rows = {m: [] for m in APP_MAG_LIMITS}

# ── Main loop: one catalog load + CI selection per event ─────────────────────────
for fname in tqdm(result_files, desc="Events"):
    inj_num = extract_inj_number(fname)
    log.info(f"\n{'='*60}")
    log.info(f"Event inj_{inj_num}: {fname}")

    row = df_inj.iloc[inj_num]
    DL_true = row['DL']       # Mpc, kept only for diagnostics in stats CSV
    z_true  = row['z']        # truth, diagnostics only

    result    = read_in_result(fname)
    posterior = result.posterior
    ra_samp   = posterior["ra"].values
    dec_samp  = posterior["dec"].values
    dL_samp   = posterior["luminosity_distance"].values

    # 95th percentile of GW dL posterior — far edge of the 90% dL CI, used
    # to derive the fixed absolute-magnitude cut M_lim = m_lim - mu(dL_high_90).
    dL_high_90 = float(np.percentile(dL_samp, 95))
    mu_high    = 5.0 * np.log10(dL_high_90) + 25.0
    log.info(f"  DL_true={DL_true:.1f} Mpc, z_true={z_true:.4f}  |  "
             f"dL_high_90={dL_high_90:.1f} Mpc, mu={mu_high:.3f}")

    # Load Uchuu tiles — done once per event, reused for all mag cuts
    log.info(f"  Loading Uchuu tiles...")
    catalog = load_uchuu_for_event(ra_samp, dec_samp, ci_level=CI_LEVEL)
    if catalog is None or len(catalog) == 0:
        log.warning(f"  No Uchuu tiles loaded for inj_{inj_num}, skipping.")
        for m in APP_MAG_LIMITS:
            stats_rows[m].append(dict(
                inj_num=inj_num, DL_true=DL_true, z_true=z_true,
                dL_high_90=dL_high_90, mu_high=mu_high,
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

    # Pull arrays once outside the m_lim loop — used by every cut.
    abs_mag_vals = np.asarray(galaxies_in_CI[ABS_MAG_COL])
    ci_halo_ids  = np.asarray(galaxies_in_CI['HostHaloID'])
    host_in_CI   = bool(np.any(ci_halo_ids == host_id))

    # Compute the GW likelihood ONCE on the full CI sample. For each m_lim
    # cut we just index the rows by mag_mask — this is the dominant speedup.
    if n_in_CI > 0:
        gw_likes_full = H0_likelihood(ra_samp, dec_samp, dL_samp,
                                      galaxies_in_CI, H0_GRID)
    else:
        gw_likes_full = None

    # ── Inner loop: each m_lim defines a volume-limited subcatalog of depth M_lim ─
    for m_lim in APP_MAG_LIMITS:
        M_lim = m_lim - mu_high
        log.info(f"  --- app m_lim={m_lim}  =>  abs M_lim={M_lim:.3f}  (cut on {ABS_MAG_COL}) ---")

        base_row = dict(
            inj_num     = inj_num,
            DL_true     = DL_true,
            z_true      = z_true,
            dL_high_90  = dL_high_90,
            mu_high     = mu_high,
            area_90     = area_90,
            n_in_CI     = n_in_CI,
            app_mag_limit = m_lim,
            abs_mag_limit = M_lim,
            skipped     = False,
            skip_reason = "",
        )

        if n_in_CI == 0:
            stats_rows[m_lim].append({**base_row, **dict(
                n_after_cut=0, frac_kept=np.nan,
                med_abs_mag=np.nan,
                host_in_CI=False, host_survives_cut=False,
                host_abs_mag=np.nan,
            )})
            continue

        mag_mask = abs_mag_vals <= M_lim
        n_cut    = int(mag_mask.sum())
        frac_kept = n_cut / n_in_CI

        med_abs = float(np.median(abs_mag_vals[mag_mask])) if n_cut > 0 else np.nan

        host_match    = (ci_halo_ids == host_id) & mag_mask
        host_surv_cut = bool(np.any(host_match))
        host_abs_mag  = float(abs_mag_vals[host_match][0]) if host_surv_cut else np.nan

        log.info(
            f"    galaxies: {n_in_CI} -> {n_cut} ({frac_kept*100:.1f}% kept)"
            f" | host_in_CI={host_in_CI}, host_survives_cut={host_surv_cut}"
            f" | host abs_mag={host_abs_mag:.2f}"
        )

        stats_entry = {**base_row, **dict(
            n_after_cut       = n_cut,
            frac_kept         = frac_kept,
            med_abs_mag       = med_abs,
            host_in_CI        = host_in_CI,
            host_survives_cut = host_surv_cut,
            host_abs_mag      = host_abs_mag,
        )}

        if n_cut == 0:
            log.info(f"    No galaxies after cut, skipping H0 computation.")
            stats_rows[m_lim].append(stats_entry)
            continue

        # Index the precomputed full-CI likelihood by the mag-cut mask;
        # H0_posterior receives the cut catalog for its luminosity-weight branch.
        gw_likes = gw_likes_full[mag_mask]
        gals_cut = galaxies_in_CI[mag_mask]
        like_H0  = H0_posterior(
            gw_likes, gals_cut, H0_GRID, df_inj,
            selection_label=SELECTION_EFF,
            luminosity_weight=False,
            self_chosen_luminosity_weight=False,
            app_mag_lim=m_lim,
            dL_high_90=dL_high_90,
        )

        H0_dicts[m_lim][f"event_{inj_num}"] = like_H0
        stats_rows[m_lim].append(stats_entry)

log.info(f"\n{'='*60}")
log.info("All events processed. Saving results...")

# ── Credible-interval helper ────────────────────────────────────────────────────
def credible_interval(H0_grid, posterior, level=0.68):
    dH  = np.diff(H0_grid)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (posterior[:-1] + posterior[1:]) * dH)])
    cdf /= cdf[-1]
    tail = (1.0 - level) / 2.0
    lo  = np.interp(tail,        cdf, H0_grid)
    hi  = np.interp(1.0 - tail,  cdf, H0_grid)
    med = np.interp(0.5,         cdf, H0_grid)
    return lo, hi, med

# ── Save per-mag-limit results ──────────────────────────────────────────────────
for m_lim in APP_MAG_LIMITS:
    out_dir  = os.path.join(OUTPUT_ROOT, f"mlim_{m_lim}")
    h0_dict  = H0_dicts[m_lim]

    if h0_dict:
        joint = np.ones_like(H0_GRID)
        for like in h0_dict.values():
            joint *= like
        joint /= np.trapezoid(joint, H0_GRID)
        log.info(f"[mlim={m_lim}] {len(h0_dict)} events contributed to joint posterior.")
    else:
        joint = np.zeros_like(H0_GRID)
        log.warning(f"[mlim={m_lim}] No events contributed; joint posterior is zero.")

    npz_path = os.path.join(out_dir, "H0_likelihoods.npz")
    np.savez(npz_path, H0_grid=H0_GRID, joint_H0_posterior=joint, **h0_dict)
    log.info(f"[mlim={m_lim}] Saved {npz_path}")

    csv_path = os.path.join(out_dir, "per_event_stats.csv")
    pd.DataFrame(stats_rows[m_lim]).to_csv(csv_path, index=False)
    log.info(f"[mlim={m_lim}] Saved {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, like in h0_dict.items():
        ax.plot(H0_GRID, like, alpha=0.4, lw=1, label=name)
    ax.plot(H0_GRID, joint, color="k", lw=2.5, label="Joint posterior")

    if np.any(joint > 0):
        lo95, hi95, med = credible_interval(H0_GRID, joint, level=0.95)
        lo68, hi68, _   = credible_interval(H0_GRID, joint, level=0.68)
        ax.fill_between(H0_GRID, 0, joint,
                        where=(H0_GRID >= lo95) & (H0_GRID <= hi95),
                        color="k", alpha=0.10, label=f"95% CI [{lo95:.1f}, {hi95:.1f}]")
        ax.fill_between(H0_GRID, 0, joint,
                        where=(H0_GRID >= lo68) & (H0_GRID <= hi68),
                        color="k", alpha=0.20, label=f"68% CI [{lo68:.1f}, {hi68:.1f}]")
        ax.axvline(med, color="k", ls=":", lw=1.5, label=f"Median {med:.1f}")
        log.info(f"[mlim={m_lim}] Joint CI: median={med:.2f}, 68%=[{lo68:.2f},{hi68:.2f}], 95%=[{lo95:.2f},{hi95:.2f}]")

    ax.axvline(70, color="gray", ls="--", lw=1, label=r"$H_0 = 70$")
    ax.set_xlabel(r"$H_0$ [km/s/Mpc]", fontsize=13)
    ax.set_ylabel(r"$p(H_0)$", fontsize=13)
    ax.set_title(
        rf"$r_\mathrm{{app}} \leq {m_lim}$ → $M_\mathrm{{abs}} \leq m - \mu(D_L)$"
        f"\n(cut on {ABS_MAG_COL}, {len(h0_dict)} events)",
        fontsize=12,
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "H0_posteriors.png"), dpi=300)
    plt.close(fig)
    log.info(f"[mlim={m_lim}] Saved H0_posteriors.png")

# ── Comparison plot: joint posteriors across all app-mag thresholds ─────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["C0", "C1", "C2"]
for m_lim, color in zip(APP_MAG_LIMITS, colors):
    npz   = np.load(os.path.join(OUTPUT_ROOT, f"mlim_{m_lim}", "H0_likelihoods.npz"))
    joint = npz["joint_H0_posterior"]
    n_ev  = len(H0_dicts[m_lim])

    if np.any(joint > 0):
        lo95, hi95, med = credible_interval(H0_GRID, joint, level=0.95)
        lo68, hi68, _   = credible_interval(H0_GRID, joint, level=0.68)
        ax.fill_between(H0_GRID, 0, joint,
                        where=(H0_GRID >= lo95) & (H0_GRID <= hi95),
                        color=color, alpha=0.10)
        ax.fill_between(H0_GRID, 0, joint,
                        where=(H0_GRID >= lo68) & (H0_GRID <= hi68),
                        color=color, alpha=0.25)
        ax.axvline(med, color=color, ls=":", lw=1.2)
        label = rf"$m_\mathrm{{app}}\leq{m_lim}$ → $M_\mathrm{{abs}}\leq m-\mu$ ({n_ev} ev) — med={med:.1f}, 68%=[{lo68:.1f},{hi68:.1f}]"
    else:
        label = rf"$m_\mathrm{{app}}\leq{m_lim}$ (no events)"

    ax.plot(H0_GRID, joint, color=color, lw=2, label=label)

ax.axvline(70, color="gray", ls="--", lw=1, label=r"$H_0 = 70$")
ax.set_xlabel(r"$H_0$ [km/s/Mpc]", fontsize=14)
ax.set_ylabel(r"$p(H_0)$", fontsize=14)
ax.set_title(
    f"Volume-limited assumption test: H0 posterior vs. abs-mag depth\n"
    rf"(cut: $M_\mathrm{{abs}} \leq m_\mathrm{{app,lim}} - \mu(D_L)$, {ABS_MAG_COL}, first {N_EVENTS} {args.siren_type} events)",
    fontsize=12,
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
cmp_path = os.path.join(OUTPUT_ROOT, "comparison_H0_posteriors.png")
fig.savefig(cmp_path, dpi=300)
plt.close(fig)
log.info(f"Saved comparison plot: {cmp_path}")
log.info("Done.")
