#!/usr/bin/env python

import os
import re
import glob
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
import pyarrow.parquet as pq
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from ligo.skymap.postprocess import find_greedy_credible_levels
from bilby.core.result import read_in_result
from tqdm import tqdm

from find_gal_in_CI_varying_H0 import find_galaxies_in_sky_and_distance_CI_healpix
from H0_likelihood import H0_likelihood
from H0_posterior import H0_posterior
from prior import prior_dl

# ── Configuration ──────────────────────────────────────────────────────────────
CATALOG_DIR  = "/hildafs/projects/phy220048p/share/Uchuu/z_0.5_healpix_catalog"
UCHUU_NSIDE  = 64   # nside of the per-pixel parquet files (49152 = 12 × 64²)

filename     = "/hildafs/projects/phy220048p/share/gwbench_dark_siren/bilby_result/Ucchuu_golden_injection_list_HsLsIs.csv"
base_folder  = "/hildafs/projects/phy220048p/share/gwbench_dark_siren/bilby_result/Ucchuu_golden_A#"
folder       = "Ucchuu_golden_results"
save_dir     = os.path.join(folder, "sky_maps")
h0_likelihood_output = os.path.join(folder, "H0_likelihoods.npz")
selection_effects    = "HLI#G"

H0_grid  = np.linspace(60, 80, 40)
ci_level = 0.9

# ── Load injection list (ra/dec in degrees) ────────────────────────────────────
df = pd.read_csv(filename)

# ── Save config ────────────────────────────────────────────────────────────────
os.makedirs(folder, exist_ok=True)
timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
config_file = os.path.join(folder, f"config_{timestamp}.txt")
with open(config_file, "w") as f:
    f.write("Configuration Log\n")
    f.write("=" * 60 + "\n")
    f.write(f"Timestamp        : {datetime.datetime.now().isoformat()}\n")
    f.write(f"Catalog          : Uchuu z<0.5 HEALPix ({CATALOG_DIR})\n")
    f.write(f"Input filename   : {filename}\n")
    f.write(f"Base folder      : {base_folder}\n")
    f.write(f"Output folder    : {folder}\n")
    f.write(f"Selection effects: {selection_effects}\n")
    f.write(f"H0 grid          : [{H0_grid[0]}, {H0_grid[-1]}], n={len(H0_grid)}\n")
print(f"Configuration saved to {config_file}")

# ── Find Bilby result files ────────────────────────────────────────────────────
def extract_inj_number(path):
    match = re.search(r'inj_(\d+)', path)
    return int(match.group(1)) if match else -1

result_files = sorted(
    glob.glob(f"{base_folder}/inj_*/bilby_inj_*_result.json"),
    key=extract_inj_number
)
print(f"Found {len(result_files)} Bilby result files.")

# ── Uchuu per-event catalog loader ─────────────────────────────────────────────
def load_uchuu_for_event(ra_samples, dec_samples, ci_level=0.9):
    """
    Load Uchuu parquet tiles covering the sky CI of one event.
    ra_samples/dec_samples are in radians (bilby posterior).
    Returns an astropy Table with stellar_mass column, or None if empty.
    """
    npix     = hp.nside2npix(2048)
    prob_map = np.zeros(npix)
    theta    = 0.5 * np.pi - dec_samples
    phi      = ra_samples
    np.add.at(prob_map, hp.ang2pix(2048, theta, phi), 1)
    prob_map /= prob_map.sum()

    cl          = find_greedy_credible_levels(prob_map)
    inside_high = np.where(cl <= ci_level)[0]

    # Downgrade nside=2048 CI pixels to nside=64 catalog tiles
    theta_c, phi_c = hp.pix2ang(2048, inside_high)
    cat_pixels = np.unique(hp.ang2pix(UCHUU_NSIDE, theta_c, phi_c))

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

# ── Main loop ──────────────────────────────────────────────────────────────────
H0_likelihoods_dict = {}
n_selected_list     = []
area_90_list        = []

for fname in tqdm(result_files, desc="Processing events"):
    inj_num = extract_inj_number(fname)
    print(f"\nProcessing injection {inj_num}: {fname}")

    row = df.iloc[inj_num]

    result    = read_in_result(fname)
    posterior = result.posterior

    ra_samples  = posterior["ra"].values                   # radians
    dec_samples = posterior["dec"].values                  # radians
    dL_samples  = posterior["luminosity_distance"].values  # Mpc

    # Load Uchuu tiles covering the sky CI for this event
    catalog = load_uchuu_for_event(ra_samples, dec_samples, ci_level=ci_level)
    if catalog is None or len(catalog) == 0:
        print(f"  No Uchuu galaxies loaded for injection {inj_num}, skipping.")
        continue

    # Find the injected host galaxy index by HostHaloID
    host_id    = row['HostHaloID']
    host_match = np.where(catalog['HostHaloID'] == host_id)[0]
    injected_idx = int(host_match[0]) if len(host_match) > 0 else None
    if injected_idx is None:
        print(f"  Warning: host HostHaloID={host_id} not found in loaded tiles for inj {inj_num}.")

    # Galaxy selection within sky + distance CI
    galaxies_in_CI, area_90 = find_galaxies_in_sky_and_distance_CI_healpix(
        ra_samples, dec_samples, dL_samples,
        catalog,
        injected_idx=injected_idx,
        ci_level=ci_level,
        show_plot=True,
        event_id=inj_num,
        save_dir=save_dir,
    )
    n_selected_list.append(len(galaxies_in_CI))
    area_90_list.append(area_90)

    if len(galaxies_in_CI) == 0:
        print(f"  No galaxies in CI for injection {inj_num}, skipping likelihood.")
        continue

    # H0 likelihood
    gw_likes = H0_likelihood(
        ra_samples, dec_samples, dL_samples,
        galaxies_in_CI, H0_grid,
    )

    # H0 posterior (no luminosity weighting — Uchuu has no gmag column)
    like_H0 = H0_posterior(
        gw_likes, galaxies_in_CI, H0_grid, df,
        selection_label=selection_effects,
        luminosity_weight=False,
        self_chosen_luminosity_weight=False,
    )

    H0_likelihoods_dict[f"event_{inj_num}"] = like_H0

# ── Joint posterior ────────────────────────────────────────────────────────────
if H0_likelihoods_dict:
    joint_H0_likelihood = np.ones_like(H0_grid)
    for like in H0_likelihoods_dict.values():
        joint_H0_likelihood *= like
    joint_H0_posterior = joint_H0_likelihood / np.trapz(joint_H0_likelihood, H0_grid)
else:
    print("No events processed successfully.")
    joint_H0_posterior = np.zeros_like(H0_grid)

np.savez(
    h0_likelihood_output,
    H0_grid=H0_grid,
    joint_H0_posterior=joint_H0_posterior,
    **H0_likelihoods_dict,
)
print(f"Saved H0 likelihoods to {h0_likelihood_output}")

# ── Update injection list with statistics ──────────────────────────────────────
processed_inj_nums = [extract_inj_number(f) for f in result_files[:len(n_selected_list)]]
s_n = pd.Series(n_selected_list, index=processed_inj_nums)
s_a = pd.Series(area_90_list,    index=processed_inj_nums)
df['n_selected_galaxies'] = df.index.map(s_n).fillna(0).astype(int)
df['area_90']             = df.index.map(s_a).fillna(0.0)
df.to_csv(filename, index=False)
print(f"Updated {filename} with n_selected_galaxies and area_90 columns.")

# ── Plot ───────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
for name, like in H0_likelihoods_dict.items():
    plt.plot(H0_grid, like, alpha=0.5, label=name)
plt.plot(H0_grid, joint_H0_posterior, color="k", lw=2, label="Joint Posterior")
plt.xlabel(r"$H_0$ [km/s/Mpc]")
plt.ylabel(r"$p(H_0)$")
plt.legend()
plt.grid()
plt.savefig(os.path.join(folder, "H0_posteriors.png"), dpi=300)
print("Saved H0_posteriors.png")
