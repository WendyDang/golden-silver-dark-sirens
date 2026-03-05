#!/usr/bin/env python


import pandas as pd
import numpy as np
from find_gal_in_CI_varying_H0 import find_galaxies_in_sky_and_distance_CI_healpix
import os

# --- Read data ---
from astropy.io import fits
from astropy.table import Table
# Choose catalog by setting `catalog_choice` to one of the keys below.
# For a one-button change, edit the value of `catalog_choice`.
catalog_choice = "COSMOS"  
_catalog_paths = {
    # input your choice of catalog
}

catalog_path = _catalog_paths.get(catalog_choice.upper())
if catalog_path is None:
    raise ValueError(f"Unknown catalog_choice: {catalog_choice}. Valid options: {list(_catalog_paths.keys())}")

catalog = Table.read(catalog_path, format='fits')

filename = "./gwbench_GDS_HLI#_HET.txt"
folder = "GDS_sharp_HLI" #to store the results 
save_dir = "./" + folder + "/sky_map_cosmos" #store skymaps if saved any
base_folder = "./Bilby_automate/HLI#_golden_PE_relative" # where bilby results are stored
selection_effects= 'HLI#G'  # change to "default" or one of the keys in beta_dict as needed, see H0_posterior
h0_likelihood_output = "./" + folder + "/H0_likelihoods_sf.npz" #output 
df = pd.read_csv(filename, sep="\t")

# --- Save configuration to text file for record-keeping ---
import datetime
os.makedirs(folder, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
config_file = os.path.join(folder, f"config_{timestamp}.txt")
with open(config_file, "w") as f:
    f.write(f"Configuration Log\n")
    f.write(f"{'='*60}\n")
    f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
    f.write(f"Catalog Choice: {catalog_choice}\n")
    f.write(f"Catalog Path: {catalog_path}\n")
    f.write(f"Input Filename: {filename}\n")
    f.write(f"Output Folder: {folder}\n")
    f.write(f"Save Directory: {save_dir}\n")
    f.write(f"Base Folder (Bilby Results): {base_folder}\n")
    f.write(f"Selection Effects Label: {selection_effects}\n")
    f.write(f"H0 Likelihood Output: {h0_likelihood_output}\n")
    
print(f"✅ Configuration saved to {config_file}")

H0_grid = np.linspace(60, 80, 40)

import glob
from bilby.core.result import read_in_result

# result_files = sorted(glob.glob(f"{base_folder}/inj_*/bilby_inj_*_result.json"))
import re

# Sort numerically based on the injection number
def extract_inj_number(path):
    match = re.search(r'inj_(\d+)', path)
    return int(match.group(1)) if match else -1

result_files = sorted(
    glob.glob(f"{base_folder}/inj_*/bilby_inj_*_result.json"),
    key=extract_inj_number
)


# or 10**(...) if base-10 log

# samples_all[event, sample, param_index]
# param_index: 0=ra, 1=dec, 2=log_DL



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from H0_posterior import H0_posterior
from prior import prior_dl
from H0_likelihood import H0_likelihood
from tqdm import tqdm


# Your inputs


ci_level = 0.9

# --- Prepare storage ---
H0_likelihoods_dict = {}
n_selected_list = []
area_90_list = []


# --- Loop through each event ---
for i, fname in tqdm(enumerate(result_files), total=len(result_files), desc="Processing events"):
    print(f"Processing {fname}...")
    if df['DL'][i] < 980:

        result = read_in_result(fname)
        posterior = result.posterior

        ra_samples = posterior["ra"].values
        dec_samples = posterior["dec"].values
        dL_samples = posterior["luminosity_distance"].values

        # --- 1. Shift COSMOS field to event injection position ---
        catalog_shifted = catalog.copy()
        # position of galaxy with the closest dL to the injection
        injected_idx = df['idx_COSMOS'][i]
        current_ra_center = catalog['ra'][injected_idx]
        current_dec_center = catalog['dec'][injected_idx]
        current_z_center = catalog['z_hetdex'][injected_idx]


    
        # position of event injection
        new_ra_center = np.degrees(df['ra'][i])   # radians → deg
        new_dec_center = np.degrees(df['dec'][i]) # radians → deg
        ra_offset = new_ra_center - current_ra_center
        dec_offset = new_dec_center - current_dec_center

        catalog_shifted['ra'] = catalog['ra'] + ra_offset
        catalog_shifted['dec'] = catalog['dec'] + dec_offset
        
        # --- Check if shifted dec is out of bounds ---
        min_dec_before = np.min(catalog_shifted['dec'])
        max_dec_before = np.max(catalog_shifted['dec'])
        out_of_bounds_before = np.sum((catalog_shifted['dec'] < -90) | (catalog_shifted['dec'] > 90))
        
        # --- Clip RA to [0, 360) and Dec to [-90, 90] ---
        catalog_shifted['ra'] = np.mod(catalog_shifted['ra'], 360.0)  # Wrap RA to [0, 360)
        catalog_shifted['dec'] = np.clip(catalog_shifted['dec'], -90, 90)  # Clip Dec to [-90, 90]
        
        min_dec_after = np.min(catalog_shifted['dec'])
        max_dec_after = np.max(catalog_shifted['dec'])
        out_of_bounds_after = np.sum((catalog_shifted['dec'] < -90) | (catalog_shifted['dec'] > 90))
        
        if out_of_bounds_before > 0:
            print(f"⚠️ WARNING Event {i}: {out_of_bounds_before} galaxies had dec out of bounds!")
            print(f"   Original dec range: [{np.min(catalog['dec']):.2f}, {np.max(catalog['dec']):.2f}]")
            print(f"   Dec offset applied: {dec_offset:.2f}°")
            print(f"   Before clipping: [{min_dec_before:.2f}, {max_dec_before:.2f}]")
            print(f"   After clipping: [{min_dec_after:.2f}, {max_dec_after:.2f}]")
            print(f"   ✅ Clipped {out_of_bounds_before} galaxies to valid range.")
        
        print('Shifted COSMOS field center coordinates:', catalog_shifted['ra'][injected_idx], catalog_shifted['dec'][injected_idx])
        print('Bilby injection coordinates:', new_ra_center, new_dec_center)
 
        # --- 3. Find galaxies consistent with sky + distance ---
        galaxies_in_CI_sky_distance,area_90 = find_galaxies_in_sky_and_distance_CI_healpix(
            ra_samples, dec_samples, dL_samples,
            catalog_shifted,
            injected_idx=injected_idx,
            ci_level=ci_level,
            show_plot=True,
            event_id=i,
            save_dir=save_dir  # keep False for batch, True if debugging
        )
        n_selected = len(galaxies_in_CI_sky_distance)
        n_selected_list.append(n_selected)
        area_90_list.append(area_90)
        
        # --- 4. Compute GW likelihoods ---
        gw_likes = H0_likelihood(
            ra_samples, dec_samples, dL_samples,
            galaxies_in_CI_sky_distance, H0_grid
        )
        

        # --- 5. Convert into H0 likelihood ---
        like_H0 = H0_posterior(
            gw_likes, galaxies_in_CI_sky_distance, H0_grid, df,
            selection_label=selection_effects,
            gmag_key='gmag',
            luminosity_weight=False,
            self_chosen_luminosity_weight=False
            
        )

        # Store result
        H0_likelihoods_dict[f"event_{i}"] = like_H0
    else:
        print(f"Skipping event {i} due to no host found.")
        continue

# --- Joint posterior across all events ---
joint_H0_likelihood = np.ones_like(H0_grid)
for like in H0_likelihoods_dict.values():
    joint_H0_likelihood *= like

joint_H0_posterior = joint_H0_likelihood / np.trapz(joint_H0_likelihood, H0_grid)

import numpy as np

np.savez(
    h0_likelihood_output,
    H0_grid=H0_grid,
    joint_H0_posterior=joint_H0_posterior,
    **H0_likelihoods_dict
)
print(f"Saved H0 likelihoods to {h0_likelihood_output}")

# safe assign: truncate if lists longer than df, pad with zeros if shorter
# ...existing code...
# safe assign: truncate if lists longer than df, pad with zeros if shorter
s_n = pd.Series(n_selected_list)
s_a = pd.Series(area_90_list)
if len(s_n) > len(df): s_n = s_n.iloc[:len(df)]
if len(s_a) > len(df): s_a = s_a.iloc[:len(df)]
df['n_selected_galaxies'] = s_n.reindex(df.index, fill_value=0).astype(int)
df['area_90'] = s_a.reindex(df.index, fill_value=0.0)

# save and exit
df.to_csv(filename, sep="\t", index=False)
print(f"✅ Added n_selected_galaxies column and saved to {filename}")
# --- Plot result ---
plt.figure(figsize=(10, 6))
for name, like in H0_likelihoods_dict.items():
    plt.plot(H0_grid, like, alpha=0.5, label=name)

plt.plot(H0_grid, joint_H0_posterior, color="k", lw=2, label="Joint Posterior")
plt.xlabel(r"$H_0$ [km/s/Mpc]")
plt.ylabel(r"$p(H_0)$")
plt.legend()
plt.grid()
plt.savefig("./"+ folder +"/H0_posteriors.png", dpi=300)
plt.show()
