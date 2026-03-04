import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from prior import prior_dl
from astropy.cosmology import FlatLambdaCDM
import healpy as hp
from ligo.skymap.postprocess import find_greedy_credible_levels
import os
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
import ligo.skymap.plot

# Define H0 scan range
H0_values = np.linspace(60, 80, 40)  # km/s/Mpc

#USE find_galaxies_in_sky_and_distance_CI_healpix

def find_galaxies_in_sky_and_distance_CI(ra_samples, dec_samples, dL_samples, em_catalog,
                                         ci_level=0.9, show_plot=False):
    """
    Select galaxies within:
    1) Sky area 90% CI from GW posterior (RA, Dec).
    2) Luminosity distance 90% CI, allowing H0 to vary from 60–80 km/s/Mpc.

    Parameters
    ----------
    ra_samples : array-like
        GW posterior samples for RA (in radians).
    dec_samples : array-like
        GW posterior samples for Dec (in radians).
    dL_samples : array-like
        GW posterior samples for luminosity distance (in Mpc).
    em_catalog : astropy.table.Table or structured array
        Must contain 'ra', 'dec' (deg) and 'z_hetdex' columns.
    ci_level : float
        Credible interval level (default 0.9).
    show_plot : bool
        Whether to plot sky selection.

    Returns
    -------
    galaxies_selected : same type as em_catalog
        Galaxies meeting both criteria.
    """

    # -------------------------------
    # Step 1: Sky 90% CI selection
    # -------------------------------
    sky_samples = np.vstack([ra_samples, dec_samples])
    sky_kde = gaussian_kde(sky_samples)

    ra_gal = np.radians(em_catalog['ra'])
    dec_gal = np.radians(em_catalog['dec'])
    sky_positions = np.vstack([ra_gal, dec_gal])
    sky_post = sky_kde(sky_positions)

    sorted_sky = np.sort(sky_post)[::-1]
    cdf_sky = np.cumsum(sorted_sky) / np.sum(sorted_sky)
    sky_thresh = sorted_sky[np.searchsorted(cdf_sky, ci_level)]
    inside_sky = sky_post >= sky_thresh

    # -------------------------------
    # Step 2: Luminosity distance 90% CI selection (variable H0)
    # -------------------------------
    low_dl, high_dl = np.percentile(dL_samples, [(1 - ci_level) / 2 * 100,
                                                 (1 + ci_level) / 2 * 100])

    z_gal = em_catalog['z_hetdex']

    def in_dl_CI_for_any_H0(z):
        for H0 in H0_values:
            cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)
            dL_val = cosmo.luminosity_distance(z).to('Mpc').value
            if low_dl <= dL_val <= high_dl:
                return True
        return False

    inside_dl = np.array([in_dl_CI_for_any_H0(z) for z in z_gal])

    # -------------------------------
    # Step 3: Combine selections
    # -------------------------------
    final_selection = inside_sky & inside_dl
    galaxies_selected = em_catalog[final_selection]

    if show_plot:
        plt.figure(figsize=(7, 6))
        plt.scatter(em_catalog['ra'], em_catalog['dec'], s=1, alpha=0.3, label='All galaxies')
        plt.scatter(np.degrees(ra_gal[final_selection]), np.degrees(dec_gal[final_selection]),
                    s=10, color='red', label='Selected')
        plt.scatter(np.degrees(ra_samples), np.degrees(dec_samples), s=1, alpha=0.2,
                    color='gray', label='GW posterior')
        plt.xlabel('RA [deg]')
        plt.ylabel('Dec [deg]')
        plt.title(f'{int(ci_level * 100)}% CI: Sky + Distance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print(f"Number of galaxies inside {int(ci_level * 100)}% CI for sky + distance: {np.sum(final_selection)}")
    return galaxies_selected


import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM


cosmos = [FlatLambdaCDM(H0=H0, Om0=0.3) for H0 in H0_values]


def find_galaxies_in_sky_and_distance_CI_fast(
    ra_samples, dec_samples, dL_samples, em_catalog,
    ci_level=0.9, show_plot=False, grid_size=400
):
    """
    Accelerated version:
    1) Precompute sky KDE on grid, then interpolate for galaxies.
    2) Vectorize luminosity distance calculation over H0.

    Parameters
    ----------
    ra_samples, dec_samples : array-like
        GW posterior samples for RA, Dec (in radians).
    dL_samples : array-like
        GW posterior samples for luminosity distance (Mpc).
    em_catalog : astropy.table.Table
        Must contain 'ra', 'dec' (deg) and 'z_hetdex'.
    ci_level : float
        Credible interval level (default 0.9).
    show_plot : bool
        Plot results if True.
    grid_size : int
        Resolution of KDE grid (default 200).

    Returns
    -------
    galaxies_selected : same type as em_catalog
        Galaxies meeting both criteria.
    """

    # -------------------------------
    # Step 1: Sky 90% CI selection (grid KDE)
    # -------------------------------
    sky_samples = np.vstack([ra_samples, dec_samples])
    sky_kde = gaussian_kde(sky_samples)

    # Build grid
    ra_grid = np.linspace(ra_samples.min(), ra_samples.max(), grid_size)
    dec_grid = np.linspace(dec_samples.min(), dec_samples.max(), grid_size)
    RA, DEC = np.meshgrid(ra_grid, dec_grid)
    grid_points = np.vstack([RA.ravel(), DEC.ravel()])

    # Evaluate KDE on grid and normalize
    grid_vals = sky_kde(grid_points).reshape(RA.shape)

    # Flatten, sort for threshold
    sorted_vals = np.sort(grid_vals.ravel())[::-1]
    cdf = np.cumsum(sorted_vals) / np.sum(sorted_vals)
    sky_thresh = sorted_vals[np.searchsorted(cdf, ci_level)]

    # Interpolator for galaxies
    interp = RegularGridInterpolator((dec_grid, ra_grid), grid_vals, bounds_error=False, fill_value=0.0)

    ra_gal = np.radians(em_catalog['ra'])
    dec_gal = np.radians(em_catalog['dec'])
    gal_positions = np.vstack([dec_gal, ra_gal]).T
    sky_post = interp(gal_positions)
    inside_sky = sky_post >= sky_thresh

    # -------------------------------
    # Step 2: Luminosity distance CI selection (vectorized)
    # -------------------------------
    low_dl, high_dl = np.percentile(
        dL_samples, [(1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100]
    )

    z_gal = em_catalog['z_hetdex']
    inside_dl = np.zeros_like(z_gal, dtype=bool)

    # Vectorized loop over cosmologies
    for cosmo in cosmos:
        dL_vals = cosmo.luminosity_distance(z_gal).to('Mpc').value
        inside_dl |= (low_dl <= dL_vals) & (dL_vals <= high_dl)

    # -------------------------------
    # Step 3: Combine selections
    # -------------------------------
    final_selection = inside_sky & inside_dl
    galaxies_selected = em_catalog[final_selection]

    if show_plot:
        plt.figure(figsize=(7, 6))
        plt.scatter(em_catalog['ra'], em_catalog['dec'], s=1, alpha=0.3, label='All galaxies')
        plt.scatter(em_catalog['ra'][final_selection], em_catalog['dec'][final_selection],
                    s=10, color='red', label='Selected')
        plt.scatter(np.degrees(ra_samples), np.degrees(dec_samples),
                    s=1, alpha=0.2, color='gray', label='GW posterior')
        plt.xlabel('RA [deg]')
        plt.ylabel('Dec [deg]')
        plt.title(f'{int(ci_level*100)}% CI: Sky + Distance (accelerated)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print(f"Number of galaxies inside {int(ci_level*100)}% CI for sky + distance: {np.sum(final_selection)}")
    return galaxies_selected

def find_galaxies_in_sky_and_distance_CI_healpix(
    ra_samples, dec_samples, dL_samples, em_catalog,
    ci_level=0.9, show_plot=False, nside=2048,injected_idx=None,event_id=None, save_dir=None
):
    """
    Accelerated version:
    1) Precompute sky KDE on grid, then interpolate for galaxies.
    2) Vectorize luminosity distance calculation over H0.

    Parameters
    ----------
    ra_samples, dec_samples : array-like
        GW posterior samples for RA, Dec (in radians).
    dL_samples : array-like
        GW posterior samples for luminosity distance (Mpc).
    em_catalog : astropy.table.Table
        Must contain 'ra', 'dec' (deg) and 'z_hetdex'.
    ci_level : float
        Credible interval level (default 0.9).
    show_plot : bool
        Plot results if True.
    grid_size : int
        Resolution of KDE grid (default 200).

    Returns
    -------
    galaxies_selected : same type as em_catalog
        Galaxies meeting both criteria.
    """
    from astropy.cosmology import FlatLambdaCDM
    from astropy.cosmology import z_at_value
    import astropy.units as u

    # --- Step 1: Make HEALPix probability map from posterior samples ---
    npix = hp.nside2npix(nside)
    prob_map = np.zeros(npix)

    # Convert to healpy angles
    theta = 0.5 * np.pi - dec_samples  # colatitude
    phi = ra_samples                   # longitude
    pix_idx = hp.ang2pix(nside, theta, phi)

    # Bin samples into pixels
    for p in pix_idx:
        prob_map[p] += 1
    prob_map /= prob_map.sum()

    # Compute credible levels
    credible_levels = find_greedy_credible_levels(prob_map)
    # Area of pixels with credible level <= 0.9 (90%)
    pix_area_deg2 = hp.nside2pixarea(nside, degrees=True)
    area_90 = pix_area_deg2 * np.count_nonzero(credible_levels <= 0.9)
    print(f"90% credible area: {area_90:.2f} deg²")
    # Mask pixels inside 90% CI


    # --- Step 2: Sky selection for galaxies ---
    # Mask pixels inside 90% CI
    inside_mask = credible_levels <= ci_level

    # Get pixel indices for all galaxies
    theta_gal = 0.5 * np.pi - np.radians(em_catalog['dec'])
    phi_gal = np.radians(em_catalog['ra'])
    pix_gal = hp.ang2pix(nside, theta_gal, phi_gal)

    # Select galaxies inside 90% region
    inside_sky = inside_mask[pix_gal]
    sel_pix = np.unique(pix_gal[inside_sky])
    sel_area = pix_area_deg2 * len(sel_pix)
    print("Area covered by selected galaxies:", sel_area, "deg²")
    # --- Check whether injected sky position falls inside the HEALPix sky mask ---
    # if injected_ra is not None and injected_dec is not None:
    #     theta_inj = 0.5 * np.pi - np.radians(injected_dec)
    #     phi_inj = np.radians(injected_ra)
    #     pix_inj = hp.ang2pix(nside, theta_inj, phi_inj)
    #     inj_in_inside_sky = bool(inside_mask[pix_inj])
    #     print(f"Injected (deg): {injected_ra:.6f}, {injected_dec:.6f} -> pix {pix_inj}, inside_sky: {inj_in_inside_sky}")




    # -------------------------------
    # Step 2: Luminosity distance CI selection (vectorized)
    # -------------------------------
    low_dl, high_dl = np.percentile(
        dL_samples, [(1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100]
    )
    # low_dl, high_dl = np.percentile(
    #     dL_samples, [(1 - 0.99) / 2 * 100, (1 + 0.99) / 2 * 100]
    # )
    H0_range = np.linspace(60, 80, 21)  # km/s/Mpc
    Om0 = 0.3

# --- Compute the redshift limits corresponding to dL_CI across H0 range ---
    z_min_list, z_max_list = [], []
    for H0 in H0_values:
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        # Invert luminosity distance to find z at low_dl and high_dl
        z_min = z_at_value(cosmo.luminosity_distance, low_dl * u.Mpc,zmax=1)
        z_min_list.append(z_min)
        z_max = z_at_value(cosmo.luminosity_distance, high_dl * u.Mpc,zmax=1)
        z_max_list.append(z_max)

    # Take the broadest possible z-range allowed by any cosmology
    z_min_all = np.min(z_min_list)
    z_max_all = np.max(z_max_list)

    # --- Apply selection ---
    z_gal = em_catalog["z_hetdex"]
    inside_dl = (z_gal >= z_min_all) & (z_gal <= z_max_all)

    # print(f"90% CI distance corresponds to z ∈ [{z_min_all:.4f}, {z_max_all:.4f}] over H0∈[60,80]")



    # -------------------------------
    # Step 3: Combine selections
    # -------------------------------
    final_selection = inside_sky & inside_dl
    galaxies_selected = em_catalog[final_selection]
    
    if injected_idx is not None and 0 <= injected_idx < len(em_catalog):
        survived_sky = inside_sky[injected_idx]
        survived_dl  = inside_dl[injected_idx]
        survived_final = final_selection[injected_idx]

    if survived_final:
        print(f"✅ Injected galaxy at index {injected_idx} SURVIVED all cuts.")
    else:
        print(f"❌ Injected galaxy at index {injected_idx} FAILED selection.")
        if not survived_sky and not survived_dl:
            print("   → Failed both sky and distance cuts.")
            print(f"90% CI distance corresponds to z ∈ [{z_min_all:.4f}, {z_max_all:.4f}] over H0∈[60,80]")

        elif not survived_sky:
            print("   → Failed the sky localization cut.")
        elif not survived_dl:
            print("   → Failed the luminosity distance cut.")
            print(f"90% CI distance corresponds to z ∈ [{z_min_all:.4f}, {z_max_all:.4f}] over H0∈[60,80]")
            print(f"Injected galaxy z: {em_catalog['z_hetdex'][injected_idx]:.4f}")
        else:
            print(f"⚠️ Invalid injected_idx ({injected_idx}) or out of bounds.")





    if show_plot:

    # --- SkyCoord for galaxies ---
        gal_all = SkyCoord(ra=em_catalog['ra'] * u.deg, dec=em_catalog['dec'] * u.deg)
        gal_sel = SkyCoord(ra=em_catalog['ra'][final_selection] * u.deg,
                        dec=em_catalog['dec'][final_selection] * u.deg)
        gal_true = SkyCoord(ra=em_catalog['ra'][injected_idx] * u.deg, dec=em_catalog['dec'][injected_idx] * u.deg)

        z_sel = em_catalog['z_hetdex'][final_selection]
        ra_center = np.median(ra_samples)
        dec_center = np.median(dec_samples)
        center_coord = SkyCoord(ra=ra_center * u.rad, dec=dec_center * u.rad)
        gal_all = SkyCoord(ra=em_catalog['ra'] * u.deg, dec=em_catalog['dec'] * u.deg)
        gal_sel = SkyCoord(ra=em_catalog['ra'][final_selection] * u.deg,
                           dec=em_catalog['dec'][final_selection] * u.deg)
        
        
        



        from astropy.cosmology import FlatLambdaCDM

        # --- Cosmology ---
        H0 = 70  # km/s/Mpc
        Om0 = 0.3
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

        # --- Galaxy redshifts ---
        z_gal = em_catalog['z_hetdex'][final_selection]
        #print("z_selected:", z_gal)

        # # --- Convert galaxy redshifts to luminosity distance (Mpc) ---
        # DL_gal = cosmo.luminosity_distance(z_gal).to('Mpc').value

        # --- Compute closeness relative to injected distance ---
        closeness = np.abs((z_gal - em_catalog['z_hetdex'][injected_idx]) / em_catalog['z_hetdex'][injected_idx]) * 100
        



        # --- Create figure ---
        fig = plt.figure(figsize=(8, 7))

        ax = plt.axes(projection='astro zoom', center=center_coord, radius='0.4 deg')

        # --- Sky map (probability map as grayscale background) ---
        ax.imshow_hpx(prob_map, cmap='cylon', alpha=0.8, label=r'$90\%$ credible region')
        ax.grid()

        # --- Plot all galaxies (gray) ---
        ax.scatter(
            gal_all.ra, gal_all.dec,
            transform=ax.get_transform('world'),
            color='gray', s=1, alpha=0.3, label='All galaxies'
        )
        ax.scatter(gal_true.ra, gal_true.dec,
                   color='green', s=200, marker='*', label='Injected Galaxy',
                   transform=ax.get_transform('world'))

        # --- Plot selected galaxies color-coded by redshift ---
        sc = ax.scatter(
            gal_sel.ra, gal_sel.dec,
            c=closeness, cmap='plasma', s=15, alpha=0.8,
            transform=ax.get_transform('world'),
            label='Selected galaxies'
        )

        # --- Colorbar ---
        cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label(r'$\Delta z / z~(\%)$', fontsize=16)

        cbar.ax.tick_params(labelsize=16)

        # --- Labels & title ---
        ax.coords['ra'].set_axislabel('RA', fontsize=16)
        ax.coords['dec'].set_axislabel('Dec', fontsize=16)
        ax.coords['ra'].set_ticklabel(size=16)
        ax.coords['dec'].set_ticklabel(size=16)

        ax.legend(loc='upper right', fontsize=16)
        

        plt.tight_layout()

        # --- Save or show ---
        if save_dir is not None and event_id is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"event_{event_id}_sky_map.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()


    print(f"Number of galaxies inside {int(ci_level*100)}% CI for sky + distance: {np.sum(final_selection)}")
    return galaxies_selected,area_90