# gw_likelihood.py

import numpy as np
from scipy.stats import gaussian_kde
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from tqdm import tqdm
from prior import prior_dl


# def gw_likelihood(
#     ra_samples,
#     dec_samples, 
#     dL_samples,
#     galaxy_catalog,
#     H0_grid=np.linspace(60, 80, 20)):
#     """
   
#     Parameters
#     ----------
#     ra_samples : array-like
#         RA posterior samples from GW data [radians].
#     dec_samples : array-like
#         Dec posterior samples from GW data [radians].
#     dL_samples : array-like
#         Luminosity distance posterior samples [Mpc].
#     galaxy_catalog : astropy Table or structured NumPy array
#         EM catalog with at least RA, Dec (degrees) and redshift columns.
   
#     Returns
#     -------
#     gw_likelihood : len(galaxy_catalog) x len(H0_grid) array
#         Normalized H0 likelihood for each galaxy in the catalog.
#     """
    

#     # Build KDEs
#     samples = np.vstack([ra_samples, dec_samples, dL_samples])
#     kde = gaussian_kde(samples)

    

#     gw_likelihood = np.zeros((len(galaxy_catalog),len(H0_grid)))

#     for j, galaxy in enumerate(tqdm(galaxy_catalog, desc="Galaxies")):
#         for i, H0 in enumerate(H0_grid):
#             cosmology = FlatLambdaCDM(H0=H0, Om0=0.3)
#             dL_gal = cosmology.luminosity_distance(galaxy['z_hetdex']).to(u.Mpc).value
#             p = kde([np.radians(galaxy['ra']), np.radians(galaxy['dec']), dL_gal])/prior_dl(dL_gal)
#             gw_likelihood[j, i] = p  # or whatever value you want to store
#         #gw_likelihood[j] /= np.sum(gw_likelihood[j])
#    return gw_likelihood

def gw_likelihood(
    ra_samples,
    dec_samples, 
    dL_samples,
    galaxy_catalog,
    H0_grid=np.linspace(60, 80, 20)):
    """
    Compute normalized GW likelihood per galaxy and H0.
    """
    # Build KDE
    samples = np.vstack([ra_samples, dec_samples, dL_samples])
    kde = gaussian_kde(samples)

    # Precompute cosmologies
    cosmologies = [FlatLambdaCDM(H0=H0, Om0=0.3) for H0 in H0_grid]

    gw_likelihood = np.zeros((len(galaxy_catalog), len(H0_grid)))

    for j, galaxy in enumerate(tqdm(galaxy_catalog, desc="Galaxies")):
        # Precompute distances for all H0 at once
        z = galaxy['z_hetdex']
        dL_gals = np.array([cosmo.luminosity_distance(z).to(u.Mpc).value for cosmo in cosmologies])

        # Stack points for KDE
        ras = np.full(len(H0_grid), np.radians(galaxy['ra']))
        decs = np.full(len(H0_grid), np.radians(galaxy['dec']))
        points = np.vstack([ras, decs, dL_gals])  # shape (3, len(H0_grid))

        # Evaluate in batch
        probs = kde(points) / prior_dl(dL_gals)

        gw_likelihood[j, :] = probs

    return gw_likelihood
