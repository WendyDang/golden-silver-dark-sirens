# H0_likelihood.py

import numpy as np
from scipy.stats import gaussian_kde
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from tqdm import tqdm
from prior import prior_dl



def H0_likelihood(
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

    H0_likelihood = np.zeros((len(galaxy_catalog), len(H0_grid)))

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

        H0_likelihood[j, :] = probs

    return H0_likelihood
