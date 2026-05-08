# H0_likelihood.py

import numpy as np
from scipy.stats import gaussian_kde
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from prior import prior_dl

# Cache cosmologies keyed by the H0 grid bytes — same grid across the run,
# so we build the FlatLambdaCDM list once.
_COSMO_CACHE = {}


def _cosmologies_for(H0_grid):
    key = H0_grid.tobytes()
    cached = _COSMO_CACHE.get(key)
    if cached is None:
        cached = [FlatLambdaCDM(H0=H0, Om0=0.3) for H0 in H0_grid]
        _COSMO_CACHE[key] = cached
    return cached


def H0_likelihood(
    ra_samples,
    dec_samples,
    dL_samples,
    galaxy_catalog,
    H0_grid=np.linspace(60, 80, 20)):
    """
    Compute normalized GW likelihood per galaxy and H0.

    Vectorized implementation: one cosmology call per H0 over all galaxies,
    and a single batched KDE evaluation for all (galaxy, H0) pairs.

    Returns
    -------
    array of shape (n_galaxies, n_H0)
    """
    samples = np.vstack([ra_samples, dec_samples, dL_samples])
    kde = gaussian_kde(samples)
    cosmologies = _cosmologies_for(H0_grid)

    N_gal = len(galaxy_catalog)
    N_H0  = len(H0_grid)

    z_gals   = np.array(galaxy_catalog['zcos'])
    ra_gals  = np.radians(np.array(galaxy_catalog['ra']))
    dec_gals = np.radians(np.array(galaxy_catalog['dec']))

    # One vectorized dL call per cosmology over all galaxies → (N_gal, N_H0)
    dL_matrix = np.column_stack([
        cosmo.luminosity_distance(z_gals).to(u.Mpc).value
        for cosmo in cosmologies
    ])

    # All KDE evaluation points in one shot: (3, N_H0 * N_gal)
    # Layout: first N_gal entries are all galaxies at H0[0], next block at H0[1], etc.
    ra_all  = np.tile(ra_gals, N_H0)
    dec_all = np.tile(dec_gals, N_H0)
    dL_all  = dL_matrix.T.ravel()       # (N_H0, N_gal) flattened row-major

    probs_all  = kde(np.vstack([ra_all, dec_all, dL_all]))
    probs_all /= prior_dl(dL_all)

    # Reshape to (N_gal, N_H0)
    return probs_all.reshape(N_H0, N_gal).T
