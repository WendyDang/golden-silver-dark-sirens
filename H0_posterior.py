import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

M_sun_g    = 5.12
_ref_cosmo = FlatLambdaCDM(H0=70, Om0=0.3)   # reference cosmology for lum-weight branch


def H0_posterior(
    H0_likelihood,
    galaxy_catalog,
    H0_grid,
    df,
    selection_label,
    gmag_key='gmag',
    luminosity_weight=False,
    self_chosen_luminosity_weight=False):
    """
    Compute the H0 posterior by marginalizing over host galaxies,
    optionally with luminosity weighting.
    """
    H0_beta = np.linspace(60, 80, 10)

    beta_dict = {
        "HLI#G": np.array([
            0.03337349, 0.03872403, 0.04146044, 0.04433673, 0.04783259,
            0.05041423, 0.05641019, 0.05719307, 0.06195887, 0.06996622
        ]),
        "HLI#S": np.array([
            0.04333718, 0.04596151, 0.04681771, 0.04829200, 0.04907971,
            0.05044047, 0.05233983, 0.05298423, 0.05454983, 0.05573224
        ]),
        "HLV+S": np.array([
            0.03707336, 0.04041492, 0.04229452, 0.04528477, 0.04872514,
            0.05082453, 0.05479356, 0.05700167, 0.05979213, 0.06466416
        ]),
        'HLI#S, COSMOS': np.array([
            0.04113569, 0.045531,   0.04665237, 0.04826579, 0.05014428,
            0.04926973, 0.0550313,  0.05380664, 0.05257146, 0.05631918
        ]),
        'HLI#S, SHELA, 0.5': np.array([
            0.04107071, 0.0430606,  0.04590545, 0.0464678,  0.04920621,
            0.051407,   0.05396409, 0.05408589, 0.05603155, 0.05867211
        ]),
    }

    # --- Selection function ---
    if selection_label == "default":
        beta_H0 = H0_grid ** 3
        beta_H0 = beta_H0 / np.trapezoid(beta_H0, H0_grid)
    else:
        if selection_label not in beta_dict:
            raise ValueError(f"Unknown selection_label: {selection_label}")
        beta_vals = beta_dict[selection_label]
        beta_vals = beta_vals / np.trapezoid(beta_vals, H0_beta)
        beta_interp = interp1d(
            H0_beta, np.log(beta_vals),
            kind="linear", bounds_error=False, fill_value="extrapolate",
        )
        beta_H0 = np.exp(beta_interp(H0_grid))

    # --- Luminosity weights ---
    if luminosity_weight:
        # Vectorized: compute absolute magnitude and luminosity for all galaxies
        gmag_vals = np.array(galaxy_catalog[gmag_key])
        z_vals    = np.array(galaxy_catalog['zcos'])
        dL_pc     = _ref_cosmo.luminosity_distance(z_vals).to('Mpc').value * 1e6
        M_g       = gmag_vals - 5 * (np.log10(dL_pc) - 1)
        weights   = 10 ** (-0.4 * (M_g - M_sun_g))

    elif self_chosen_luminosity_weight:
        # Empirical luminosity weighting from the injected host population
        mask      = df["host_found"] == 1
        M_abs_inj = df["abs_mag_nearest"][mask].values
        L_inj     = 10 ** (-0.4 * (M_abs_inj - M_sun_g))
        kde       = gaussian_kde(np.log10(L_inj), bw_method='scott')

        M_abs_gal = np.array(galaxy_catalog['mag_abs'])
        L_gal     = 10 ** (-0.4 * (M_abs_gal - M_sun_g))
        pdf_vals  = kde(np.log10(L_gal)).ravel()
        pdf_vals  = np.maximum(pdf_vals, 1e-12)
        pdf_vals /= np.sum(pdf_vals)
        weights   = pdf_vals

    else:
        weights = np.ones(H0_likelihood.shape[0])

    # --- Normalize weights and marginalize over galaxies ---
    weights = weights / np.sum(weights)
    print("Shape of final weights:", weights.shape)

    H0_likelihood = np.sum(H0_likelihood.T * weights, axis=1)

    # --- Selection-effect correction and normalization ---
    H0_likelihood /= beta_H0
    H0_likelihood /= np.trapezoid(H0_likelihood, H0_grid)

    return H0_likelihood
