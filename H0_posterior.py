import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from prior import prior_dl

# Constants
c_km_s = 299792.458  # Speed of light in km/s

# Assumed absolute magnitude of the Sun in g band
M_g_sun = 5.12

import numpy as np
from scipy.interpolate import interp1d

import numpy as np
from scipy.interpolate import interp1d

# Constants
c_km_s = 299792.458  # km/s
M_g_sun = 5.12



def H0_posterior(
    H0_likelihood,
    galaxy_catalog,
    H0_grid,
    df,
    selection_label, #"HLI#S"
    gmag_key='gmag',
    luminosity_weight=False,
    self_chosen_luminosity_weight=False):
    """
    Compute the H0 likelihood by marginalizing over host galaxies,
    optionally with luminosity weighting or empirical weighting from injected galaxies.
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
        'HLI#S, COSMOS': np.array([0.04113569, 0.045531, 0.04665237, 0.04826579, 0.05014428, 0.04926973,
                                   0.0550313, 0.05380664, 0.05257146, 0.05631918]),

        'HLI#S, SHELA, 0.5': np.array([0.04107071, 0.0430606,  0.04590545, 0.0464678,  0.04920621, 0.051407,
 0.05396409, 0.05408589, 0.05603155, 0.05867211])
        
    }

    # --- Selection function ---
    if selection_label == "default":
        beta_H0 = H0_grid**3
        beta_H0 = beta_H0 / np.trapz(beta_H0, H0_grid)

    else:
        if selection_label not in beta_dict:
            raise ValueError(f"Unknown selection_label: {selection_label}")

        beta_vals = beta_dict[selection_label]
        beta_vals = beta_vals / np.trapz(beta_vals, H0_beta)

        beta_interp = interp1d(
            H0_beta,
            np.log(beta_vals),
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        beta_H0 = np.exp(beta_interp(H0_grid))


    # Constants
    M_g_sun = 5.12      # Sun’s absolute magnitude in g-band (approx)
    M_sun_g = 5.12

    # --- CASE 1: standard luminosity weighting ---
    if luminosity_weight:
        weights_list = []

        for j in range(len(gw_likelihood)):
            # Get single galaxy data
            gmag = galaxy_catalog[gmag_key][j]
            z_gal = galaxy_catalog['z_hetdex'][j]

            # Compute luminosity distance (Mpc → pc)
            dL_gal = cosmo.luminosity_distance(z_gal).to('Mpc').value
            dL_gal_pc = dL_gal * 1e6

            # Compute absolute magnitude and luminosity
            M_g = gmag - 5 * (np.log10(dL_gal_pc) - 1)
            Lg = 10 ** (-0.4 * (M_g - M_sun_g))

            # Store in list
            weights_list.append(Lg)

        # Convert to numpy array
        weights = np.array(weights_list)
        


    # --- CASE 2: empirical luminosity weighting from injected population ---
    # elif self_chosen_luminosity_weight:
    #     # Build empirical luminosity PDF from injected population
    #     mask = df["dL_nearest"] < 980
    #     gmag_inj = df["gmag_nearest"][mask].values
    #     dL_inj = df["dL_nearest"][mask].values  # Mpc
    #     M_abs_inj = gmag_inj - 5 * np.log10(dL_inj) - 25
    #     L_inj = 10 ** (-0.4 * (M_abs_inj - M_sun_g))
    #     logL_inj = np.log10(L_inj)

    #     # KDE for empirical luminosity distribution
    #     kde = gaussian_kde(logL_inj, bw_method='scott')

    #     weights_list = []
    #     logL_gal_list = []

    #     for j in range(len(gw_likelihood)):
    #         gmag = galaxy_catalog[gmag_key][j]
    #         z_gal = galaxy_catalog['z_hetdex'][j]
    #         dL_gal = cosmo.luminosity_distance(z_gal).to('Mpc').value
    #         dL_gal_pc = dL_gal * 1e6
    #         M_g = gmag - 5 * (np.log10(dL_gal_pc) - 1)
    #         Lg = 10 ** (-0.4 * (M_g - M_sun_g))
    #         logL_gal_list.append(np.log10(Lg))

    #     # --- Convert to array and evaluate KDE for all galaxies at once ---
    #     logL_gal = np.array(logL_gal_list)
    #     pdf_vals = kde(logL_gal).ravel()
    #     print("Shape of PDF values:", pdf_vals.shape)

    #     # --- Clean up and normalize ---
    #     pdf_vals = np.maximum(pdf_vals, 1e-12)
    #     pdf_vals /= np.sum(pdf_vals)
    #     weights = pdf_vals
    #     print("Shape of weights:", weights.shape)
    elif self_chosen_luminosity_weight:
        

        # Build empirical luminosity PDF from injected population
        mask = df["host_found"] ==1
        M_abs_inj = df["abs_mag_nearest"][mask].values

        # Convert to luminosity
        L_inj = 10 ** (-0.4 * (M_abs_inj - M_sun_g))
        logL_inj = np.log10(L_inj)

        # KDE for empirical luminosity distribution
        kde = gaussian_kde(logL_inj, bw_method='scott')

        # --- Use precalculated absolute magnitudes ---
        M_abs_gal = galaxy_catalog['mag_abs']

        # Convert to luminosity
        L_gal = 10 ** (-0.4 * (M_abs_gal - M_sun_g))
        logL_gal = np.log10(L_gal)

        # Evaluate KDE
        pdf_vals = kde(logL_gal).ravel()

        # Clean up and normalize
        pdf_vals = np.maximum(pdf_vals, 1e-12)
        pdf_vals /= np.sum(pdf_vals)

        weights = pdf_vals



    # --- CASE 3: no luminosity weighting ---
    else:
        weights = np.ones(gw_likelihood.shape[0])



    # --- Normalize and compute final H0 likelihood ---
    weights = weights / np.sum(weights)
    print("Shape of final weights:", weights.shape)


    H0_likelihood = np.sum(gw_likelihood.T * weights, axis=1)
   
    # Selection effect correction and normalization
    
    H0_likelihood /= beta_H0
    H0_likelihood /= np.trapz(H0_likelihood, H0_grid)

    return H0_likelihood


