import os
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

M_sun_g    = 5.12
_ref_cosmo = FlatLambdaCDM(H0=70, Om0=0.3)   # reference cosmology for lum-weight branch

# ── Precomputed beta(H0) source ──────────────────────────────────────────────────
_BETA_DIR      = ("/hildafs/projects/phy220048p/share/gwbench_dark_siren/"
                  "gwbench/beta_maglim_tests/test1_vs_events")
_BETA_H0_GRID  = np.linspace(60, 80, 20)   # grid used in test_beta_vs_events.py


def _load_beta_from_disk(app_mag_lim, dL_high_90, h0_grid):
    """
    Load precomputed beta(H0) for the dL bin containing dL_high_90 and
    interpolate onto h0_grid.

    Bin edges come from bin_metadata.npz written by test_beta_vs_events.py;
    they are uniform-in-z (Planck15-mapped to dL) by construction.  Each
    event's dL_high_90 (95th-percentile of the GW dL posterior) is digitised
    into one bin.  Files live in _BETA_DIR/mlim_{app_mag_lim}/.
    """
    meta_path = os.path.join(_BETA_DIR, f"mlim_{app_mag_lim}", 'bin_metadata.npz')
    meta      = np.load(meta_path)
    dL_edges  = meta['dL_bin_edges']    # length n_bins+1
    n_bins    = len(dL_edges) - 1

    # np.digitize: returns 1..n_bins for values inside the edges; clip to
    # 0..n_bins-1.  dL_high_90 below the first edge → bin 0; above last → last bin.
    bin_idx = int(np.clip(np.digitize(dL_high_90, dL_edges) - 1, 0, n_bins - 1))

    beta_path = os.path.join(_BETA_DIR, f"mlim_{app_mag_lim}",
                             f"beta_H0_event{bin_idx}.npy")
    beta_vals = np.load(beta_path).astype(float)

    beta_interp = interp1d(
        _BETA_H0_GRID, np.log(beta_vals),
        kind='linear', bounds_error=False, fill_value='extrapolate',
    )
    return np.exp(beta_interp(h0_grid))


def H0_posterior(
    H0_likelihood,
    galaxy_catalog,
    H0_grid,
    df,
    selection_label,
    gmag_key='gmag',
    luminosity_weight=False,
    self_chosen_luminosity_weight=False,
    beta_H0_precomputed=None,
    app_mag_lim=None,
    dL_high_90=None):
    """
    Compute the H0 posterior by marginalizing over host galaxies,
    optionally with luminosity weighting.

    Beta(H0) priority:
      1. app_mag_lim + dL_high_90 → loaded from disk (_BETA_DIR), with the dL
         bin containing dL_high_90 selected via digitisation against
         bin_metadata.npz.  This is the H0-independent apparent-magnitude
         prescription matching test_beta_vs_events.py.
      2. beta_H0_precomputed      → used directly as a precomputed array
      3. selection_label          → looked up in hardcoded beta_dict / H0^3
    """
    H0_beta = np.linspace(60, 80, 10)

    beta_dict = {
        "HLI#G": np.array([
            0.03288448, 0.03618937, 0.03967933, 0.04335953, 0.04723767, 0.05132482,
            0.05563382, 0.06017661, 0.06496137, 0.06999046
        ]),
        "HLI#S": np.array([
            0.03752442, 0.03918533, 0.04179825, 0.04474899, 0.04789355, 0.05117279,
            0.05454209, 0.05797123, 0.06144599, 0.06495914
        ])
        # "HLV+S": np.array([
        #     0.03707336, 0.04041492, 0.04229452, 0.04528477, 0.04872514,
        #     0.05082453, 0.05479356, 0.05700167, 0.05979213, 0.06466416
        # ]),
        # 'HLI#S, COSMOS': np.array([
        #     0.04113569, 0.045531,   0.04665237, 0.04826579, 0.05014428,
        #     0.04926973, 0.0550313,  0.05380664, 0.05257146, 0.05631918
        # ]),
        # 'HLI#S, SHELA, 0.5': np.array([
        #     0.04107071, 0.0430606,  0.04590545, 0.0464678,  0.04920621,
        #     0.051407,   0.05396409, 0.05408589, 0.05603155, 0.05867211
        # ]),
    }

    # --- Selection function ---
    if app_mag_lim is not None:
        beta_H0 = _load_beta_from_disk(app_mag_lim, dL_high_90, H0_grid)
        beta_H0 /= np.trapezoid(beta_H0, H0_grid)
    elif beta_H0_precomputed is not None:
        beta_H0 = np.asarray(beta_H0_precomputed, dtype=float)
        beta_H0 = beta_H0 / np.trapezoid(beta_H0, H0_grid)
    elif selection_label == "default":
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
