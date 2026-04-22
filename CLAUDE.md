# CLAUDE.md — golden-silver-dark-sirens

A Python package for computing H0 posteriors from golden and silver gravitational-wave dark sirens, using a shifted COSMOS galaxy catalog as the host galaxy catalog.

## Working directory

```
/hildafs/projects/phy220048p/share/gwbench_dark_siren/golden-silver-dark-sirens/
```

All paths in the code are relative to this directory unless stated otherwise.

## Source files

| File | Purpose |
|------|---------|
| `SDS_bilby.py` | Main entry point. Loops over GW events, calls galaxy selection, likelihood, and posterior. |
| `find_gal_in_CI_varying_H0.py` | Selects galaxies within the sky + distance credible region of a GW event. Three implementations: basic (`find_galaxies_in_sky_and_distance_CI`), grid-KDE fast version (`find_galaxies_in_sky_and_distance_CI_fast`), and HEALPix version (`find_galaxies_in_sky_and_distance_CI_healpix`). The HEALPix version is the one used in production. |
| `H0_likelihood.py` | Computes the GW likelihood per galaxy per H0 using a 3D Gaussian KDE over (RA, Dec, dL) samples. Returns array of shape `(n_galaxies, n_H0)`. |
| `H0_posterior.py` | Marginalizes galaxy likelihoods into an H0 posterior, applies selection-effect correction (`beta_H0`), and optionally applies luminosity weighting. |
| `prior.py` | Defines `prior_dl(dl_val)`: the volumetric luminosity-distance prior `p(dL) ∝ dVc/dz / ((1+z) * ddL/dz)`, interpolated at H0=70. |

## Key input/output paths

| Variable | Default path | Description |
|----------|-------------|-------------|
| `filename` | `./gwbench_GDS_HLI#_HET.txt` | Tab-separated event table (columns: `DL`, `ra`, `dec`, `idx_COSMOS`, `host_found`, etc.) |
| `base_folder` | `./Bilby_automate/HLI#_golden_PE_relative/` | Root of Bilby PE results; expects `inj_<N>/bilby_inj_<N>_result.json` |
| `folder` | `GDS_sharp_HLI/` | Output directory for results |
| `save_dir` | `GDS_sharp_HLI/sky_map_cosmos/` | Per-event sky map PNGs |
| `h0_likelihood_output` | `GDS_sharp_HLI/H0_likelihoods_sf.npz` | Saved H0 likelihoods and joint posterior |
| `config_file` | `GDS_sharp_HLI/config_<timestamp>.txt` | Run configuration log |

## Galaxy catalog

- **Catalog**: COSMOS (FITS format, read via `astropy.io.fits` / `astropy.table.Table`)
- Set `catalog_choice` in `SDS_bilby.py` and fill in `_catalog_paths` dict with the actual file path.
- Required columns: `ra` (deg), `dec` (deg), `z_hetdex`, `gmag`, optionally `mag_abs`.
- The catalog is **spatially shifted** per event so its center aligns with the injection sky position.

## H0 grid

- Default: `np.linspace(60, 80, 40)` in `SDS_bilby.py`
- `H0_likelihood.py` defaults to 20 points; `H0_posterior.py` uses a fixed 10-point beta grid over [60, 80].

## Selection effects (`selection_label`)

Hardcoded in `H0_posterior.py` → `beta_dict`. Available keys:

- `"HLI#G"` — golden siren, HLI# detector network
- `"HLI#S"` — silver siren, HLI# network
- `"HLV+S"` — silver siren, HLV+ network
- `"HLI#S, COSMOS"` — silver siren with COSMOS catalog
- `"HLI#S, SHELA, 0.5"` — silver siren with SHELA catalog, 0.5 threshold
- `"default"` — H0³ prior (no precomputed beta)

Set via `selection_effects` variable in `SDS_bilby.py`.

## Event filtering

Events with `df['DL'][i] >= 980` Mpc are skipped (no host found within catalog).

## Git workflow

- Remote: `https://github.com/WendyDang/golden-silver-dark-sirens`
- Branch: `main`
- All feature additions are committed here so Claude-assisted changes are tracked.
- Commit after each meaningful change; include a short description of what was added or fixed.

## Dependencies

`numpy`, `scipy`, `pandas`, `matplotlib`, `astropy`, `bilby`, `healpy`, `ligo.skymap`, `tqdm`
