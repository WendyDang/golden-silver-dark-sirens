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
| `SDS_bilby_maglim.py` | Magnitude-cut study variant. Runs the first `N_EVENTS` silver injections with three apparent-magnitude thresholds (19, 21, 23 in `AppMagStard_SDSSr`). Loads Uchuu tiles once per event, applies each cut independently, and saves per-cut results under `maglim_study/mlim_{19,21,23}/`. See "Magnitude-cut study" section below. |
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

## External data paths (read-only, outside repo)

### Bilby PE results
```
/hildafs/projects/phy220048p/share/gwbench_dark_siren/bilby_result/
├── Ucchuu_golden_injection_list_HsLsIs.csv   # event table for golden sirens (HsLsIs network)
├── Ucchuu_silver_injection_list_HsLsIs.csv   # event table for silver sirens (HsLsIs network)
├── Ucchuu_1year_injection_list_HpLpVp.csv    # event table for 1-year run (HpLpVp network)
├── Ucchuu_golden_A#/inj_<N>/bilby_inj_<N>_result.json   # golden siren PE results
├── Ucchuu_silver_A#/inj_<N>/bilby_inj_<N>_result.json   # silver siren PE results
└── Ucchuu_1year_A+/inj_<N>/bilby_inj_<N>_result.json    # 1-year PE results
```

Injection list CSV columns: `DL, HostHaloID, Mc, chi1x/y/z, chi2x/y/z, dec, eta, iota, log_DL, log_Mc, phic, psi, ra, tc, z, sky_area_90, snr, cov_dL_ra, cov_dL_dec, cov_ra_dec, err_dL, err_ra, err_dec, m1, m2, network`

Each `inj_<N>/` folder contains: `bilby_inj_<N>_result.json` (main PE output), `.log`, `_corner.png`, `_resume.pickle`, `_dynesty.pickle`, checkpoint PNGs.

### Uchuu galaxy catalog
```
/hildafs/projects/phy220048p/share/Uchuu/z_0.5_healpix_catalog/
└── healpix_000000.parquet … healpix_049151.parquet   (~49,152 files, z < 0.5)
```

Read with `pyarrow.parquet` → pandas. See `/hildafs/home/dangy/Ucchuu_injection/draw_host_galaxies.py` for the reference reading pattern.

#### Full parquet column schema

| Column | Description |
|--------|-------------|
| `ra`, `dec` | Position (degrees) |
| `zcos` | Cosmological redshift |
| `zobs` | Observed redshift (includes peculiar velocity) |
| `vlos` | Line-of-sight velocity |
| `HostHaloID`, `MainHaloID` | Halo identifiers |
| `MstarBulge`, `MstarDisk` | Stellar mass components (M_sun) |
| `HaloMass` | Total halo mass |
| `SFR` | Star formation rate |
| `Mbh` | Black hole mass |
| `Mhot`, `McoldBulge`, `McoldDisk` | Gas mass components |
| `MZgasDisk`, `ZstarBulge`, `ZstarDisk` | Metallicities |
| `MeanAgeStars` | Mean stellar age |
| `Concentration` | Halo concentration |
| `GalaxyType` | Galaxy morphology type |
| `LumAgnBol`, `LumAgnXray`, `MagAgnUV` | AGN luminosities |
| `AppMagStard_SDSSg/r/i/u/z` | **Apparent magnitudes**, SDSS bands (disk component) |
| `MagStar_SDSSg/r/i/u/z` | **Absolute magnitudes**, SDSS bands (total stellar) |
| `MagStard_SDSSg/r/i/u/z` | Absolute magnitudes, SDSS bands (disk component) |

Derived: `stellar_mass = MstarBulge + MstarDisk`. Host selection is weighted by stellar mass, min-max normalised to [1e8, 1e12] M_sun.

**Important:** the Uchuu catalog uses `zcos` for redshift; the current code references `z_hetdex` (from COSMOS). When adapting the pipeline to Uchuu, map `zcos` → wherever `z_hetdex` is read.

## Galaxy catalog (current: COSMOS)

- **Catalog**: COSMOS (FITS format, read via `astropy.io.fits` / `astropy.table.Table`)
- Set `catalog_choice` in `SDS_bilby.py` and fill in `_catalog_paths` dict with the actual file path.
- Required columns: `ra` (deg), `dec` (deg), `z_hetdex`, `gmag`, optionally `mag_abs`.
- The catalog is **spatially shifted** per event so its center aligns with the injection sky position.
- Future plan: replace with Uchuu catalog (see above); redshift column changes from `z_hetdex` → `zcos`.

## Magnitude-cut study (`SDS_bilby_maglim.py`)

Tests how apparent-magnitude-limited galaxy samples affect the H0 posterior. This is an ongoing investigation; only a pilot run (first 10 silver events) has been set up so far.

**Design:** Uchuu tiles are loaded once per event; each of the three magnitude cuts is applied independently to the CI-selected galaxy sample. The absolute magnitude is already available in the catalog (`MagStar_SDSSr`), so no distance-based conversion is needed.

| Parameter | Value |
|-----------|-------|
| Apparent magnitude column | `AppMagStard_SDSSr` |
| Absolute magnitude column (diagnostics) | `MagStar_SDSSr` |
| Magnitude limits tested | 19, 21, 23 |
| Events | First 10 silver injections (`Ucchuu_silver_A#`) |

**Output layout** (root: `maglim_study/`):
```
maglim_study/
├── config_<timestamp>.txt        # run parameters
├── run_<timestamp>.log           # full event-by-event log
├── sky_maps/                     # per-event sky maps (shared; generated once)
├── mlim_19/
│   ├── H0_likelihoods.npz        # H0_grid, joint_H0_posterior, event_<N>, ...
│   ├── H0_posteriors.png
│   └── per_event_stats.csv       # n_in_CI, n_after_cut, frac_kept, med_app_mag,
│                                 # med_abs_mag, host_in_CI, host_survives_cut,
│                                 # host_app_mag, host_abs_mag
├── mlim_21/  (same structure)
├── mlim_23/  (same structure)
└── comparison_H0_posteriors.png  # joint posteriors for all three cuts overlaid
```

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

## Conda environment

Analysis environment: `/hildafs/projects/phy220048p/dangy/gwcosmo`

Activate with:
```
conda activate /hildafs/projects/phy220048p/dangy/gwcosmo
```

Built from scratch (not derived from bilby-env). Package cache stored in
`/hildafs/projects/phy220048p/dangy/conda_pkgs` to avoid home-directory quota.
Key versions: Python 3.11, numpy 2.4, astropy 7.2, bilby 2.8, ligo.skymap 2.5, healpy 1.19.

## Git workflow

- Remote: `https://github.com/WendyDang/golden-silver-dark-sirens`
- Branch: `main`
- All feature additions are committed here so Claude-assisted changes are tracked.
- Commit after each meaningful change; include a short description of what was added or fixed.

## Dependencies

`numpy`, `scipy`, `pandas`, `matplotlib`, `astropy`, `bilby`, `healpy`, `ligo.skymap`, `tqdm`
