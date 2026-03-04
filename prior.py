# prior.py

import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# Setup cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Redshift grid
z = np.linspace(1e-4, 3.0, 1000)

# Luminosity distance in Mpc
dl = cosmo.luminosity_distance(z).to(u.Mpc).value

# Differential comoving volume dVc/dz in Mpc^3/sr
dVc_dz = cosmo.differential_comoving_volume(z).to(u.Mpc**3 / u.sr).value

# Numerical derivative of dL with respect to z: ddL/dz
ddl_dz = np.gradient(dl, z)

# Apply change-of-variable: p(dL) = (dVc/dz / (1+z)) * dz/ddL
pi_dl = dVc_dz / ((1 + z) * ddl_dz)

# Interpolate the prior as a function of dL
pi_dl_interp = interp1d(dl, pi_dl, bounds_error=False, fill_value=0.0)

# Prior function
def prior_dl(dl_val):
    return pi_dl_interp(dl_val) / np.sum(pi_dl_interp(dl_val))
