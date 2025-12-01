from __future__ import annotations

import time
from functools import partial
import os
import itertools

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=10'
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

from spectral import envi
import optax

from isofit.radiative_transfer.luts import load
from isofit.core.common import envi_header
from isofit.core.geometry import Geometry
from isofit.configs import configs
from isofit.core.fileio import IO
from isofit import ray
from isofit.surface.surface import Surface
from isofit.core.common import svd_inv

from isojax.common import largest_divisible_core
from isojax.lut import lut_grid, check_bounds
from isojax.interpolator import multilinear_interpolator
from isojax.inversions import heuristic_atmosphere, invert_algebraic


class MultiComponentSurface(Surface):
    """A model of the surface based on a collection of multivariate
    Gaussians, with one or more equiprobable components and full
    covariance matrices.

    To evaluate the probability of a new spectrum, we calculate the
    Mahalanobis distance to each component cluster, and use that as our
    Multivariate Gaussian surface model.
    """

    def __init__(self, full_config: Config):
        """."""

        super().__init__(full_config)

        config = full_config.forward_model.surface

        # Models are stored as dictionaries in .mat format
        # TODO: enforce surface_file existence in the case of multicomponent_surface
        self.component_means = self.model_dict["means"]
        self.component_covs = self.model_dict["covs"]

        self.n_comp = len(self.component_means)
        self.wl = self.model_dict["wl"][0]
        self.n_wl = len(self.wl)

        # Set up normalization method
        self.normalize = self.model_dict["normalize"]
        if self.normalize == "Euclidean":
            self.norm = lambda r: norm(r)
        elif self.normalize == "RMS":
            self.norm = lambda r: np.sqrt(np.mean(pow(r, 2)))
        elif self.normalize == "None":
            self.norm = lambda r: 1.0
        else:
            raise ValueError("Unrecognized Normalization: %s\n" % self.normalize)

        self.selection_metric = config.selection_metric
        self.select_on_init = config.select_on_init

        # Reference values are used for normalizing the reflectances.
        # in the VSWIR regime, reflectances are normalized so that the model
        # is agnostic to absolute magnitude.
        self.refwl = np.squeeze(self.model_dict["refwl"])
        self.idx_ref = [np.argmin(abs(self.wl - w)) for w in np.squeeze(self.refwl)]
        self.idx_ref = np.array(self.idx_ref)

        # Cache some important computations
        self.Covs, self.Cinvs = [], []
        self.mus = jnp.array(self.component_means[:, self.idx_ref])
        for i in range(self.n_comp):
            Cov = self.component_covs[i]
            self.Covs.append(np.array([Cov[j, self.idx_ref] for j in self.idx_ref]))
            self.Cinvs.append(svd_inv(self.Covs[-1]))

        # Variables retrieved: each channel maps to a reflectance model parameter
        rmin, rmax = 0, 2.0
        self.statevec_names = ["RFL_%04i" % int(w) for w in self.wl]
        self.idx_surface = np.arange(len(self.statevec_names))

        # Change this if you don't want to analytical solve for all the full statevector elements.
        self.analytical_iv_idx = np.arange(len(self.statevec_names))

        self.bounds = [[rmin, rmax] for w in self.wl]
        self.scale = [1.0 for w in self.wl]
        self.init = [0.15 * (rmax - rmin) + rmin for v in self.wl]
        self.idx_lamb = np.arange(self.n_wl)
        self.n_state = len(self.statevec_names)

        # Surface specific attributes. Can override in inheriting classes
        self.full_glint = False

    def xa(self, x0, lamb_norm):
        """We pick a surface model component using the Mahalanobis distance.

        This always uses the Lambertian (non-specular) version of the
        surface reflectance. If the forward model initialize via heuristic
        (i.e. algebraic inversion), the component is only calculated once
        based on that first solution. That state is preserved in the
        geometry object.
        """
        def distance(lamb_ref, ref_mu):
            return jnp.sum(
                jnp.power(jnp.subtract(lamb_ref, ref_mu), 2),
                axis=1
            )

        distance_vmap = jax.vmap(distance, in_axes=(None, 0))

        lamb_ref_norm = jnp.divide(
            x0[:, self.idx_ref], 
            lamb_norm[:, jnp.newaxis]
        )

        return jnp.multiply(
            self.component_means[jnp.argmin(
                distance_vmap(lamb_ref_norm, self.mus),
                axis=0
            )][:, self.idx_ref],
            lamb_norm[:, jnp.newaxis]
        )

    jax.jit
    def Sa(self):
        """Covariance of prior distribution, calculated at state x. We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        # TODO To handle in memory efficient way 
        # hold as arrays of size: (n_comp, n_surface_state, n_surface_state)

        # Normalize the covariances
        C = jnp.divide(
            self.component_covs,
            jnp.mean(jax.vmap(
                jnp.diag, in_axes=0
                )(self.component_covs), axis=1)[:, jnp.newaxis, jnp.newaxis]
        )

        # Is there a way to avoid this boolean??
        # This is not JAX friendly
        D, P = jax.vmap(jnp.linalg.eigh)(C)
        inv_eps = 1e-6
        if jnp.any(D < 0) or jnp.any(np.isnan(D)):
            D, P = np.linalg.eigh(
                jnp.add(C, jnp.eye(self.n_state) * inv_eps)
            )

        # Do the decomposition
        Ds = jax.vmap(jnp.diag)(1 / jnp.sqrt(D))
        L = jnp.matmul(P, Ds)

        Cinv_sqrt = jnp.matmul(L, jax.vmap(lambda A: A.T)(P))
        Cinv = jnp.matmul(L, jax.vmap(lambda A: A.T)(L))

        return C, Cinv, Cinv_sqrt


    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a state vector."""

        x_surface = np.zeros(len(self.statevec_names))
        if len(rfl_meas) != len(self.idx_lamb):
            raise ValueError("Mismatched reflectances")
        for i, r in zip(self.idx_lamb, rfl_meas):
            x_surface[i] = max(
                self.bounds[i][0] + 0.001, min(self.bounds[i][1] - 0.001, r)
            )

        return x_surface

    def calc_rfl(self, x_surface, geom):
        """Non-Lambertian reflectance.

        Inputs:
        x_surface : np.ndarray
            Surface portion of the statevector element
        geom : Geometry
            Isofit geometry object

        Outputs:
        rho_dir_dir : np.ndarray
            Reflectance quantity for downward direct photon paths
        rho_dif_dir : np.ndarray
            Reflectance quantity for downward diffuse photon paths

        NOTE:
            We do not handle direct and diffuse photon path reflectance
            quantities differently for the multicomponent surface model.
            This is why we return the same quantity for both outputs.
        """

        rho_dir_dir = rho_dif_dir = self.calc_lamb(x_surface, geom)

        return rho_dir_dir, rho_dif_dir

    def calc_lamb(self, x_surface, geom):
        """Lambertian reflectance."""

        return x_surface[self.idx_lamb]

    def calc_Ls(self, x_surface, geom):
        """Emission of surface, as a radiance."""

        return np.zeros(self.n_wl, dtype=float)





config_path = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock_6c/config/emit20220818t205752_isofit.json'
surface = MultiComponentSurface(config)

xa = surface.xa(x0_d, x0_norm_d)
xa_im = np.reshape(xa, (rdn_im.shape[0], rdn_im.shape[1], xa.shape[1]))

Sa, Sa_inv, Sa_inv_Sqrt = surface.Sa()
config = configs.create_new_config(config_path)

lut_path = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock_6c/lut_full/lut.nc'
lut = load(lut_path)
wl = lut.wl.data
points = np.array([point for point in lut.point.data])
lut_names = list(lut.coords)[2:]
jlut = lut_grid(lut, lut_names)
rdn_file = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_rdn_b0106_v01.img'
obs_file = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_obs_b0106_v01.img'
loc_file = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_loc_b0106_v01.img'

rdn = envi.open(envi_header(rdn_file))
rdn_im = rdn.open_memmap(interleave='bip')

wl = np.array(rdn.metadata['wavelength']).astype(float)
bands = [
    np.argmin(np.abs(650 - wl)),
    np.argmin(np.abs(540- wl)),
    np.argmin(np.abs(450- wl)),
]

obs = envi.open(envi_header(obs_file))
obs_im = obs.open_memmap(interleave='bip')

loc = envi.open(envi_header(loc_file))
loc_im = loc.open_memmap(interleave='bip')

# Test batching
meas_list = np.reshape(rdn_im, (rdn_im.shape[0] * rdn_im.shape[1], rdn_im.shape[2]))

# Construct geoms
obs_flat = np.reshape(obs_im, (rdn_im.shape[0] * rdn_im.shape[1], obs_im.shape[2]))
loc_flat = np.reshape(loc_im, (rdn_im.shape[0] * rdn_im.shape[1], loc_im.shape[2]))

test_aod = 0.1
test_h2o = 2.0
point_list = np.zeros((len(obs_flat), 6))
point_list[:, 0] = test_aod
point_list[:, 1] = test_h2o
point_list[:, 2] = obs_flat[:, 2]

delta_phi = np.abs(obs_flat[:, 3] - obs_flat[:, 1])
point_list[:, 3] = np.minimum(delta_phi, 360 - delta_phi)
point_list[:, 4] = obs_flat[:, 4]
point_list[:, 5] = loc_flat[:, 2]

bounds = np.array([
    [np.min(lut.coords['AOT550'].values), np.max(lut.coords['AOT550'].values)],
    [np.min(lut.coords['H2OSTR'].values), np.max(lut.coords['H2OSTR'].values)],
    [np.min(lut.coords['observer_zenith'].values), np.max(lut.coords['observer_zenith'].values)],
    [np.min(lut.coords['relative_azimuth'].values), np.max(lut.coords['relative_azimuth'].values)],
    [np.min(lut.coords['solar_zenith'].values), np.max(lut.coords['solar_zenith'].values)],
    [np.min(lut.coords['surface_elevation_km'].values), np.max(lut.coords['surface_elevation_km'].values)],
])

point_list = check_bounds(
    point_list,
    bounds[:, 0],
    bounds[:, 1]
)

b865 = np.argmin(abs(wl - 865))
b945 = np.argmin(abs(wl - 945))
b1040 = np.argmin(abs(wl - 1040))

fix_aod = 0.1
h2os = heuristic_atmosphere(
    np.unique(lut.coords['H2OSTR'].data), 
    meas_list, 
    point_list, 
    jlut, 
    fix_aod=fix_aod, 
    batchsize=100000, 
    b865=b865,
    b945=945,
    b1040=1040
)

point_list = point_list.at[:, 0].set(fix_aod)
point_list = point_list.at[:, 1].set(h2os)
# invert_algebraic = jax.vmap(invert_algebraic)
x0 = invert_algebraic(
    meas_list,
    jlut['rhoatm'](point_list),
    jlut['dir_dir'](point_list),
    jlut['dir_dif'](point_list),
    jlut['dif_dir'](point_list),
    jlut['dif_dif'](point_list),
    jlut['sphalb'](point_list),
)

imnorm = lambda x, vmin, vmax: (x - vmin) / (vmax - vmin)
x0_im = np.reshape(x0, rdn_im.shape)

plt.imshow(imnorm(x0_im[..., bands], 0, .3))
plt.show()

surface = MultiComponentSurface(config)

lamb_ref = jnp.divide(x0, jnp.linalg.norm(x0, axis=1)[:, jnp.newaxis])

def distance(lamb_ref, ref_mu):
    return jnp.sum(jnp.power(jnp.subtract(lamb_ref, ref_mu), 2), axis=1)

mus = jnp.array(surface.mus)
distance_vmap = jax.vmap(distance, in_axes=(None, 0))
test = jnp.min(distance_vmap(lamb_ref[:, surface.idx_ref], mus), axis=0)

surface = MultiComponentSurface(config)
x0_norm = jnp.linalg.norm(x0[:, surface.idx_ref], axis=1)

mesh_size = largest_divisible_core(len(x0))
mesh = jax.make_mesh((mesh_size,), ('x'))
x0_d = jax.device_put(x0, NamedSharding(mesh, P('x')))
x0_norm_d = jax.device_put(x0_norm, NamedSharding(mesh, P('x')))

xa = surface.xa(x0_d, x0_norm_d)
xa_im = np.reshape(xa, (rdn_im.shape[0], rdn_im.shape[1], xa.shape[1]))

plt.imshow(imnorm(xa_im[..., bands], 0, .2))
plt.show()



Sa, Sa_inv, Sa_inv_Sqrt = surface.Sa()
