from __future__ import annotations

import time
from functools import partial
import os
import itertools

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

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
        self.component_means = jnp.array(self.model_dict["means"])
        self.component_covs = jnp.array(self.model_dict["covs"])

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
        self.idx_lamb = np.arange(self.n_wl)
        self.statevec_names = ["RFL_%04i" % int(w) for w in self.wl]
        self.idx_surface = np.arange(len(self.statevec_names))
        self.n_state = len(self.statevec_names)

        # Cache some important computations
        self.Covs, self.Cinvs = [], []
        self.mus = jnp.array(self.component_means)
        for i in range(self.n_comp):
            Cov = self.component_covs[i]
            self.Covs.append(np.array([Cov[j, self.idx_ref] for j in self.idx_ref]))
            self.Cinvs.append(svd_inv(self.Covs[-1]))

        (
            self.Sa_norm,
            self.Sa_inv_norm,
            self.Sa_inv_sqrt_norm
        ) = self.norm_Sa_cache()

        # Change this if you don't want to analytical solve for all the full statevector elements.
        self.analytical_iv_idx = np.arange(len(self.statevec_names))

        rmin, rmax = 0, 2.0
        self.bounds = [[rmin, rmax] for w in self.wl]
        self.scale = [1.0 for w in self.wl]
        self.init = [0.15 * (rmax - rmin) + rmin for v in self.wl]

        # Surface specific attributes. Can override in inheriting classes
        self.full_glint = False

    def component(self, x0, lamb_norm):
        lamb_ref_norm = jnp.divide(
            x0, 
            lamb_norm
        )
        def distance(lamb_ref, ref_mu):
            return jnp.sum(
                jnp.power(jnp.subtract(lamb_ref, ref_mu), 2),
                axis=1
            )

        distance_vmap = jax.vmap(distance, in_axes=(None, 0))

        return jnp.argmin(
            distance_vmap(lamb_ref_norm, self.mus),
            axis=0
        )

    def xa_cache(self):
        return self.component_means

    def xa(self, x0, lamb_norm):
        """We pick a surface model component using the Mahalanobis distance.

        This always uses the Lambertian (non-specular) version of the
        surface reflectance. If the forward model initialize via heuristic
        (i.e. algebraic inversion), the component is only calculated once
        based on that first solution. That state is preserved in the
        geometry object.
        """
        def distance(lamb_ref, ref_mu):
            return jnp.sum(jnp.power(jnp.subtract(lamb_ref, ref_mu), 2))

        distance_vmap = jax.vmap(distance, in_axes=(None, 0))

        lamb_ref_norm = jnp.divide(
            x0, 
            lamb_norm
        )

        return jnp.multiply(
            self.component_means[jnp.argmin(
                distance_vmap(lamb_ref_norm, self.mus),
                axis=0
            )],
            lamb_norm
        )

    def Sa(self, ci, lamb_norm):

        Sa = jnp.multiply(self.Sa_norm[ci, ...], lamb_norm **2)
        scale = jnp.sqrt(jnp.mean(jnp.diag(Sa)))

        return (
            Sa,
            jnp.divide(self.Sa_inv_norm[ci, ...], scale**2),
            jnp.divide(self.Sa_inv_sqrt_norm[ci, ...] , scale)
        )

    def norm_Sa_cache(self):
        """Covariance of prior distribution, calculated at state x. We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        # hold as arrays of size: (n_comp, n_surface_state, n_surface_state)
        # Normalize the covariances
        C = self.component_covs
        C_norm = jnp.divide(
            C,
            jnp.mean(
                jax.vmap(
                    jnp.diag, 
                    in_axes=0
                )(self.component_covs), 
                axis=1
            )[:, jnp.newaxis, jnp.newaxis]
        )

        # Is there a way to avoid this boolean??
        # This is not JAX friendly
        D, P = jax.vmap(jnp.linalg.eigh)(C_norm)
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

        return C_norm, Cinv, Cinv_sqrt


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

    def calc_rfl(self, x_surface):
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

        rho_dir_dir = rho_dif_dir = self.calc_lamb(x_surface)

        return rho_dir_dir, rho_dif_dir

    def calc_lamb(self, x_surface):
        """Lambertian reflectance."""

        return x_surface[:, self.idx_lamb]

    def analytical_model(
        self,
        background,
        L_tot,
    ):
        """
        Linearization of the surface reflectance terms to use in the
        AOE inner loop (see Susiluoto, 2025). We set the quadratic
        spherical albedo term to a constant background, which
        simplifies the linearization
        background = s * rho_bg
        """
        # If you ignore multi-scattering
        theta = L_tot + (L_tot * background / (1 - background))
        # theta = L_tot

        H = jnp.eye(self.n_wl, self.n_wl)
        H = theta[:, jnp.newaxis] * H

        return H
