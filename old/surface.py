from __future__ import annotations

import numpy as np
from scipy.io import loadmat
from scipy.linalg import block_diag
import jax
import jax.numpy as jnp

from isofit.core.common import svd_inv
from isofit.surface.surface import Surface


class SurfaceParams:
    def __init__(self, full_config: Config):
        config = full_config.forward_model.surface

        # Models are stored as dictionaries in .mat format
        # TODO: enforce surface_file existence in the case of multicomponent_surface
        model_dict = loadmat(config.surface_file)
        self.prior_means = [mean for mean in model_dict["means"]]
        self.prior_covs = [cov for cov in model_dict["covs"]]
        self.n_comp = len(self.prior_means)
        self.wl = model_dict["wl"][0]
        self.n_wl = len(self.wl)

        # Set up normalization method
        self.norm = lambda r: jax.numpy.linalg.norm(r)

        self.select_on_init = config.select_on_init

        # Reference values are used for normalizing the reflectances.
        # in the VSWIR regime, reflectances are normalized so that the model
        # is agnostic to absolute magnitude.
        refwl = np.squeeze(model_dict["refwl"])
        self.idx_ref = [np.argmin(abs(self.wl - w)) for w in np.squeeze(refwl)]
        self.idx_ref = np.array(self.idx_ref)

        filt = lambda mean: mean[self.idx_ref]
        self.mus = jax.tree.map(filt, self.prior_means)
        # Variables retrieved: each channel maps to a reflectance model parameter
        rmin, rmax = 0, 2.0
        self.statevec_names = ["RFL_%04i" % int(w) for w in self.wl]
        # self.idx_surface = np.arange(len(self.statevec_names))

        self.bounds = [[rmin, rmax] for w in self.wl]
        # self.scale = [1.0 for w in self.wl]
        # self.init = [0.15 * (rmax - rmin) + rmin for v in self.wl]
        self.idx_lamb = np.arange(self.n_wl)
        self.n_state = len(self.statevec_names)

class MultiComponentSurface:
    def __init__(self, idx_lamb, idx_ref, n_state, prior_means, 
                 prior_covs, norm, n_comp, statevec_names, mus):
        self.idx_lamb = idx_lamb
        self.idx_ref = idx_ref
        self.n_state = n_state
        self.prior_means = jnp.array(prior_means)
        self.prior_covs = jnp.array(prior_covs)
        self.norm = norm
        self.n_comp = n_comp
        self.statevec_names = statevec_names
        self.mus = mus

    @jax.jit
    def component(self, x):
        """We pick a surface model component using the Mahalanobis distance.

        This always uses the Lambertian (non-specular) version of the
        surface reflectance. If the forward model initialize via heuristic
        (i.e. algebraic inversion), the component is only calculated once
        based on that first solution. That state is preserved in the
        geometry object.
        """

        if self.n_comp <= 1:
            return 0
        else:
            x_surface = x

        # Get the (possibly normalized) reflectance
        lamb = self.calc_lamb(x_surface)
        lamb_ref = lamb[self.idx_ref]
        lamb_ref = lamb_ref / self.norm(lamb_ref)

        # Euclidean distances
        eq = lambda rmu: sum(pow(lamb_ref - rmu, 2))
        mds = jax.tree.map(eq, self.mus)
        closest = jnp.argmin(jnp.array(mds))

        return (
            self.prior_means.at[closest].get(),
            self.prior_covs.at[closest].get()
        )

    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {
            'idx_lamb': self.idx_lamb,
            'idx_ref': self.idx_ref,
            'n_state': self.n_state,
            'prior_means': self.prior_means,
            'prior_covs': self.prior_covs,
            'norm': self.norm,
            'n_comp': self.n_comp,
            'statevec_names': self.statevec_names,
            'mus': self.mus
        }  # static values
        
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def calc_lamb(self, x_surface):
        """Lambertian reflectance."""

        return x_surface[self.idx_lamb]

    def calc_rfl(self, x_surface):
        """Non-Lambertian reflectance."""
        rfl = self.calc_lamb(x_surface)

        return rfl

    @jax.jit
    def xa(self, x_surface, prior_mean):
        """Mean of prior distribution, calculated at state x. We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function. This always uses the
        Lambertian (non-specular) version of the surface reflectance."""

        lamb = self.calc_lamb(x_surface)
        lamb_ref = lamb[self.idx_ref]

        mu = jnp.zeros(self.n_state)

        lamb_mu = prior_mean * self.norm(lamb_ref)

        mu = mu.at[self.idx_lamb].set(lamb_mu)

        return mu

    @jax.jit
    def Sa(self, x_surface, prior_cov):
        """Covariance of prior distribution, calculated at state x. We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        lamb = self.calc_lamb(x_surface)
        lamb_ref = lamb[self.idx_ref]
        Cov = prior_cov * (self.norm(lamb_ref) ** 2)

        # If there are no other state vector elements, we're done.
        if len(self.statevec_names) == len(self.idx_lamb):
            return Cov

        # Embed into a larger state vector covariance matrix
        nprefix = self.idx_lamb[0]
        nsuffix = len(self.statevec_names) - self.idx_lamb[-1] - 1
        Cov_prefix = jnp.zeros((nprefix, nprefix))
        Cov_suffix = jnp.zeros((nsuffix, nsuffix))

        return jax.scipy.linalg.block_diag(Cov_prefix, Cov, Cov_suffix)


class SurfaceWrapper(MultiComponentSurface):
    def __init__(self, idx_lamb, idx_ref, n_state, prior_means, 
                 prior_covs, norm, n_comp, statevec_names, mus):
        super().__init__(idx_lamb, idx_ref, n_state, prior_means,
                 prior_covs, norm, n_comp, statevec_names, mus)

    def xa(self, x_surface):
        prior_mean, _ = self.component(x_surface)
        return super().xa(x_surface, prior_mean)

    def Sa(self, x_surface):
        _, prior_cov = self.component(x_surface)
        return super().Sa(x_surface, prior_cov)

