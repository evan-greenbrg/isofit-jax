import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import tree_util
import numpy as np

from isofit.core.common import eps
from isofit.radiative_transfer.radiative_transfer import RadiativeTransfer
from isofit.radiative_transfer import luts
from isofit.radiative_transfer.engines import Engines


class VectorInterpolator:
    def __init__(
        self,
        grid_input,
        data_input,
    ):
        # Determine if this a singular unique value, if so just return that directly
        val = data_input[(0,) * data_input.ndim]
        # if np.isnan(val) and np.isnan(data_input).all() or np.all(data_input == val):
        #     self.method = -1
        #     self.value = val
        #     return

        self.single_point_data = None

        # Lists and arrays are mutable, so copy first
        grid = grid_input.copy()
        data = data_input.copy()

        # Check if we are using a single grid point. If so, store the grid input.
        if np.prod(list(map(len, grid))) == 1:
            self.single_point_data = data
        self.n = data.shape[-1]

        # None to disable, 0 for unlimited, negatives == 1
        self.cache_size = 0

        self.gridtuples = [np.array(t) for t in grid]
        self.gridarrays = data
        self.binwidth = [
            t[1:] - t[:-1] for t in self.gridtuples
        ]  # binwidth arrays for each dimension
        self.maxbaseinds = np.array([len(t) - 1 for t in self.gridtuples])

    def __call__(self, *args):
        return self._multilinear_grid(
            *args,
            gridtuples=self.gridtuples,
            maxbaseinds=self.maxbaseinds,
            binwidth=self.binwidth,
            gridarrays=jnp.array(self.gridarrays),
            nwl=self.gridarrays.shape[-1]
        )

    def _multilinear_grid(self, points, gridtuples, maxbaseinds, binwidth, gridarrays, nwl=285):
        deltas = jnp.zeros(points.shape, dtype=jnp.int32)
        idxs = jnp.zeros((points.size, 2), dtype=jnp.int32)

        for i, point in enumerate(points):
            delts, slices = self._lookup(
                i, point, 
                jnp.array(gridtuples[i]), 
                maxbaseinds[i], 
                jnp.array(binwidth[i])
            )

            idxs = idxs.at[i].set(slices)
            deltas = deltas.at[i].set(delts)


        dynamic_slice = jax.jit(
            self._dynamic_slice,
            static_argnames=('nwl',)
        )
        cube = dynamic_slice(gridarrays, idxs, nwl)

        # Only linear interpolate sliced dimensions
        for i, idx in enumerate(idxs):
            cube = cube.at[0].multiply(1 - deltas.at[i].get())
            cube = cube.at[1].multiply(deltas.at[i].get())
            cube = cube.at[0].add(cube.at[1].get())
            cube = cube.at[0].get()

        return cube

    def _lookup(self, i, point, gridtuples, maxbaseinds, binwidth):
        j = jnp.searchsorted(gridtuples[:-1], point) - 1
        delta = jnp.divide(jnp.subtract(point, gridtuples[j]), binwidth[j])

        lower = jnp.max(
            jnp.array([jnp.min(jnp.array([maxbaseinds, j])), 0])
        )
        upper = jnp.max(
            jnp.array([jnp.min(jnp.array([maxbaseinds + 2, j + 2])), 2])
        )

        slices = jnp.array([(lower, upper) for i in gridtuples])
        slices = slices.at[0].set([lower, lower])
        slices = slices.at[-1].set([upper - 1, upper - 1])

        delts = jnp.array([delta for i in gridtuples])
        delts = delts.at[0].set(np.nan)
        delts = delts.at[-1].set(np.nan)

        k = jnp.argmin(jnp.abs(jnp.subtract(point, gridtuples)))

        return delts.at[k].get(), slices.at[k].get()

    @staticmethod
    def _dynamic_slice(gridarrays, idxs, nwl):
        size1 = jnp.subtract(
            idxs.at[(0, 1)].get(),
            idxs.at[(0, 0)].get()
        )

        cube = jax.lax.dynamic_slice(
            gridarrays,
            [idxs.at[(0, 0)].get(), idxs.at[(1, 0)].get(), 0],
            [2, 2, nwl],
        )

        return cube


def confPriority(key, configs):
    """
    Selects a key from a config if the value for that key is not None
    Prioritizes returning the first value found in the configs list

    TODO: ISOFIT configs are annoying and will create keys to NoneTypes
    Should use mlky to handle key discovery at runtime instead of like this
    """
    value = None
    for config in configs:
        if hasattr(config, key):
            value = getattr(config, key)
            if value is not None:
                break
    return value


class RTEparams:
    _keys = [
        "interpolator_style",
        "overwrite_interpolator",
        "lut_grid",
        "lut_path",
        "wavelength_file",
    ]
    def __init__(self, full_config):
        config = full_config.forward_model.radiative_transfer
        confIT = full_config.forward_model.instrument

        self.lut_grid = config.lut_grid
        self.statevec_names = config.statevector.get_element_names()

        self.rt_engines = []
        for idx in range(len(config.radiative_transfer_engines)):
            confRT = config.radiative_transfer_engines[idx]

            # Generate the params for this RTE
            params = {
                key: confPriority(key, [confRT, confIT, config]) for key in self._keys
            }
            params["engine_config"] = confRT

            # Select the right RTE and initialize it
            rte = Engines[confRT.engine_name](**params)
            self.rt_engines.append(rte)

            # Make sure the length of the config statevectores match the engine's assumed statevectors
            if (expected := len(config.statevector.get_element_names())) != (
                got := len(rte.indices.x_RT)
            ):
                error = f"Mismatch between the number of elements for the config statevector and LUT.indices.x_RT: {expected=}, {got=}"
                Logger.error(error)
                raise AttributeError(error)

        # The rest of the code relies on sorted order of the individual RT engines which cannot
        # be guaranteed by the dict JSON or YAML input
        self.rt_engines.sort(key=lambda x: x.wl[0])

        # Retrieved variables.  We establish scaling, bounds, and
        # initial guesses for each state vector element.  The state
        # vector elements are all free parameters in the RT lookup table,
        # and they all have associated dimensions in the LUT grid.
        self.bounds, self.scale, self.init = [], [], []
        self.prior_mean, self.prior_sigma = [], []

        for sv, sv_name in zip(*config.statevector.get_elements()):
            self.bounds.append(sv.bounds)
            self.scale.append(sv.scale)
            self.init.append(sv.init)
            self.prior_sigma.append(sv.prior_sigma)
            self.prior_mean.append(sv.prior_mean)

        self.bounds = np.array(self.bounds)
        self.scale = np.array(self.scale)
        self.init = np.array(self.init)
        self.prior_mean = np.array(self.prior_mean)
        self.prior_sigma = np.array(self.prior_sigma)

        self.wl = np.concatenate([RT.wl for RT in self.rt_engines])

        self.bvec = config.unknowns.get_element_names()
        self.bval = np.array([x for x in config.unknowns.get_elements()[0]])

        self.solar_irr = np.concatenate([RT.solar_irr for RT in self.rt_engines])

class RTEcalc:
    def __init__(self, 
        RT_indices,
        lut, lut_names,
        geom_indices,
        solar_irr,
        rt_mode,
        prior_mean,
        prior_sigma):
        
        self.RT_indices = RT_indices
        self.lut = lut
        self.lut_names = lut_names
        self.geom_indices = geom_indices
        self.coupling_terms = ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]
        self.solar_irr = solar_irr
        self.rt_mode = rt_mode
        self.prior_mean = jnp.array(prior_mean)
        self.prior_sigma = jnp.array(prior_sigma)

    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {
            'RT_indices': self.RT_indices,
            'lut': self.lut,
            'lut_names': self.lut_names,
            'geom_indices': self.geom_indices,
            'solar_irr': self.solar_irr,
            'rt_mode': self.rt_mode,
            'prior_mean': self.prior_mean,
            'prior_sigma': self.prior_sigma,
        }  # static values
        
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
    @jax.jit
    def get_rtm_quantities(
        self,
        x_RT, 
        geom_vals,
    ):

        point = jnp.zeros(len(self.lut_names))
        point = point.at[jnp.array(self.RT_indices)].set(x_RT)

        for i, val in zip(self.geom_indices, geom_vals):
            point = point.at[i].set(val)

        return {key: lut(point) for key, lut in self.lut.items()}


    @jax.jit
    def get_L(self, x_RT, geom_vals, coszen, cos_i):
        r = self.get_rtm_quantities(x_RT, geom_vals)

        # Coupled radiance
        L_coupled = jnp.zeros((len(self.coupling_terms), len(self.solar_irr)))
        for i, key in enumerate(self.coupling_terms):
            L_coupled = L_coupled.at[i].set(
                self.solar_irr * coszen / np.pi * r[key]
                if self.rt_mode == "transm"
                else r[key]
            )

        topography_shift = jnp.array([
            1 / (coszen * cos_i),
            1, 
            1 / (coszen * cos_i),
            1, 
        ])
        L_coupled = jnp.multiply(L_coupled.T, topography_shift)

        L_dir_dir = L_coupled.at[:, 0].get()
        L_dif_dir = L_coupled.at[:, 1].get()
        L_dir_dif = L_coupled.at[:, 2].get()
        L_dif_dif = L_coupled.at[:, 3].get()

        # Path Radiance
        if self.rt_mode == "rdn":
            L_atm = r["rhoatm"]
        else:
            L_atm = (self.solar_irr * coszen) / jnp.pi * r["rhoatm"]

        sphalb = r["sphalb"]

        return L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif, L_atm, sphalb 

    @jax.jit
    def xa(self, prior_mean):
        """Pull the priors from each of the individual RTs."""
        return prior_mean

    @jax.jit

    def Sa(self, prior_sigma):
        """Pull the priors from each of the individual RTs."""
        return jnp.diagflat(jnp.power(prior_sigma, 2))


@jax.jit
def calc_rdn(
    rho,
    s_alb,
    L_atm,
    L_dir_dir,
    L_dif_dir,
    L_dir_dif,
    L_dif_dif,
):
    # Atmospheric spherical albedo
    atm_surface_scattering = s_alb * rho

    L_tot = L_dir_dir + L_dif_dir + L_dir_dif + L_dif_dif
    # TOA radiance model
    ret = (
        L_atm
        + L_dir_dir * rho
        + L_dif_dir * rho
        + L_dir_dif * rho
        + L_dif_dif * rho
        + (L_tot * atm_surface_scattering * rho) / (1 - s_alb * rho)
    )

    return ret


def initialize_rt(config):
    """
    Wrapper to initialize the Isofit RT.
    This will kick off the RTE
    """
    return RadiativeTransfer(config)


def build_interpolators(lut, lut_names):
    """
    Builds the interpolators using the LUT store

    TODO: optional load from/write to disk
    """
    luts_interp = {}

    ds = lut.unstack("point")

    # Make sure its in expected order, wl at the end
    ds = ds.transpose(*lut_names, "wl")

    grid = [ds[key].data for key in lut_names]

    # Create the unique
    for key in luts.Keys.alldim:
        luts_interp[key] = VectorInterpolator(
            grid_input=grid,
            data_input=ds[key].load().data,
        )

    return luts_interp

