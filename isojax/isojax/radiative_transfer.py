from __future__ import annotations

import logging
from functools import partial

import numpy as np
import jax.numpy as jnp
import jax

from isofit.core import units
from isofit.core.common import eps
from isofit.radiative_transfer.engines import Engines


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


class RadiativeTransfer:
    """This class controls the radiative transfer component of the forward
    model. An ordered dictionary is maintained of individual RTMs (MODTRAN,
    for example). We loop over the dictionary concatenating the radiation
    and derivatives from each RTM and interval to form the complete result.

    In general, some of the state vector components will be shared between
    RTMs and bands. For example, H20STR is shared between both VISNIR and
    TIR. This class maintains the master list of statevectors.
    """

    # Keys to retrieve from 3 sections to use the preferred
    # Prioritizes retrieving from radiative_transfer_engines first, then instrument, then radiative_transfer
    _keys = [
        "interpolator_style",
        "overwrite_interpolator",
        "lut_grid",
        "lut_path",
        "wavelength_file",
    ]

    def __init__(self, full_config: Config, jlut):
        config = full_config.forward_model.radiative_transfer
        confIT = full_config.forward_model.instrument

        self.jlut = jlut
        self.lut_grid = config.lut_grid
        self.statevec_names = config.statevector.get_element_names()

        # Temporary to fit with the existing isofit config
        confRT = config.radiative_transfer_engines[0]
        params = {
            key: confPriority(key, [confRT, confIT, config]) for key in self._keys
        }
        params["engine_config"] = confRT
        self.rt_engine = Engines[confRT.engine_name](**params)

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

        self.bounds = jnp.array(self.bounds)
        self.scale = jnp.array(self.scale)
        self.init = jnp.array(self.init)
        self.xa = jnp.array(self.prior_mean)
        self.prior_sigma = jnp.array(self.prior_sigma)

        (
            self._Sa,
            self.Sa_inv,
            self.Sa_inv_sqrt
        ) = self.norm_Sa_cache()

        # Temporary I don't feel like debugging config rn
        self.wl = self.rt_engine.wl

        self.bvec = config.unknowns.get_element_names()
        self.bval = np.array([x for x in config.unknowns.get_elements()[0]])

        self.solar_irr = self.rt_engine.solar_irr

    def norm_Sa_cache(self):
        C = jnp.diagflat(jnp.power(self.prior_sigma, 2))
        D, P = jnp.linalg.eigh(C)
        Ds = jnp.diag(1 / jnp.sqrt(D))
        L = jnp.matmul(P, Ds)

        Cinv_sqrt = jnp.matmul(L, P.T)
        Cinv = jnp.matmul(L, L.T)

        return C, Cinv, Cinv_sqrt

    def Sa(self):
        return self._Sa, self.Sa_inv, self.Sa_inv_sqrt

    @staticmethod
    def calc_rdn(
        rho_dir_dir,
        rho_dif_dir,
        L_atm,
        s_alb,
        L_dir_dir,
        L_dif_dir,
        L_dir_dif,
        L_dif_dif,
    ):
        """
        Physics-based forward model to calculate at-sensor radiance.
        Includes topography, background reflectance, and glint.
        """

        # TOA radiance model
        return (
            L_atm
            + L_dir_dir * rho_dir_dir
            + L_dif_dir * rho_dif_dir
            + L_dir_dif * rho_dir_dir
            + L_dif_dif * rho_dif_dir
            + (
                (
                    L_dir_dir
                    + L_dir_dir
                    + L_dif_dir
                    + L_dif_dir
                ) * s_alb * rho_dif_dir**2
            ) / (1 - s_alb * rho_dif_dir)
        )

    def drdn_dRTb(self, point, rho_dir, rho_dif):
        """Derivative of estimated rdn w.r.t. H2O_ABSCO
        H2OSTR is implemented by default and explicitely here
        """
        i = self.statevec_names.index("H2OSTR")
        perturb = 1.0 + eps
        point = jnp.vstack([
            point,
            point
        ])
        point = point.at[1, i].set(point[1, i] * perturb)

        L_atm = self.jlut['rhoatm'](point)
        L_dir_dir = self.jlut['dir_dir'](point)
        L_dir_dif = self.jlut['dir_dif'](point)
        L_dif_dir = self.jlut['dif_dir'](point)
        L_dif_dif = self.jlut['dif_dif'](point)
        s_alb = self.jlut['sphalb'](point)

        rdn = self.calc_rdn(
            rho_dir,
            rho_dif,
            L_atm,
            s_alb,
            L_dir_dir,
            L_dif_dir,
            L_dir_dif,
            L_dif_dif,
        )

        return jnp.squeeze(jnp.diff(rdn, axis=0) / eps)
