#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: Evan Greenberg, evan.greenberg@jpl.nasa.gov
#
from __future__ import annotations

import logging
from copy import deepcopy
from functools import partial

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.linalg import block_diag
import jax
import jax.numpy as jnp

from isofit.core.common import eps

from isojax.surface import MultiComponentSurface
from isojax.radiative_transfer import RadiativeTransfer 
from isojax.instrument import Instrument

# Logger = logging.getLogger(__file__)


class ForwardModel:
    """ForwardModel contains all the information about how to calculate
    radiance measurements at a specific spectral calibration, given a
    state vector. It also manages the distributions of unretrieved,
    unknown parameters of the state vector (i.e. the S_b and K_b
    matrices of Rodgers et al.

    State vector elements always go in the following order:
      (1) Surface parameters
      (2) Radiative Transfer (RT) parameters
      (3) Instrument parameters

    The parameter bounds, scales, initial values, and names are all
    ordered in this way.  The variable self.statevec contains the name
    of each state vector element, in the proper ordering.

    The "b" vector corresponds to the K_b calculations in Rogers (2000);
    the variables bvec and bval represent the model unknowns' names and
    their  magnitudes, respectively.  Larger magnitudes correspond to
    a larger variance in the unknown values.  This acts as additional
    noise for the purpose of weighting the measurement information
    against the prior."""

    def __init__(self, full_config: Config, cache_RT: RadiativeTransfer = None):
        # load in the full config (in case of inter-module dependencies) and
        # then designate the current config
        self.full_config = full_config

        # Build the instrument model
        self.instrument = Instrument(self.full_config)
        self.n_meas = self.instrument.n_chan

        # Build the radiative transfer model
        if cache_RT:
            self.RT = cache_RT
        else:
            self.RT = RadiativeTransfer(self.full_config)

        # Build the surface model
        self.surface = MultiComponentSurface(full_config)

        # Check to see if using supported calibration surface model
        if self.surface.n_wl != len(self.RT.wl) or not np.all(
            np.isclose(self.surface.wl, self.RT.wl, atol=0.01)
        ):
            Logger.warning(
                "Surface and RTM wavelengths differ - if running at higher RTM"
                " spectral resolution or with variable wavelength position, this"
                " is expected.  Otherwise, consider checking the surface model."
            )

        # Build combined vectors from surface, RT, and instrument
        bounds, scale, init, statevec, bvec, bval = ([] for i in range(6))
        for obj_with_statevec in [self.surface, self.RT, self.instrument]:
            bounds.extend([deepcopy(x) for x in obj_with_statevec.bounds])
            scale.extend([deepcopy(x) for x in obj_with_statevec.scale])
            init.extend([deepcopy(x) for x in obj_with_statevec.init])
            statevec.extend([deepcopy(x) for x in obj_with_statevec.statevec_names])

            bvec.extend([deepcopy(x) for x in obj_with_statevec.bvec])
            bval.extend([deepcopy(x) for x in obj_with_statevec.bval])

        self.bounds = tuple(np.array(bounds).T)
        self.scale = np.array(scale)
        self.init = np.array(init)
        self.statevec = statevec
        self.nstate = len(self.statevec)

        self.bvec = np.array(bvec)
        self.nbvec = len(self.bvec)
        self.bval = np.array(bval)
        self.Sb = np.diagflat(np.power(self.bval, 2))

        """Set up state vector indices - 
        MUST MATCH ORDER FROM ABOVE ASSIGNMENT

        Sometimes, it's convenient to have the index of the entire surface
        as one variable, and sometimes you want the sub-components
        Split surface state vector indices to cover cases where we retrieve
        additional non-reflectance surface parameters
        """
        # entire surface portion
        self.idx_surface = np.arange(len(self.surface.statevec_names), dtype=int)

        # surface reflectance portion
        self.idx_surf_rfl = self.idx_surface[: len(self.surface.idx_lamb)]

        # non-reflectance surface parameters
        self.idx_surf_nonrfl = self.idx_surface[len(self.surface.idx_lamb) :]

        # radiative transfer portion
        self.idx_RT = np.arange(len(self.RT.statevec_names), dtype=int) + len(
            self.idx_surface
        )

        # instrument portion
        self.idx_instrument = (
            np.arange(len(self.instrument.statevec_names), dtype=int)
            + len(self.idx_surface)
            + len(self.idx_RT)
        )

        self.surface_b_inds = np.arange(len(self.surface.bvec), dtype=int)
        self.RT_b_inds = np.arange(len(self.RT.bvec), dtype=int) + len(
            self.surface_b_inds
        )
        self.instrument_b_inds = (
            np.arange(len(self.instrument.bvec), dtype=int)
            + len(self.surface_b_inds)
            + len(self.RT_b_inds)
        )

        # Load model discrepancy correction
        if full_config.forward_model.model_discrepancy_file is not None:
            D = loadmat(full_config.forward_model.model_discrepancy_file)
            self.model_discrepancy = D["cov"]
        else:
            self.model_discrepancy = None

    def calc_meas(self, x, L_atm, s_alb, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif):
        rho_dir_dir, rho_dif_dir = self.surface.calc_rfl(
            x[:, self.idx_surface]
        )

        return self.RT.calc_rdn(
            rho_dir_dir,
            rho_dif_dir,
            L_atm,
            s_alb,
            L_dir_dir,
            L_dif_dir,
            L_dir_dif,
            L_dif_dif,
        )

    def Seps(self, x, meas, points, jlut):
        if self.model_discrepancy is not None:
            Gamma = self.model_discrepancy
        else:
            Gamma = 0

        @partial(
            jax.vmap,
            in_axes=(0, 0, None)
        )
        def calc(Sy, Kb, Gamma): 
            return Sy + Kb.dot(self.Sb).dot(Kb.T) + Gamma
        
        return calc(
            self.instrument.Sy(meas),
            self.Kb(meas, x, points, jlut),
            Gamma
        )

    def Kb(self, meas, x, points, jlut):
        """Derivative of measurement with respect to unmodeled & unretrieved
        unknown variables, e.g. S_b. This is  the concatenation of Jacobians
        with respect to parameters of the surface, radiative transfer model,
        and instrument.  Currently we only treat uncertainties in the
        instrument and RT model.

        Has to be part of vmap??
        """
        # Call surface reflectance w.r.t. surface, upsample
        rho_dir_dir, rho_dif_dir = self.surface.calc_rfl(
            x[:, self.idx_surface]
        )

        dRTb = self.RT.drdn_dRTb(
            points,
            rho_dir_dir,
            rho_dif_dir,
            jlut,
        )

        # To get derivatives w.r.t. instrument, downsample to instrument wavelengths
        dinstrumentb = self.instrument.dmeas_dinstrumentb(meas)

        @jax.vmap
        def combine(dRTb, dinstrumentb):
            # Put it together
            Kb = jnp.zeros((self.n_meas, self.nbvec), dtype=float)
            Kb = Kb.at[:, self.RT_b_inds].set(dRTb[:, jnp.newaxis])
            Kb = Kb.at[:, self.instrument_b_inds].set(dinstrumentb)
            return Kb

        return combine(dRTb, dinstrumentb)

    def xa(self, x, lamb_norm):
        xa_surface = self.surface.xa(x, lamb_norm)
        xa_RT = self.RT.xa

        @partial(
            jax.vmap,
            in_axes=(0, None)
        )
        def concat(xa_surface, xa_RT):
            return jnp.concatenate((xa_surface, xa_RT), axis=0)
        
        return concat(xa_surface, xa_RT)

    def Sa(self, ci, lamb_norm):
        (
            Sa_surface,
            Sa_inv_norm,
            Sa_inv_sqrt_norm
        ) = self.surface.Sa(ci, lamb_norm)

        Sa = jax.vmap(
            jax.scipy.linalg.block_diag,
            in_axes=(0, None)
        )(Sa_surface, self.RT.Sa)

        @jax.vmap
        def calc_scale(Sa):
            return jnp.sqrt(jnp.mean(jnp.diag(Sa)))

        scale = calc_scale(Sa)[:, jnp.newaxis]

        @partial(
            jax.vmap,
            in_axes=(0, None, 0)
        )
        def block_and_scale(inv_surface, inv_RT, scale):
            return (
                jax.scipy.linalg.block_diag(
                    inv_surface,
                   inv_RT 
                )
                / jnp.pow(scale, 2)
            )

        Sa_inv = block_and_scale(
            Sa_inv_norm,
            self.RT.Sa_inv_norm,
            scale
        )

        Sa_inv_sqrt = block_and_scale(
            Sa_inv_sqrt_norm,
            self.RT.Sa_inv_sqrt_norm,
            scale
        )

        return Sa, Sa_inv, Sa_inv_sqrt


