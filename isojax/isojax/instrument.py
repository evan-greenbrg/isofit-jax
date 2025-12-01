from __future__ import annotations

import logging
from functools import partial

import numpy as np
from scipy.interpolate import interp1d, splev, splrep
from scipy.io import loadmat
from scipy.signal import convolve
import jax.numpy as jnp
import jax

from isofit.core import units
from isofit.core.common import (
    emissive_radiance,
    eps,
    load_wavelen,
    resample_spectrum,
    spectral_response_function,
)

### Variables ###

# Max. wavelength difference (nm) that does not trigger expensive resampling
wl_tol = 0.01


### Classes ###


class Instrument:
    def __init__(self, full_config: Config):
        """A model of the spectrometer instrument, including spectral
        response and noise covariance matrices. Noise is typically calculated
        from a parametric model, fit for the specific instrument.  It is a
        function of the radiance level."""

        config = full_config.forward_model.instrument

        # If needed, skip first index column and/or convert to nanometers
        self.wl_init, self.fwhm_init = load_wavelen(config.wavelength_file)
        self.n_chan = len(self.wl_init)

        self.fast_resample = config.fast_resample

        self.bounds = config.statevector.get_all_bounds()
        self.scale = config.statevector.get_all_scales()
        self.init = config.statevector.get_all_inits()
        self.prior_mean = np.array(config.statevector.get_all_prior_means())
        self.prior_sigma = np.array(config.statevector.get_all_prior_sigmas())
        self.statevec_names = config.statevector.get_element_names()
        self.n_state = len(self.statevec_names)

        if config.SNR is not None:
            self.model_type = "SNR"
            self.snr = config.SNR
        elif config.parametric_noise_file is not None:
            self.model_type = "parametric"
            self.noise_file = config.parametric_noise_file
            coeffs = np.loadtxt(self.noise_file, delimiter=" ", comments="#")
            p_a, p_b, p_c = [
                interp1d(coeffs[:, 0], coeffs[:, col], fill_value="extrapolate")
                for col in (1, 2, 3)
            ]
            self.noise = np.array([[p_a(w), p_b(w), p_c(w)] for w in self.wl_init])
            self.integrations = config.integrations

        elif config.pushbroom_noise_file is not None:
            self.model_type = "pushbroom"
            self.noise_file = config.pushbroom_noise_file
            D = loadmat(self.noise_file)
            self.ncols = D["columns"][0, 0]
            if self.n_chan != np.sqrt(D["bands"][0, 0]):
                logging.error("Noise model mismatches wavelength # bands")
                raise ValueError("Noise model mismatches wavelength # bands")
            cshape = (self.ncols, self.n_chan, self.n_chan)
            self.covs = D["covariances"].reshape(cshape)
            self.integrations = config.integrations

        elif config.nedt_noise_file is not None:
            self.model_type = "NEDT"
            self.noise_file = config.nedt_noise_file
            self.noise_data = np.loadtxt(self.noise_file, delimiter=",", skiprows=8)
            noise_data_w_nm = units.micron_to_nm(self.noise_data[:, 0])
            noise_data_NEDT = self.noise_data[:, 1]
            nedt = interp1d(noise_data_w_nm, noise_data_NEDT)(self.wl_init)

            T, emis = 300.0, 0.95  # From Glynn Hulley, 2/18/2020
            _, drdn_dT = emissive_radiance(emis, T, self.wl_init)
            self.noise_NESR = nedt * drdn_dT

        else:
            raise IndexError("Please define the instrument noise.")
            # This should never be reached, as an error is designated in the config read

        # We track several unretrieved free variables, that are specified
        # in a fixed order (always start with relative radiometric
        # calibration)
        self.bvec = ["Cal_Relative_%04i" % int(w) for w in self.wl_init] + [
            "Cal_Spectral",
            "Cal_Stray_SRF",
        ]
        self.bval = np.zeros(self.n_chan + 2)

        if config.unknowns is not None:
            # First we take care of radiometric uncertainties, which add
            # in quadrature.  We sum their squared values.  Systematic
            # radiometric uncertainties account for differences in sampling
            # and radiative transfer that manifest predictably as a function
            # of wavelength.
            if config.unknowns.channelized_radiometric_uncertainty_file is not None:
                f = config.unknowns.channelized_radiometric_uncertainty_file
                u = np.loadtxt(f, comments="#")
                if len(u.shape) > 0 and u.shape[1] > 1:
                    u = u[:, 1]
                self.bval[: self.n_chan] = self.bval[: self.n_chan] + pow(u, 2)

            # Uncorrelated radiometric uncertainties are consistent and
            # independent in all channels.
            if config.unknowns.uncorrelated_radiometric_uncertainty is not None:
                u = config.unknowns.uncorrelated_radiometric_uncertainty
                self.bval[: self.n_chan] = self.bval[: self.n_chan] + pow(
                    np.ones(self.n_chan) * u, 2
                )

            # Radiometric uncertainties combine via Root Sum Square...
            # Be careful to avoid square roots of zero!
            small = np.ones(self.n_chan) * eps
            self.bval[: self.n_chan] = np.maximum(self.bval[: self.n_chan], small)
            self.bval[: self.n_chan] = np.sqrt(self.bval[: self.n_chan])

            # Now handle spectral calibration uncertainties
            if config.unknowns.wavelength_calibration_uncertainty is not None:
                self.bval[-2] = config.unknowns.wavelength_calibration_uncertainty
            if config.unknowns.stray_srf_uncertainty is not None:
                self.bval[-1] = config.unknowns.stray_srf_uncertainty

        # Determine whether the calibration is fixed.  If it is fixed,
        # and the wavelengths of radiative transfer modeling and instrument
        # are the same, then we can bypass computationally expensive sampling
        # operations later.
        self.calibration_fixed = True
        if (
            config.statevector.GROW_FWHM is not None
            or config.statevector.WL_SHIFT is not None
            or config.statevector.WL_SPACE is not None
        ):
            self.calibration_fixed = False

    def dmeas_dinstrumentb(self, meas):
        """Jacobian of radiance with respect to the instrument parameters
        that are unknown and not retrieved, i.e., the inevitable persisting
        uncertainties in instrument spectral and radiometric calibration.

        Input: meas, a vector of size n_chan
        Returns: Kb_instrument, a matrix of size [n_measurements x nb_instrument]
        """
        @partial(
            jax.vmap,
            in_axes=(0, None)
        )
        def hstack(diagflat_rdn, zeros):
            return jnp.hstack((diagflat_rdn, zeros))

        # Uncertainty due to radiometric calibration
        return hstack(jax.vmap(jnp.diagflat)(meas), jnp.zeros((self.n_chan, 2)))

    def Sy(self, meas):
        """
        Only support parametric noise (from file) for now
        """
        noise_plus_meas = self.noise[:, 1] + meas
        nedl = jnp.abs(
            self.noise[:, 0] * jnp.sqrt(noise_plus_meas) + self.noise[:, 2]
        )
        nedl = nedl / jnp.sqrt(self.integrations)

        return jax.vmap(jnp.diagflat)(jnp.power(nedl, 2))
