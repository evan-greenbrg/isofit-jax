import sys
import logging
from surface import SurfaceWrapper
import multiprocessing
import os
import time
from collections import OrderedDict
from glob import glob

import click
from matplotlib import pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from spectral.io import envi
from scipy.optimize import least_squares
import scipy

from isofit import ray
from isofit.configs import configs
from isofit.core.common import envi_header, load_spectrum, svd_inv_sqrt
from isofit.core.common import conditional_gaussian, eps
from isofit.core.fileio import IO, write_bil_chunk
from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.inversion.inverse import Inversion
from isofit.inversion.inverse_simple import invert_analytical
from isofit.inversion.inverse_simple import invert_simple
from isofit.radiative_transfer.radiative_transfer import RadiativeTransfer
from isofit.radiative_transfer import luts

import jax
jax.config.update('jax_num_cpu_devices', 15)

import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax import tree_util
import jax.scipy.optimize as jopt
import jaxopt

from rte import RTEcalc, build_interpolators, VectorInterpolator, calc_rdn, RTEparams
from surface import SurfaceParams, SurfaceWrapper


tree_util.register_pytree_node(
    RTEcalc, 
    RTEcalc._tree_flatten, 
    RTEcalc._tree_unflatten
)

tree_util.register_pytree_node(
    SurfaceWrapper, 
    SurfaceWrapper._tree_flatten, 
    SurfaceWrapper._tree_unflatten
)


class JForward:
    def __init__(self, rte_calc, rte, surface, idx_surface, idx_RT):
        self.rte_calc = rte_calc
        self.geom_values = list(rte.indices.geom.values())
        self.surface = surface
        self.idx_surface = idx_surface
        self.idx_RT = idx_RT

    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {
            'rte_calc': self.rte_calc,
            'geom_values': self.geom_values,
            'surface': self.surface,
            'idx_surface': self.idx_surface,
            'idx_RT': self.idx_RT,
        }  # static values
        
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def calc_meas(self, x, coszen, cos_i, geom_values):

        L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif, L_atm, sphalb = self.rte_calc.get_L(
            x.at[self.idx_RT].get(),
            self.geom_values,
            coszen,
            cos_i
        )

        rfl = x_surface

        return calc_rdn(
            rfl,
            sphalb,
            L_atm,
            L_dir_dir,
            L_dif_dir,
            L_dir_dif,
            L_dif_dif,
        )

    def loss_function(self, x, coszen, cos_i, seps_inv_sqrt, xa, sa_inv_sqrt):
        # set up full-sized state vector

        # Measurement cost term.  Will calculate reflectance and Ls from
        # the state vector.

        est_meas = jax.jit(self.calc_meas)(
            jnp.array(x),
            coszen, 
            cos_i
        )
        est_meas_window = est_meas[winidx]
        meas_window = meas[winidx]
        meas_resid = (
            est_meas_window - meas_window
        ).dot(seps_inv_sqrt)

        # Prior cost term
        prior_resid = (x - xa).dot(sa_inv_sqrt)

        # Total cost
        total_resid = jnp.concatenate((meas_resid, prior_resid))

        return jnp.sum(total_resid)

    @staticmethod
    def negative_check(val):
        return np.max([val, 1e-5])

    def calc_seps(self, noise_plus_meas, noise, integrations, winidx):
        """
        Just the measurement noise
        """
        
        nedl = jnp.abs(
            noise[:, 0] * jnp.sqrt(noise_plus_meas) + noise[:, 2]
        )
        nedl = nedl / jnp.sqrt(integrations)

        seps = jnp.diagflat(jnp.power(nedl, 2))

        wn = len(winidx)
        seps_win = jnp.zeros((wn, wn))
        for i in range(wn):
            seps_win = seps_win.at[i].set(
                seps.at[winidx[i]].get()[winidx]
            )

        return seps_win

    def calc_conditional_prior(self, x_surface, rte_prior_mean, rte_prior_sigma):
        xa = rte_calc.xa(rte_calc.prior_mean)
        xa = jnp.concatenate((
            surface.xa(x_surface), 
            rte_calc.xa(rte_prior_mean),
        ), axis=0)
        Sa = jax.scipy.linalg.block_diag(
            surface.Sa(x_surface), 
            rte_calc.Sa(rte_prior_sigma),
        )

        Sa_inv, Sa_inv_sqrt = svd_inv_sqrt(Sa)

        return xa, Sa, Sa_inv, Sa_inv_sqrt

    def unpack(self, x):
        """Unpack the state vector in appropriate index ordering."""

        x_surface = x[self.idx_surface]
        x_RT = x[self.idx_RT]
        return x_surface, x_RT


def initialize_rt(config):
    """
    Wrapper to initialize the Isofit RT.
    This will kick off the RTE
    """
    return RadiativeTransfer(config)


@jax.jit
def svd_inv_sqrt(C):

    D, P = jax.scipy.linalg.eigh(C)

    # Removing check - See if this bites me
    Ds = jnp.diag(1 / jnp.sqrt(D))

    L = P @ Ds
    Cinv_sqrt = L @ P.T
    Cinv = L @ L.T

    return Cinv, Cinv_sqrt



config_file = '/Users/bgreenbe/Projects/IsofitDev/RTandGlintTests/Water/PriorTests/Prior7/config/emit20240814T104137_isofit.json'
config =  configs.create_new_config(config_file)
fm = ForwardModel(config)
iv = Inversion(config, fm)
wl = fm.RT.wl


esd = IO.load_esd()
in_root = '/Users/bgreenbe/Projects/IsofitDev/RTandGlintTests/Water/PriorTests/Sample1'
sub_root = '/Users/bgreenbe/Projects/IsofitDev/RTandGlintTests/Water/PriorTests/Prior7/input'
out_root = '/Users/bgreenbe/Projects/IsofitDev/RTandGlintTests/Water/PriorTests/Prior7/output'

obs = envi.open(envi_header(os.path.join(
    in_root, 'emit20240814T104137_obs'
)))
obs_im = obs.open_memmap(interleave="bip")

loc = envi.open(envi_header(os.path.join(
    in_root, 'emit20240814T104137_loc'
)))
loc_im = loc.open_memmap(interleave="bip")

rdn = envi.open(envi_header(os.path.join(
    in_root, 'emit20240814T104137_rdn'
)))
rdn_im = rdn.open_memmap(interleave="bip")

sub_obs = envi.open(envi_header(os.path.join(
    sub_root, 'emit20240814T104137_subs_obs'
)))
sub_obs_im = sub_obs.open_memmap(interleave="bip")

sub_loc = envi.open(envi_header(os.path.join(
    sub_root, 'emit20240814T104137_subs_loc'
)))
sub_loc_im = sub_loc.open_memmap(interleave="bip")

sub_rdn = envi.open(envi_header(os.path.join(
    sub_root, 'emit20240814T104137_subs_rdn'
)))
sub_rdn_im = sub_rdn.open_memmap(interleave="bip")

lbl = envi.open(envi_header(os.path.join(
    out_root,
    'emit20240814T104137_lbl'
)))
lbl_im = lbl.open_memmap(interleave="bip")

atm= envi.open(envi_header(os.path.join(
    out_root,
    'emit20240814T104137_atm_interp'
)))
atm_im = atm.open_memmap(interleave="bip")

r, c = [100, 100]
sr = int(lbl_im[r, c])
sub_meas = sub_rdn_im[sr, 0, :]
sub_geom = Geometry(
    obs=sub_obs_im[sr, 0, :], 
    loc=sub_loc_im[sr, 0, :], 
    esd=esd
)
meas = rdn_im[r, c, :]
geom = Geometry(
    obs=obs_im[r, c, :], 
    loc=loc_im[r, c, :], 
    esd=esd
)

x = fm.init
x_surface, x_RT, x_instrument = fm.unpack(np.copy(x))

rt = initialize_rt(config)
rte = rt.rt_engines[0]
lut = rte.lut
lut_names = rt.rt_engines[0].lut_names
lut = build_interpolators(lut, lut_names)

rte_params = RTEparams(config)

rte_calc = RTEcalc(
    rte.indices.x_RT,
    lut,
    lut_names,
    list(rte.indices.geom.keys()),
    rt.solar_irr,
    rte.rt_mode,
    rte_params.prior_mean,
    rte_params.prior_sigma,
)

coszen, cos_i = geom.check_coszen_and_cos_i(rte.coszen)
L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif, L_atm, sphalb = rte_calc.get_L(
    x_RT,
    list(rte.indices.geom.values()),
    coszen,
    cos_i
)

xa = rte_calc.xa(rte_calc.prior_mean)
Sa = rte_calc.Sa(rte_calc.prior_sigma)

surface_params = SurfaceParams(config)
surface = SurfaceWrapper(
    surface_params.idx_lamb,
    surface_params.idx_ref,
    surface_params.n_state,
    surface_params.prior_means,
    surface_params.prior_covs,
    surface_params.norm,
    surface_params.n_comp,
    surface_params.statevec_names,
    surface_params.mus
)
coszen, cos_i = geom.check_coszen_and_cos_i(rte.coszen)
jfm = JForward(rte_calc, rte, surface, fm.idx_surface, fm.idx_RT)

windows = config.implementation.inversion.windows
winidx = jnp.array((), dtype=int)  # indices of retrieval windows
for lo, hi in windows:
    idx = jnp.where(
        jnp.logical_and(
            fm.instrument.wl_init > lo, fm.instrument.wl_init < hi
        )
    )[0]
    winidx = np.concatenate((winidx, idx), axis=0)

integrations = fm.instrument.integrations
noise = fm.instrument.noise


noise_plus_meas = noise[:, 1] + meas
noise_plus_meas = jnp.array(
    jax.tree.map(jfm.negative_check, list(noise_plus_meas))
)

seps_win = jax.jit(jfm.calc_seps)(
    noise_plus_meas, noise, integrations, winidx
)
seps_inv, seps_inv_sqrt = svd_inv_sqrt(seps_win)

xa, sa, sa_inv, sa_inv_sqrt = jax.jit(
    jfm.calc_conditional_prior
)(x_surface, rte_calc.prior_mean, rte_calc.prior_sigma)


solver = jaxopt.ScipyBoundedMinimize(fun=jfm.loss_function, method="SLSQP")
test = solver.run(x, [-1, 1], coszen, cos_i, seps_inv_sqrt, xa, sa_inv_sqrt)

test = jopt.minimize(
    jax.jit(jfm.loss_function),
    x,
    (coszen, cos_i, seps_inv_sqrt, xa, sa_inv_sqrt),
    method='BFGS'
)

loss = jfm.loss_function(x, coszen, cos_i, seps_inv_sqrt, xa, sa_inv_sqrt)


plt.plot(wl, test.x[:len(wl)])
plt.show()

est_meas = jax.jit(self.calc_meas)(
    jnp.array(x),
    coszen, 
    cos_i,
    []
)

test = jax.jacobian(jfm.calc_meas)
test = jax.grad(jfm.calc_meas, argnums=0)
g = test(x, coszen, cos_i)

iv_res = iv.invert(meas, geom)

losses = []
test = ((res), (coszen), (cos_i), (seps_inv_sqrt), (xa), (sa_inv_sqrt))
tree = [
    [res] + [coszen, cos_i, seps_inv_sqrt, xa, sa_inv_sqrt] for res in iv_res
]
res_map = jax.tree.map(
    jfm.loss_function, 
    [res for res in iv_res], 
    [coszen for i in iv_res], 
    [cos_i for i in iv_res], 
    [seps_inv_sqrt for i in iv_res], 
    [xa for i in iv_res], 
    [sa_inv_sqrt for i in iv_res]
)

meas_jac = jax.jacobian(jfm.calc_meas, argnums=0)
jac_map = jax.tree.map(
    meas_jac,
    [res for res in iv_res],
    [coszen for i in iv_res], 
    [cos_i for i in iv_res], 
)

for jac in jac_map:
    plt.plot(wl, np.diag(jac))
plt.show()


L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif, L_atm, sphalb = self.rte_calc.get_L(
    x.at[self.idx_RT].get(),
    self.geom_values,
    coszen,
    cos_i
)

test = jax.jacobian(rte_calc.get_L)
out = test(
    x[jfm.idx_RT],
    jfm.geom_values,
    coszen,
    cos_i
)

r = jax.jacobian(rte_calc.get_rtm_quantities)

# iso_jac = iv.jacobian(x, geom, seps_inv_sqrt)[:len(wl), :]
iso_jac = fm.K(x, geom)
jax_jac = meas_jac(x, coszen, cos_i)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(iso_jac)
axs[1].imshow(jax_jac)
plt.show()


# TODO Questions:
# Why isn't the minimize converging properly?
# Why isn't the gradient out of rte_calc.get_L correct?


@jax.jit
def test(x_RT, coszen, cos_i, geom_values):
    (
        L_dir_dir, 
        L_dif_dir, 
        L_dir_dif, 
        L_dif_dif, 
        L_atm, 
        sphalb 
    ) = rte_calc.get_L(
        x_RT,
        geom_values,
        coszen,
        cos_i
    )

    return L_dir_dir

x_RT  = [.4, 2.87]
geom_values = list(rte.indices.geom.values())
L = test(x_RT, coszen, cos_i, geom_values)
print(L[-1])

calc_meas = jax.jit(jfm.calc_meas)
calc_meas_jac = jax.jacobian(calc_meas)
g = calc_meas_jac(x, coszen, cos_i)

eps = 1e-4

test = jax.grad(jfm.loss_function)
g = test(x, coszen, cos_i, seps_inv_sqrt, xa, sa_inv_sqrt)

K = []
x_perturbs = x + np.eye(len(x)) * eps
loss = jfm.loss_function(x, coszen, cos_i, seps_inv_sqrt, xa, sa_inv_sqrt)
for x_perturb in list(x_perturbs):
    losse = jfm.loss_function(x_perturb, coszen, cos_i, seps_inv_sqrt, xa, sa_inv_sqrt)
    K.append((losse - loss) / eps)

isoK = fm.K(x, geom)
J = iv.jacobian(x, geom, seps_inv_sqrt)

plt.plot(np.diag(isoK))
plt.plot(np.diag(g))
plt.show()

diff = np.array(isoK - g)
diff[diff == 0] = None
plt.imshow(diff)
plt.show()



# Can I write a version without the priors?
# Carry the LUT slightly differently that allows for it to be passed into the function.
