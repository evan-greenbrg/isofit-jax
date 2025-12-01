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


def lut_grid(lut, lut_names):
    # 1d LUT 
    wl = lut.wl.data
    points = np.array([[point[0], point[1]] for point in lut.point.data])

    # Initialize arrays
    lut = lut.unstack("point")
    lut = lut.transpose(*lut_names, "wl")
    rhoatm = lut['rhoatm'].load().data
    shpalb = lut['sphalb'].load().data
    transm_down_dir = lut['transm_down_dir'].load().data
    transm_down_dif = lut['transm_down_dif'].load().data
    transm_up_dir = lut['transm_up_dir'].load().data
    transm_up_dif = lut['transm_up_dif'].load().data
    dir_dir = lut['dir-dir'].load().data
    dir_dif = lut['dir-dif'].load().data
    dif_dir = lut['dif-dir'].load().data
    dif_dif = lut['dif-dif'].load().data
    #return {'sphalb': sphalb, 'transm': transm, 'rhoatm': rhoatm}

    jpoints = tuple((np.unique(points[:, 0]), np.unique(points[:, 1])))
    int_rhoatm = jax.scipy.interpolate.RegularGridInterpolator(
        jpoints, 
        jnp.array(rhoatm), 
        method='linear'
    )
    int_sphalb = jax.scipy.interpolate.RegularGridInterpolator(
        jpoints, 
        jnp.array(sphalb), 
        method='linear'
    )
    int_transm_down_dir = jax.scipy.interpolate.RegularGridInterpolator(
        jpoints, 
        jnp.array(transm_down_dir), 
        method='linear'
    )
    int_transm_down_dif = jax.scipy.interpolate.RegularGridInterpolator(
        jpoints, 
        jnp.array(transm_down_dif), 
        method='linear'
    )
    int_transm_up_dir = jax.scipy.interpolate.RegularGridInterpolator(
        jpoints, 
        jnp.array(transm_up_dir), 
        method='linear'
    )
    int_transm_up_dif = jax.scipy.interpolate.RegularGridInterpolator(
        jpoints, 
        jnp.array(transm_up_dif), 
        method='linear'
    )
    int_dir_dir = jax.scipy.interpolate.RegularGridInterpolator(
        jpoints, 
        jnp.array(dir_dir), 
        method='linear'
    )
    int_dir_dif = jax.scipy.interpolate.RegularGridInterpolator(
        jpoints, 
        jnp.array(dir_dif), 
        method='linear'
    )
    int_dif_dir = jax.scipy.interpolate.RegularGridInterpolator(
        jpoints, 
        jnp.array(dif_dir), 
        method='linear'
    )
    int_dif_dif = jax.scipy.interpolate.RegularGridInterpolator(
        jpoints, 
        jnp.array(dif_dif), 
        method='linear'
    )

    return {
        'rhoatm': int_rhoatm,
        'sphalb': int_sphalb, 
        'transm_down_dir': int_transm_down_dir,
        'transm_down_dif': int_transm_down_dif,
        'transm_up_dir': int_transm_down_dir,
        'transm_up_dif': int_transm_down_dir,
        'dir_dir': int_dir_dir,
        'dir_dif': int_dir_dif,
        'dif_dir': int_dir_dir,
        'dif_dif': int_dir_dif,
    }


@jax.jit
def calc_rdn(x_surface, x_RT, coszen, cos_i, solar_irr, lut_dict):
    L_down_dir = (
        solar_irr 
        * coszen 
        / np.pi 
        * lut_dict['transm_down_dir'](x_RT.at[:].get())
    )
    L_down_dif = (
        solar_irr 
        * coszen 
        / np.pi 
        * lut_dict['transm_down_dif'](x_RT.at[:].get())
    )
    L_up_dir = (
        solar_irr 
        * coszen 
        / np.pi 
        * lut_dict['transm_up_dir'](x_RT.at[:].get())
    )
    L_up_dif = (
        solar_irr 
        * coszen 
        / np.pi 
        * lut_dict['transm_up_dif'](x_RT.at[:].get())
    )

    L_dir_dir = L_down_dir * L_up_dir / (coszen * cos_i)
    L_dif_dir = L_down_dif * L_up_dir
    L_dir_dif = L_down_dir * L_up_dif / (coszen * cos_i)
    L_dif_dif = L_down_dif * L_up_dif 

    L_atm = (solar_irr * coszen) / jnp.pi * lut_dict["rhoatm"](x_RT)
    sphalb = lut_dict["sphalb"](x_RT)

    rfl = x_surface

    L_tot = L_dir_dir + L_dif_dir + L_dir_dif + L_dif_dif
    # TOA radiance model
    return (
        L_atm
        + L_dir_dir * rfl
        + L_dif_dir * rfl
        + L_dir_dif * rfl
        + L_dif_dif * rfl
        + (L_tot * sphalb * rfl * rfl) / (1 - sphalb * rfl)
    )


def loss_function(x, meas, coszen, cos_i, solar_irr, nwl, lut_dict):

    est_meas = calc_rdn(
        x.at[:nwl].get(),
        x.at[nwl:].get(),
        coszen,
        cos_i,
        solar_irr,
        lut_dict
    )

    resid = (
        est_meas - meas
    )**2

    return jnp.sum(resid)


config_file = '/Users/bgreenbe/Projects/IsofitDev/RTandGlintTests/Water/PriorTests/Prior7/config/emit20240814T104137_isofit.json'
config =  configs.create_new_config(config_file)
fm = ForwardModel(config)
iv = Inversion(config, fm)
wl = jnp.array(fm.RT.wl)


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

rt = initialize_rt(config)
rte = rt.rt_engines[0]
lut = rte.lut
lut_names = rt.rt_engines[0].lut_names
lut = build_interpolators(lut, lut_names)
lut = {key: lut for key, lut in lut.items()}

coszen, cos_i = geom.check_coszen_and_cos_i(rte.coszen)
solar_irr = fm.RT.solar_irr
x = fm.init
nwl = len(wl)

jloss = jax.jit(loss_function, static_argnums=[5])
test = jloss(x, meas, coszen, cos_i, solar_irr, nwl)

meas = np.array(meas.copy())
loss, grad = jax.value_and_grad(jloss)(
    x, meas, coszen, cos_i, solar_irr, nwl
)

ds = lut.unstack("point")
ds = ds.transpose(*lut_names, "wl")
gridarrays = ds['sphalb'].load().data
sphalb = gridarrays

points = np.array([[point[0], point[1]] for point in lut.point.data])
jpoints = tuple((np.unique(points[:, 0]), np.unique(points[:, 1])))
int_sphalb = jax.scipy.interpolate.RegularGridInterpolator(
    jpoints, sphalb, method='linear'
)
query = np.array([0.15, 2.7])
ret = int_sphalb(query)


lut_dict = lut_grid(lut, lut_names)


x = fm.init
x_surface = x[:285]
x_RT = x[285:]
test = calc_rdn(x_surface, x_RT, coszen, cos_i, solar_irr, lut_dict)
jloss = jax.jit(loss_function, static_argnums=[5])
loss, grad = jax.value_and_grad(jloss)(
    x, meas, coszen, cos_i, solar_irr, nwl, lut_dict
)


test = lut_dict['transm_down_dir'](x_RT)
L_down_dir = solar_irr * coszen / np.pi * lut_dict['transm_down_dir'](x_RT)

test = jopt.minimize(
    jloss,
    x,
    (
       jnp.array(meas), 
       jnp.array(coszen), 
       jnp.array(cos_i), 
       jnp.array(solar_irr), 
       nwl, 
       lut_dict
    ),
    method='BFGS'
)

plt.plot(wl, test.x[:len(wl)])
plt.show()

jax.make_jaxpr(calc_rdn)(
    jnp.array(x_surface),
    jnp.array(x_RT),
    jnp.array(coszen),
    jnp.array(cos_i),
    jnp.array(solar_irr),
    lut_dict
)


calc_rdn(
    jnp.array(x_surface),
    jnp.array(x_RT),
    jnp.array(coszen),
    jnp.array(cos_i),
    jnp.array(solar_irr),
    lut_dict
)
