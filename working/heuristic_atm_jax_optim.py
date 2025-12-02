import time
from functools import partial
import os
import itertools

import click
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

from spectral import envi
import optax

from isofit.radiative_transfer.luts import load
from isofit.core.common import envi_header
from isofit.core.geometry import Geometry
from isofit.core.fileio import IO
# from isofit import ray

from isojax.common import largest_divisible_core
from isojax.lut import lut_grid, check_bounds
from isojax.interpolator import multilinear_interpolator
from isojax.inversions import invert_algebraic, heuristic_atmosphere


def main(lut_path, rdn_file, obs_file, loc_file, fixed_aod=0.1, batchsize=1000):
    lut = load(lut_path)
    lut_names = list(lut.coords)[2:]
    jlut = lut_grid(lut, lut_names)

    rdn = envi.open(envi_header(rdn_file))
    rdn_im = rdn.open_memmap(interleave='bip')
    wl = np.array(rdn.metadata['wavelength']).astype(float)

    obs = envi.open(envi_header(obs_file))
    obs_im = obs.open_memmap(interleave='bip')

    loc = envi.open(envi_header(loc_file))
    loc_im = loc.open_memmap(interleave='bip')

    meas_list = np.reshape(
        rdn_im, 
        (rdn_im.shape[0] * rdn_im.shape[1], rdn_im.shape[2])
    )

    obs_flat = np.reshape(
        obs_im,
        (rdn_im.shape[0] * rdn_im.shape[1], obs_im.shape[2])
    )
    loc_flat = np.reshape(
        loc_im,
        (rdn_im.shape[0] * rdn_im.shape[1], loc_im.shape[2])
    )

    # Construct "Geom"
    temp_aod = 0.1
    temp_h2o = 2.0
    point_list = np.zeros((len(obs_flat), 6))
    point_list[:, 0] = temp_aod
    point_list[:, 1] = temp_h2o
    point_list[:, 2] = obs_flat[:, 2]

    delta_phi = np.abs(obs_flat[:, 3] - obs_flat[:, 1])
    point_list[:, 3] = np.minimum(delta_phi, 360 - delta_phi)
    point_list[:, 4] = obs_flat[:, 4]
    point_list[:, 5] = loc_flat[:, 2]

    bounds = np.array([
        [
            np.min(lut.coords['AOT550'].values),
            np.max(lut.coords['AOT550'].values)
        ],
        [
            np.min(lut.coords['H2OSTR'].values),
            np.max(lut.coords['H2OSTR'].values)
        ],
        [
            np.min(lut.coords['observer_zenith'].values),
            np.max(lut.coords['observer_zenith'].values)
        ],
        [
            np.min(lut.coords['relative_azimuth'].values),
            np.max(lut.coords['relative_azimuth'].values)
        ],
        [
            np.min(lut.coords['solar_zenith'].values),
            np.max(lut.coords['solar_zenith'].values)
        ],
        [
            np.min(lut.coords['surface_elevation_km'].values),
            np.max(lut.coords['surface_elevation_km'].values)
        ],
    ])

    point_list = check_bounds(
        point_list,
        bounds[:, 0],
        bounds[:, 1]
    )

    b865 = np.argmin(abs(wl - 865))
    b945 = np.argmin(abs(wl - 945))
    b1040 = np.argmin(abs(wl - 1040))

    h2os = heuristic_atmosphere(
        jnp.array(np.unique(lut.coords['H2OSTR'].data)),
        meas_list, 
        point_list, 
        jlut, 
        fix_aod=fixed_aod, 
        batchsize=batchsize, 
        b865=b865,
        b945=b945,
        b1040=b1040
    )

    h2os = np.reshape(h2os, (rdn.shape[0], rdn.shape[1]))

    # Save file


@click.command()



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

def step(jlut, b865=65, b945=76, b1040=88):
    loss_fn = lambda v865, v945, v1040: (
        jnp.abs(1 - ((v945 * 2.) / (v1040 + v865)))
    )
    @jax.jit
    def loss(param, meas, point):

        point = point.at[1].set(param)
        pred = invert_algebraic(
              meas, 
              jlut['rhoatm'](point[jnp.newaxis, :]),
              jlut['dir_dir'](point[jnp.newaxis, :]),
              jlut['dir_dif'](point[jnp.newaxis, :]),
              jlut['dif_dir'](point[jnp.newaxis, :]),
              jlut['dif_dif'](point[jnp.newaxis, :]),
              jlut['sphalb'](point[jnp.newaxis, :]),
        )
        
        return jnp.mean(loss_fn(pred[:, b865], pred[:, b945], pred[:, b1040]))
    
    return jax.value_and_grad(loss)



# Implementation in serial -> Couldn't get ray to get much faster
n = 100000
indices = np.arange(n, len(meas_list), n)
meas_batches = np.array_split(meas_list, indices, axis=0)
point_batches = np.array_split(point_list, indices, axis=0)

step_fn = step(jlut)
batch_fn = jax.vmap(step_fn)

learning_rate = 2e-1
optimizer = optax.adam(learning_rate)

@jax.jit
def fit(meas_batch, point_batch, params, opt_state):

    vals, grads = batch_fn(params, meas_batch, point_batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state

h2os = []
ites = 0
nsteps=22
for meas_batch, point_batch in zip(meas_batches, point_batches):
    params = jnp.ones((meas_batch.shape[0])) * 1.0

    mesh = jax.make_mesh((largest_divisible_core(len(meas_batch)),), ('x'))
    meas_batch_d = jax.device_put(meas_batch, NamedSharding(mesh, P('x')))
    point_batch_d = jax.device_put(point_batch, NamedSharding(mesh, P('x')))
    params_d = jax.device_put(params, NamedSharding(mesh, P('x')))

    opt_state = optimizer.init(params)
    for j in range(nsteps):
        print(f"Step: {j} of {nsteps}")
        params_d, opt_state = fit(meas_batch_d, point_batch_d, params_d, opt_state)

    print("Done fit")
    h2os.append(params_d)

h2os = np.concatenate(h2os)
h2o_opt_im = np.reshape(h2os, (rdn_im.shape[0], rdn_im.shape[1]))

# Implementation number 2. Rather than optimization. Full grid search.
nbatches=1
meas_batch = np.array_split(meas_list, nbatches, axis=0)[0]
point_batch = np.array_split(point_list, nbatches, axis=0)[0]

# Set up multiprocessing
mesh = jax.make_mesh((10,), ('x'))
meas_batch_d = jax.device_put(meas_batch, NamedSharding(mesh, P('x')))
point_batch_d = jax.device_put(point_batch, NamedSharding(mesh, P('x')))

ratio_fn = lambda v865, v945, v1040: (
    jnp.abs(1 - ((v945 * 2.) / (v1040 + v865)))
)
h2o_grid = np.unique(lut.coords['H2OSTR'].data)
ratios = []
for i, h2o in enumerate(h2o_grid):
    print(h2o)
    point_batch_d = point_batch_d.at[:, 1].set(h2o)

    alg = invert_algebraic(
        meas_batch_d,
        jlut['rhoatm'](point_batch_d),
        jlut['dir_dir'](point_batch_d),
        jlut['dir_dif'](point_batch_d),
        jlut['dif_dir'](point_batch_d),
        jlut['dif_dif'](point_batch_d),
        jlut['sphalb'](point_batch_d),
    )
    ratios.append(ratio_fn(alg[:, b865], alg[:, b945], alg[:, b1040]))

ratios = np.array(ratios).T
h2o = h2o_grid[np.argmin(ratios, axis=1)]
h2o_im = np.reshape(h2o, (rdn_im.shape[0], rdn_im.shape[1], 1))


h2os = np.concatenate(h2os)
h2o_opt_im = np.reshape(h2os, (rdn_im.shape[0], rdn_im.shape[1]))


fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(h2o_im)
axs[1].imshow(h2o_opt_im)
plt.show()


ites = [i for i in range(ratios.shape[1])]
plt.scatter(ites, ratios[100, :])
plt.scatter(ites, ratios[200, :])
plt.scatter(ites, ratios[300, :])
plt.show()

from copy import deepcopy
base = deepcopy(alg[100, :])

plt.plot(wl, base)
plt.plot(wl, alg[100, :])
plt.ylim([-.05, .3])
plt.show()
