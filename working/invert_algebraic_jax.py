import time
from functools import partial
import os
import itertools

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
from spectral import envi
from jax.sharding import NamedSharding
from jax.sharding import Mesh, PartitionSpec as P
jax.config.update('jax_num_cpu_devices', 15)
# jax.config.update('jax_num_cpu_devices', 1)

from isofit.radiative_transfer.luts import load
from isofit.core.common import envi_header
from isofit.core.geometry import Geometry
from isofit.core.fileio import IO


norm = lambda x, vmin, vmax: (x - vmin) / (vmax - vmin)


def lut_grid(lut, lut_names):
    # 1d LUT 
    wl = lut.wl.data
    # points = np.array([[point[0], point[1]] for point in lut.point.data])
    points = np.array([point for point in lut.point.data])

    # Initialize arrays
    lut = lut.unstack("point")
    lut = lut.transpose(*lut_names, "wl")
    rhoatm = lut['rhoatm'].load().data
    sphalb = lut['sphalb'].load().data
    transm_down_dir = lut['transm_down_dir'].load().data
    transm_down_dif = lut['transm_down_dif'].load().data
    transm_up_dir = lut['transm_up_dir'].load().data
    transm_up_dif = lut['transm_up_dif'].load().data
    dir_dir = lut['dir-dir'].load().data
    dir_dif = lut['dir-dif'].load().data
    dif_dir = lut['dif-dir'].load().data
    dif_dif = lut['dif-dif'].load().data
    #return {'sphalb': sphalb, 'transm': transm, 'rhoatm': rhoatm}

    jpoints = tuple((np.unique(points[:, i]) for i in range(points.shape[1])))
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


@partial(
    jax.jit, 
    static_argnames=[
        'rhoatm', 
        'dir_dir', 'dir_dif', 'dif_dir', 'dif_dif', 
        'sphalb'
    ]
)
def get_RT(point, rhoatm, dir_dir, dir_dif, dif_dir, dif_dif, sphalb):
    return (
        rhoatm(point), 
        ( 
            dir_dir(point)
            + dir_dif(point)
            + dif_dir(point)
            + dif_dif(point)
        ),
        sphalb(point)
    )

@partial(
    jax.jit, 
    static_argnames=[
        'rhoatm', 
        'dir_dir', 'dir_dif', 'dif_dir', 'dif_dif', 
        'sphalb'
    ]
)
def invert_algebraic(
    meas, point, 
    rhoatm, dir_dir, dir_dif, dif_dir, dif_dif, sphalb
):

    return (
        1.0 
        / (
            ((
                dir_dir(point)
                + dir_dif(point)
                + dif_dir(point)
                + dif_dif(point)
            ) / (meas - rhoatm(point)))
            + sphalb(point)
        )
    )


@jax.jit
def check_bounds(vals, minb, maxb):
    return (
        jnp.minimum(
            jnp.maximum(vals, minb),
            maxb,
        )
    )


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

# Jury is out on whether the meshing is doing much...
mesh = jax.make_mesh((15, 1), ('x', 'y'))
a = jax.device_put(meas_list, NamedSharding(mesh, P('x', 'y')))
b = jax.device_put(point_list, NamedSharding(mesh, P('x', 'y')))

# jax.debug.visualize_array_sharding(b)

start_time = time.time()
res = invert_algebraic(
    a,
    b,
    jlut['rhoatm'],
    jlut['dir_dir'],
    jlut['dir_dif'],
    jlut['dif_dir'],
    jlut['dif_dif'],
    jlut['sphalb'],
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")

jax.debug.visualize_array_sharding(res)


res_im = np.reshape(res, rdn_im.shape)
