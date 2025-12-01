import logging
import os
from functools import partial
import time

import numpy as np
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from spectral import envi

from isofit.radiative_transfer.luts import load
from isofit.core.common import envi_header
from isofit.core.geometry import Geometry
from isofit.inversion.inverse import Inversion
from isofit.configs import configs
from isofit.core.fileio import IO
from isofit import ray
from isofit.core.common import svd_inv
from isofit.core.common import load_wavelen

from isojax.common import largest_divisible_core
from isojax.lut import lut_grid, check_bounds
from isojax.interpolator import interp1d
from isojax.surface import MultiComponentSurface

from isojax.forward import ForwardModel as ForwardModel
from isofit.core.forward import ForwardModel as ISOForwardModel

from isofit.inversion.inverse_simple import heuristic_atmosphere as iso_heuristic_atmosphere 
from isofit.inversion.inverse_simple import invert_algebraic as iso_invert_algebraic
from isofit.inversion.inverse_simple import invert_analytical as iso_invert_analytical


def retrieve_winidx(config):
    wl_init, fwhm_init = load_wavelen(config.forward_model.instrument.wavelength_file)
    windows = config.implementation.inversion.windows

    winidx = np.array((), dtype=int)
    for lo, hi in windows:
        idx = np.where(np.logical_and(wl_init > lo, wl_init < hi))[0]
        winidx = np.concatenate((winidx, idx), axis=0)

    return winidx


def symv(alpha, A, x, beta=0, y=0):
    return alpha * (A @ x) + beta * y


@jax.vmap
def vsymv(c_rcond, p, h, meas_i, l_atm, prprod_i):

    return symv(
        1, 
        c_rcond, 
        h.T @ symv(1, p, meas_i[winidx] - l_atm[winidx]) + prprod_i
    )

def dpotri_jax(L, lower=True):
    n = L.shape[0]
    I = jnp.eye(n, dtype=L.dtype)
    # cho_solve solves A X = B using the Cholesky factor.
    inv_A = jax.scipy.linalg.cho_solve((L, lower), I)
    return inv_A

@jax.jit
def invert_algebraic(
    meas, rhoatm, dir_dir, dir_dif, dif_dir, dif_dif, sphalb
):

    return (
        1.0 
        / (
            ((
                dir_dir
                + dir_dif
                + dif_dir
                + dif_dif
            ) / (meas - rhoatm)
            + sphalb)
        )
    )


def heuristic_atmosphere(h2o_grid, meas_list, point_list, jlut, fix_aod=0.1, batchsize=100000, b865=65, b945=76, b1040=88):

    point_list = point_list.at[:, 0].set(fix_aod)

    indices = np.arange(batchsize, len(meas_list), batchsize)
    meas_batches = np.array_split(meas_list, indices, axis=0)
    point_batches = np.array_split(point_list, indices, axis=0)
    h2os = []
    i = 0
    for i, (meas_batch, point_batch) in enumerate(zip(meas_batches, point_batches)):
        print(f"Batch: {i} of {len(meas_batches)}")

        mesh_size = largest_divisible_core(len(meas_batch))
        mesh = jax.make_mesh((mesh_size,), ('x'))
        meas_d = jax.device_put(meas_batch, NamedSharding(mesh, P('x')))
        point_d = jax.device_put(point_batch, NamedSharding(mesh, P('x')))

        ratio_fn = lambda alg: (
            (1 - ((alg[:, b945] * 2.) / (alg[:, b1040] + alg[:, b865])))
        )

        @jax.jit
        def get_ratio(h2o, point):
            point = point.at[:, 1].set(h2o)
            return ratio_fn(invert_algebraic(
                meas_d,
                jlut['rhoatm'](point),
                jlut['dir_dir'](point),
                jlut['dir_dif'](point),
                jlut['dif_dir'](point),
                jlut['dif_dif'](point),
                jlut['sphalb'](point),
            ))


        def min1d(grid, values):
            """Not actually a minimization. Just a zero-crossing"""
            return (
                interp1d(values[jnp.argsort(values)], grid[jnp.argsort(values)])(0)
            )
        
        min1d_batch = jax.vmap(min1d, in_axes=(None, 1))
        get_ratio_vmap = jax.vmap(get_ratio, in_axes=(0, None))

        ratios = get_ratio_vmap(h2o_grid, point_d)
        h2os.append(min1d_batch(jnp.array(h2o_grid), ratios))

        i += 1

    return np.concatenate(h2os)

@partial(
    jax.vmap,
    in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
)
def invert_analytical_point(fm, x0_i, meas_i, sub_i, point_i, ci_i, 
                            lamb_norm_i, L_atm, L_dir_dir,
                            L_dir_dif, L_dif_dir, L_dif_dif,
                            s_alb, winidx):

    rho_dir_dir, rho_dif_dir = fm.surface.calc_rfl(sub_i[jnp.newaxis, :])
    background = jnp.multiply(rho_dif_dir, s_alb[jnp.newaxis, :])
    H = fm.surface.analytical_model(
        background, 
        (L_dir_dir + L_dir_dif + L_dif_dir + L_dif_dif)[jnp.newaxis]
    )[:, winidx, :]
    Seps = fm.Seps(
        x0_i[jnp.newaxis, :],
        meas_i[jnp.newaxis, :],
        point_i[jnp.newaxis, :],
        jlut
    )[:, winidx, :][..., winidx]
    Sa, Sa_inv, Sa_inv_sqrt = fm.surface.Sa(ci_i, lamb_norm_i)
    xa = fm.surface.xa(x0_i[jnp.newaxis, :], lamb_norm_i)
    prprod = jax.vmap(jnp.matmul)(Sa_inv[jnp.newaxis, ...], xa)

    C = jax.lax.linalg.cholesky(Seps)
    L = jax.vmap(dpotri_jax)(C)

    P_tilde = jax.vmap(lambda h, p: ((h.T @ p) @ h).T)(H, L)
    P_rcond = Sa_inv + P_tilde

    LI_rcond = jax.lax.linalg.cholesky(P_rcond)
    C_rcond = jax.vmap(dpotri_jax)(LI_rcond)
    C_rcond_sym = jax.vmap(lambda c: (c + c.T) * .5)(C_rcond)

    # @jax.vmap
    def vsymv(c_rcond, p, h, meas_i, l_atm, prprod_i):
        return symv(
            1, 
            c_rcond, 
            h.T @ symv(1, p, meas_i[winidx] - l_atm[winidx]) + prprod_i
        )
    return vsymv(
        C_rcond[0, ...],
        L[0, ...],
        H[0, ...],
        meas_i,
        L_atm,
        prprod[0, ...]
    )


# @partial(
#     jax.jit,
#     static_argnames=['fm']
# )
def invert_analytical(fm, x0, meas, sub, points, ci, lamb_norm, seps_d,
                      L_atm_d, L_tot_d, s_alb_d, winidx):
    # Calculate background
    rho_dir_dir, rho_dif_dir = fm.surface.calc_rfl(sub)
    background = jnp.multiply(rho_dif_dir, s_alb_d)

    H = fm.surface.analytical_model(
        background, 
        L_tot_d
    )[:, winidx, :]

    Sa, Sa_inv, Sa_inv_sqrt = fm.surface.Sa(ci, lamb_norm)
    xa = fm.surface.xa(x0, lamb_norm)
    prprod = jax.vmap(jnp.matmul)(Sa_inv, xa)

    C = jax.lax.linalg.cholesky(seps_d)
    L = jax.vmap(dpotri_jax)(C)

    P_tilde = jax.vmap(lambda h, p: ((h.T @ p) @ h).T)(H, L)
    P_rcond = Sa_inv + P_tilde

    LI_rcond = jax.lax.linalg.cholesky(P_rcond)
    C_rcond = jax.vmap(dpotri_jax)(LI_rcond)
    C_rcond_sym = jax.vmap(lambda c: (c + c.T) * .5)(C_rcond)

    @jax.vmap
    def vsymv(c_rcond, p, h, meas_i, l_atm, prprod_i):

        return symv(
            1, 
            c_rcond, 
            h.T @ symv(1, p, meas_i[winidx] - l_atm[winidx]) + prprod_i
        )

    return vsymv(C_rcond, L, H, meas, L_atm_d, prprod)



config_path = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock_6c/config/emit20220818t205752_isofit.json'
config = configs.create_new_config(config_path)
fm = ForwardModel(config)
winidx = retrieve_winidx(config)

lut_path = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock_6c/lut_full/lut.nc'
lut = load(lut_path)
wl = lut.wl.data
points = np.array([point for point in lut.point.data])
lut_names = list(lut.coords)[2:]
jlut = lut_grid(lut, lut_names)
rdn_file = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_rdn_b0106_v01.img'
obs_file = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_obs_b0106_v01.img'
loc_file = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_loc_b0106_v01.img'

norm = lambda x, vmin, vmax: (x - vmin) / (vmax - vmin)
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

fix_aod = 0.1
h2os = heuristic_atmosphere(
    np.unique(lut.coords['H2OSTR'].data), 
    meas_list, 
    point_list, 
    jlut, 
    fix_aod=fix_aod, 
    batchsize=100000, 
    b865=b865,
    b945=945,
    b1040=1040
)

point_list = point_list.at[:, 0].set(fix_aod)
point_list = point_list.at[:, 1].set(h2os)
# invert_algebraic = jax.vmap(invert_algebraic)
x0 = invert_algebraic(
    meas_list,
    jlut['rhoatm'](point_list),
    jlut['dir_dir'](point_list),
    jlut['dir_dif'](point_list),
    jlut['dif_dir'](point_list),
    jlut['dif_dif'](point_list),
    jlut['sphalb'](point_list),
)

surface = MultiComponentSurface(config)
x0_norm = jnp.linalg.norm(x0[:, surface.idx_ref], axis=1)
xa = surface.xa(x0, x0_norm[:, jnp.newaxis])
cis = surface.component(x0, x0_norm[:, jnp.newaxis])
sub = x0

h2os_np = np.array(h2os)
aods = np.ones(h2os_np.shape) * 0.1
atms = jnp.array([aods, h2os]).T


@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(self, fm, jlut, winidx, 
                 loglevel='INFO', logfile=None):
        logging.basicConfig(
            format="%(levelname)s:%(asctime)s ||| %(message)s",
            level=loglevel,
            filename=logfile,
            datefmt="%Y-%m-%d,%H:%M:%S",
        )

        self.fm = fm

        # TODO Look into ways to make this pre-compiled functions
        # shared across workers
        self.inv = jax.jit(invert_analytical, static_argnums=0)
        self.jlut = jlut
        self.winidx = winidx

    def run(self, x0_d, meas_d, sub_d, point_d, ci_d, lamb_norm_d):
        seps_d = self.fm.Seps(
            x0_d, meas_d, point_d, self.jlut
        )[:, self.winidx, :][..., self.winidx]

        L_atm_d = self.jlut['rhoatm'](point_d)
        L_dir_dir_d = self.jlut['dir_dir'](point_d)
        L_dir_dif_d = self.jlut['dir_dif'](point_d)
        L_dif_dir_d = self.jlut['dif_dir'](point_d)
        L_dif_dif_d = self.jlut['dif_dif'](point_d)
        s_alb_d = self.jlut['sphalb'](point_d)

        logging.info("Running")
        res = self.inv(
            self.fm,
            x0_d,
            meas_d,
            sub_d,
            point_d,
            ci_d,
            lamb_norm_d, 
            seps_d,
            L_atm_d,
            (L_dir_dir_d + L_dir_dif_d + L_dif_dir_d + L_dif_dif_d),
            s_alb_d,
            self.winidx
        )
        res.block_until_ready()
        logging.info("Finished")
        return res


def analytical_line_multi(x0, meas_list, sub, point_list, cis, x0_norm, 
    batchsize=1000):

    # TODO This is still taking too long.
    # Not sure how to debug it...

    # Set atms
    point_list = point_list.at[:, :2].set(atms)

    # Set up batching
    indices = np.arange(batchsize, len(meas_list), batchsize)
    x0_batches = np.array_split(x0, indices, axis=0)
    meas_batches = np.array_split(meas_list, indices, axis=0)
    sub_batches = np.array_split(sub, indices, axis=0)
    point_batches = np.array_split(point_list, indices, axis=0)
    ci_batches = np.array_split(cis, indices, axis=0)
    lamb_norm_batches = np.array_split(x0_norm[:, jnp.newaxis], indices, axis=0)

    # Priors never change
    # inv = jax.jit(invert_analytical_point, static_argnums=0)
    inv = None
    inv = jax.jit(invert_analytical, static_argnums=0)

    n_workers = 10
    ray.init(num_cpus=n_workers)
    # Put worker args into Ray object
    params = [ray.put(fm), ray.put(inv), ray.put(jlut), ray.put(winidx)]
    workers = ray.util.ActorPool(
        [Worker.remote(*params) for n in range(n_workers)]
    )

    inputs = [
        [x0_d, meas_d, sub_d, point_d, ci_d, lamb_norm_d]
        for x0_d, meas_d, sub_d, point_d, ci_d, lamb_norm_d
        in zip(
            x0_batches,
            meas_batches,
            sub_batches,
            point_batches,
            ci_batches,
            lamb_norm_batches
        )
    ]
    start_time = time.time()
    res = list(
        workers.map_unordered(
            lambda a, b: a.run.remote(*b), inputs
        )
    )
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    inv_ref = ray.put(inv)
    fm_ref = ray.put(fm)

    result_refs = [
        ray_analytical.remote(
            fm_ref,
            inv_ref,
            jlut_ref,
            x0_batches[i],
            meas_batches[i],
            sub_batches[i],
            point_batches[i],
            ci_batches[i],
            lamb_norm_batches[i], 
            L_atm,
            L_dir_dir,
            L_dir_dif,
            L_dif_dir,
            L_dif_dif,
            s_alb,
            winidx
        ) for i in range(len(indices))
    ]
    lp = np.concatenate(ray.get(result_refs), axis=0)

    aoe = []
    total_start = time.time()
    for i, index in enumerate(indices):
        start_time = time.time()
        print(f"Batch: {i} of {len(meas_batches)}")
        L_atm = jlut['rhoatm'](point_batches[i])
        L_dir_dir = jlut['dir_dir'](point_batches[i])
        L_dir_dif = jlut['dir_dif'](point_batches[i])
        L_dif_dir = jlut['dif_dir'](point_batches[i])
        L_dif_dif = jlut['dif_dif'](point_batches[i])
        s_alb = jlut['sphalb'](point_batches[i])

        res = inv(
            fm,
            x0_batches[i],
            meas_batches[i],
            sub_batches[i],
            point_batches[i],
            ci_batches[i],
            lamb_norm_batches[i], 
            L_atm,
            L_dir_dir,
            L_dir_dif,
            L_dif_dir,
            L_dif_dif,
            s_alb,
            winidx
        )
        res.block_until_ready()

        aoe.append(res)
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time}")

    aoe = np.concatenate(aoe)
    total_end = time.time()
    print(f"Total elapsed time: {total_end - total_start}")

