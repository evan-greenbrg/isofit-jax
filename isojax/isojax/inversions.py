from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from isofit.core.common import load_wavelen

from isojax.interpolator import interp1d


def retrieve_winidx(config):
    wl_init, fwhm_init = load_wavelen(
        config.forward_model.instrument.wavelength_file
    )
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
            (
                (
                    dir_dir
                    + dir_dif
                    + dif_dir
                    + dif_dif
                ) / (meas - rhoatm)
                + sphalb
            )
        )
    )


def heuristic_atmosphere(h2o_grid, meas_list, point_list, jlut,
                         fix_aod=0.1, batchsize=100000,
                         b865=65, b945=76, b1040=88, nshard=1):

    def ratio_fn(alg):
        return (
            1 - ((alg[:, b945] * 2.) / (alg[:, b1040] + alg[:, b865]))
        )

    @jax.jit
    def get_ratio(h2o, point, meas):
        point = point.at[:, 1].set(h2o)
        return ratio_fn(invert_algebraic(
            meas,
            jlut['rhoatm'](point),
            jlut['dir_dir'](point),
            jlut['dir_dif'](point),
            jlut['dif_dir'](point),
            jlut['dif_dif'](point),
            jlut['sphalb'](point),
        ))

    @jax.jit
    def min1d(values):
        """Not actually a minimization. Just a zero-crossing"""
        return (
            interp1d(
                values[jnp.argsort(values)],
                h2o_grid[jnp.argsort(values)]
            )(0)
        )

    get_ratio_vmap = jax.vmap(get_ratio, in_axes=(0, None, None))
    get_ratio_pmap = jax.pmap(get_ratio_vmap, in_axes=(None, 0, 0))

    min1d_vmap = jax.vmap(min1d, in_axes=(1))
    min1d_pmap = jax.pmap(min1d_vmap, in_axes=(0))

    point_list = point_list.at[:, 0].set(fix_aod)

    indices = np.arange(batchsize, len(meas_list), batchsize)
    meas_batches = np.array_split(meas_list, indices, axis=0)
    point_batches = np.array_split(point_list, indices, axis=0)

    last_meas_batch = meas_batches[-1]
    last_point_batch = point_batches[-1]

    meas_batches = np.array(meas_batches[:-1])
    point_batches = np.array(point_batches[:-1])

    indices = np.arange(nshard, len(meas_batches), nshard)
    meas_shards = np.array_split(meas_batches, indices, axis=0)
    point_shards = np.array_split(point_batches, indices, axis=0)

    h2os = []
    for i, (meas_shard, point_shard) in enumerate(
        zip(meas_shards, point_shards)
    ):
        print(f"Shard: {i} of {len(meas_shards)}")

        ratio = get_ratio_pmap(h2o_grid, point_shard, meas_shard)
        h2o = min1d_pmap(ratio)
        h2os.append(np.concatenate(h2o, axis=0))

    ratio = get_ratio_vmap(h2o_grid, last_point_batch, last_meas_batch)
    h2o = min1d_vmap(ratio)
    h2os.append(h2o)

    h2os = np.concatenate(h2os, axis=0)

    return h2os


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
