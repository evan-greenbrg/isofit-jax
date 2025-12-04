from copy import deepcopy
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from isofit.core.common import load_wavelen

from isojax.interpolator import interp1d
from isojax.forward import ForwardModel


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


def p_tilde(h, p):
    # ((h.T @ p) @ h).T)
    return jnp.einsum(
        'jk->kj',
        jnp.einsum(
            'kj,jl->kl',
            jnp.einsum('jk,lj->kj', h, p),
            h
        )
    )


def symv(alpha, A, x, beta=0, y=0):
    return alpha * (A @ x) + beta * y


@jax.vmap
def vsymv(c_rcond, p, h, meas_i, l_atm, prprod_i):

    return symv(
        1,
        c_rcond,
        h.T @ symv(1, p, meas_i[winidx] - l_atm[winidx]) + prprod_i
    )

@partial(jax.jit, static_argnums=1)
def dpotri_jax(L, lower=True):
    n = L.shape[0]
    I = jnp.eye(n, dtype=L.dtype)
    # cho_solve solves A X = B using the Cholesky factor.
    Y = jax.lax.linalg.triangular_solve(
        a=L,
        b=I,
        left_side=False,
        transpose_a=False,
        conjugate_a=False,
        unit_diagonal=False,
        lower=lower,
    )
    X = jax.lax.linalg.triangular_solve(
        a=L,
        b=Y,
        left_side=False,
        transpose_a=True,
        conjugate_a=False,
        unit_diagonal=False,
        lower=lower,
    )
    return X
    # return jax.scipy.linalg.cho_solve((L, lower), I)


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


class InvertAnalytical:
    def __init__(self, fm):
        self.fm = fm

    def invert(self, x, meas, point, sub, s_alb,
                          L_tot, L_atm, ci, lamb_norm):

        seps = self.fm.Seps(x, meas, point)

        H = self.fm.surface.analytical_model(
            jnp.multiply(sub, s_alb),
            L_tot
        )

        Sa, Sa_inv, Sa_inv_sqrt = self.fm.surface.Sa(
            ci,
            lamb_norm
        )
        xa = self.fm.surface.xa(x, lamb_norm)
        prprod = jnp.matmul(Sa_inv, xa)

        C = jax.lax.linalg.cholesky(seps)
        L = dpotri_jax(C)

        P_tilde = p_tilde(H, L)
        P_rcond = jnp.add(Sa_inv, P_tilde)

        LI_rcond = jax.lax.linalg.cholesky(P_rcond)
        C_rcond = dpotri_jax(LI_rcond)

        def vsymv(c_rcond, p, h, meas_i, l_atm, prprod_i):
            return symv(
                1,
                c_rcond,
                h.T @ symv(1, p, meas_i - l_atm) + prprod_i
            )

        xk = vsymv(C_rcond, L, H, meas, L_atm, prprod)

        return xk


def svd_inv_sqrt(C):
    D, P = jnp.linalg.eigh(C)
    Ds = jnp.diag(1 / jnp.sqrt(D))
    L = jnp.matmul(P, Ds)

    return jnp.matmul(L, jnp.einsum('ij->ji', P))


class Invert:
    def __init__(self, fm):
        self.fm = fm

        self.winidx = retrieve_winidx(fm.full_config)
        self.winidx_mask = jnp.zeros(len(fm.idx_surface))
        self.winidx_mask = self.winidx_mask.at[self.winidx].set(1)

        self.idx_surface = 285

    def step(self):
        def loss(param, meas, point, xa, Sa):
            point = point.at[0].set(param['aod'])
            point = point.at[1].set(param['h2o'])
            # svd function needs to be optimized
            seps_inv_sqrt = svd_inv_sqrt(fm.Seps(
                param['surface'], 
                meas,
                point
            ))

            # This is going to slow me down
            pred = self.fm.RT.calc_rdn(
                param['surface'],
                param['surface'],
                self.fm.RT.jlut['rhoatm'](point[None, :])[0],
                self.fm.RT.jlut['sphalb'](point[None, :])[0],
                self.fm.RT.jlut['dir_dir'](point[None, :])[0],
                self.fm.RT.jlut['dif_dir'](point[None, :])[0],
                self.fm.RT.jlut['dir_dif'](point[None, :])[0],
                self.fm.RT.jlut['dif_dif'](point[None, :])[0],
            )
            # Measurement portion scaled by instrument noise
            meas_resid = ((pred - meas).dot(seps_inv_sqrt))

            # Prior portion scaled by prior covariance
            # TODO add in the atm prior means and covaraince
            prior_resid = (
                ((
                    jnp.concatenate([
                        param['surface'],
                        jnp.array([param['aod']]),
                        jnp.array([param['h2o']])
                    ]) - xa
                ).dot(Sa_inv_sqrt))
            ) * jnp.concatenate([self.winidx_mask, jnp.array([1., 1.])])

            # The square error isn't working...
            return jnp.sum(
                jnp.power(jnp.concatenate((meas_resid, prior_resid)), 2),
            )

        return jax.value_and_grad(loss)


iv = Invert(fm)

def fit(param, meas, point, xa, Sa, opt_state):
    value, grad = iv.step()(param, meas, point, xa, Sa)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(param, updates)

    return params, opt_state, grad

fit_vmap = jax.jit(jax.vmap(fit))
learning_rate = 3e-3
# optimizer = optax.adam(learning_rate)
nsteps=100

cmap_name = 'winter'
cmap = plt.get_cmap(cmap_name)
colors = [cmap(i) for i in np.linspace(0, 1, nsteps)]

for i in range(2):
# Set up the inversion for a single pixel
    x = x0[i, ...]
    meas = meas_list[i, ...]
    point = point_list[i, ...]
    ci = cis[i]
    lamb_norm = x0_norm[i]
    atm = atm_flat[i, ...]

    seps = fm.Seps(x, meas, point)
    seps_inv_sqrt = svd_inv_sqrt(seps)
    xa = fm.xa(x, lamb_norm)
    Sa, Sa_inv, Sa_inv_sqrt = fm.Sa(
        ci,
        lamb_norm
    )


    param = {
        'surface': x,
        'aod': atm[0],
        'h2o': atm[1],
    }
    transforms = {
        'surface': optax.adam(learning_rate=3e-3),
        'aod': optax.adam(learning_rate=3e-3),
        'h2o': optax.adam(learning_rate=1e-1),
    }
    optimizer = optax.multi_transform(
        transforms,
        {'surface': 'surface', 'aod': 'aod', 'h2o': 'h2o'}
    )

    opt_state = optimizer.init(param)

    params = []
    grads = []
    for i in range(nsteps):
        print(i)
        param, opt_state, grad = fit(param, meas, point, xa, Sa, opt_state)
        params.append(param)
        grads.append(grad)

    # TODO The gradients are bad. The optimization is wrong.
    for param, grad in zip(params, grads):
        break
    params = np.array(params)
    grads = np.array(grads)

    fig, axs = plt.subplots(2, 1, sharex=True)
    for i in range(nsteps):
        axs[0].plot(wl, params[i,:285], color=colors[i])
        axs[1].plot(wl, grads[i,:285], color=colors[i])
    # plt.plot(wl, params[0, :], color='black', ls='--', lw=2)
    # plt.plot(wl, params[-1, :], color='black', lw=2)
    axs[0].set_ylim([-.05, .5])
    # axs[1].set_ylim([-.05, .5])
    plt.show()

    fig, axs = plt.subplots(2, 2, sharex=True)
    axs[0, 0].set_ylabel('AOD')
    axs[0, 0].scatter(
        [i for i in range(nsteps)],
        params[:, -2]
    )
    axs[1, 0].set_ylabel('AOD Grad')
    axs[1, 0].scatter(
        [i for i in range(nsteps)],
        grads[:, -2]
    )
    axs[0, 1].set_ylabel('H2O')
    axs[0, 1].scatter(
        [i for i in range(nsteps)],
        params[:, -1]
    )
    axs[1, 1].set_ylabel('H2O Grad')
    axs[1, 1].scatter(
        [i for i in range(nsteps)],
        grads[:, -1]
    )
    plt.show()
    break

# Set up params
opt_state = optimizer.init(params)

# Step function should pull seps, pred, xa, Sa all from params
