from copy import deepcopy
from functools import partial
import typing

import numpy as np
import jax
import jax.numpy as jnp
import optax

from isofit.core.common import load_wavelen

from isojax.interpolator import interp1d
from isojax.forward import ForwardModel


class Parameter:
    def __init__(self, param: np.ndarray, bounds: np.ndarray[..., 2]):
        # TODO Handle regionds outside of winidx
        # These throw off the scaling

        self.value = param
        self.bounds = bounds

    def min_max_scaler(self):
        self.value = jnp.clip((
            (self.value - self.bounds[0])
            / (self.bounds[1] - self.bounds[0])
        ), 0, 1)

    def min_max_rescaler(self):
        self.value = (
            (self.value * (
                self.bounds[1] 
                - self.bounds[0]
            ) + self.bounds[0])
        )


class State(typing.NamedTuple):
    step: int
    param: jnp.ndarray
    loss: jnp.ndarray
    prev_loss: jnp.ndarray
    grad: jnp.ndarray
    opt_state: optax.transforms._combining.PartitionState
    patience: int
    done: bool


class Invert:
    def __init__(self, fm, optimizer, nsteps=20):
        self.fm = fm
        self.optimizer = optimizer

        self.winidx = retrieve_winidx(fm.full_config)
        self.winidx_mask = jnp.zeros(len(fm.idx_surface))
        self.winidx_mask = self.winidx_mask.at[self.winidx].set(1)

        self.idx_surface = 285

        # Manually tuned. Not that well...
        self.nsteps = nsteps
        self.grad_tol = 1000
        self.loss_tol = 50
        self.max_patience = 50

    def step(self):
        def loss(param, meas, point, xa, Sa_inv_sqrt, seps_inv_sqrt):
            point = point.at[0].set(param['aod'])
            point = point.at[1].set(param['h2o'])
            # svd function needs to be optimized
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
            prior_resid = (
                ((
                    jnp.concatenate([
                        param['surface'],
                        jnp.array([param['aod']]),
                        jnp.array([param['h2o']])
                    ]) - xa
                ).dot(Sa_inv_sqrt))
            ) * jnp.concatenate([self.winidx_mask, jnp.array([1., 1.])])

            # The root-square error isn't working...
            # Just the sum of square for now
            return jnp.sum(
                # jnp.power(jnp.concatenate((meas_resid, prior_resid)), 2),
                jnp.abs(jnp.concatenate((meas_resid, prior_resid))),
            )

        return jax.value_and_grad(loss)

    def run(self, x, meas, point, ci, lamb_norm):
        # TODO The optimization call can be improved
        # I've tried a number of jaxopt calls.
        # I've only gotten scipyboundedminimize to work, but this is
        # limited in how it's called.
        # Could try to get jax.scipy.minimize to work
        # That seemed the most promising combined with a transformation
        # TODO

        seps = self.fm.Seps(x[:self.idx_surface], meas, point)
        seps_inv_sqrt = svd_inv_sqrt(seps)
        xa = self.fm.xa(x[:self.idx_surface], lamb_norm)
        Sa, Sa_inv, Sa_inv_sqrt = self.fm.Sa(
            ci[0],
            lamb_norm[0]
        )

        def cond(state):
            return jnp.logical_and(
                state.step < self.nsteps,
                jnp.logical_not(state.done)
            )

        def fit(state):
            # There's got to be a way to carry fit information in opt_state
            loss, grad = self.step()(
                state.param, 
                meas, point, xa, Sa_inv_sqrt, seps_inv_sqrt
            )
            updates, opt_state = self.optimizer.update(
                grad, state.opt_state
            )
            param = optax.apply_updates(state.param, updates)

            grad_norm = optax.global_norm(grad)
            improved = jnp.greater(
                state.prev_loss - self.loss_tol, 
                loss
            )
            patience = jnp.where(
                improved, 0, state.patience + 1
            )

            new_state = State(
                step=state.step + 1,
                param=param,
                loss=loss,
                prev_loss=state.loss,
                grad=grad,
                opt_state=opt_state,
                patience=patience,
                done=jnp.logical_or(
                    grad_norm < self.grad_tol,
                    patience >= self.max_patience,
                )
            )

            return new_state
        
        param = {
            'surface': x[:self.idx_surface],
            'aod': x[self.idx_surface],
            'h2o': x[self.idx_surface + 1],
        }
        opt_state = self.optimizer.init(param)
        state = State(
            step=0,
            param=param,
            loss=jnp.inf,
            prev_loss=jnp.inf,
            grad=param,
            opt_state=opt_state,
            patience=0,
            done=False,
        )
        # With stopping criteria
        final = jax.lax.while_loop(cond, fit, state)
        
        return (
            final.param['surface'], 
            final.param['aod'],
            final.param['h2o'],
            final.grad['surface'],
            final.grad['aod'],
            final.grad['h2o'],
            final.loss,
            final.step
        )


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

class HeuristicAtmosphere:
    def __init__(self, h2o_grid, jlut, b865=65, b945=76, b1040=88):
        self.h2o_grid = h2o_grid
        self.jlut = jlut

        self.b865=b865
        self.b945=b945
        self.b1040=b1040

        self.get_ratio_vmap = jax.vmap(self.get_ratio, in_axes=(0, None, None))
        self.min1d_vmap = jax.vmap(self.min1d, in_axes=(1))

    def get_ratio(self, h2o, point, meas):

        def ratio_fn(alg):
            #TODO Convert this to the continuum
            return (
                1 - (
                    (alg[:, self.b945] * 2.) 
                    / (alg[:, self.b1040] + alg[:, self.b865])
                )
            )

        point = point.at[1].set(h2o)
        return ratio_fn(invert_algebraic(
            meas,
            self.jlut['rhoatm'](point[None, :]),
            self.jlut['dir_dir'](point[None, :]),
            self.jlut['dir_dif'](point[None, :]),
            self.jlut['dif_dir'](point[None, :]),
            self.jlut['dif_dif'](point[None, :]),
            self.jlut['sphalb'](point[None, :]),
        ))

    def min1d(self, values):
        """Not actually a minimization. Just a zero-crossing"""
        print(values.shape)
        return (
            interp1d(
                values[jnp.argsort(values[:, 0]), 0],
                self.h2o_grid[jnp.argsort(values[:, 0])]
            )(0)
        )


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
    # ((h.T @ p) @ h).T) -> Vectorized and better for jit compilation
    return jnp.einsum(
        'jk->kj',
        jnp.einsum(
            'kj,jl->kl',
            jnp.einsum('jk,lj->kj', h, p),
            h
        )
    )


def symv(alpha, A, x, beta=0, y=0):
    # TODO
    # Re-write using einsum. Need to check diums on A and x
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
    # Call internals directly
    # Read this was faster than using the linalg.cho_solve, which
    # calls these internals
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
    # Equivalent cholesky decomp call. Slower.
    # return jax.scipy.linalg.cho_solve((L, lower), I)
    return X


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


def svd_inv_sqrt(C):
    D, P = jnp.linalg.eigh(C)
    Ds = jnp.diag(1 / jnp.sqrt(D))
    L = jnp.matmul(P, Ds)

    return jnp.matmul(L, jnp.einsum('ij->ji', P))


