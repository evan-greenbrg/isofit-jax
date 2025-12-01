import numpy as np
import jax
import jax.numpy as jnp

from isojax.interpolator import multilinear_interpolator


def lut_grid(lut, lut_names):
    wl = lut.wl.data
    points = np.array([point for point in lut.point.data])

    # Initialize arrays
    lut = lut.unstack("point")
    lut = lut.transpose(*lut_names, "wl")
    rhoatm = jnp.array(lut['rhoatm'].load().data)
    sphalb = jnp.array(lut['sphalb'].load().data)
    transm_down_dir = jnp.array(lut['transm_down_dir'].load().data)
    transm_down_dif = jnp.array(lut['transm_down_dif'].load().data)
    transm_up_dir = jnp.array(lut['transm_up_dir'].load().data)
    transm_up_dif = jnp.array(lut['transm_up_dif'].load().data)
    dir_dir = jnp.array(lut['dir-dir'].load().data)
    dir_dif = jnp.array(lut['dir-dif'].load().data)
    dif_dir = jnp.array(lut['dif-dir'].load().data)
    dif_dif = jnp.array(lut['dif-dif'].load().data)

    jpoints = tuple((np.unique(points[:, i]) for i in range(points.shape[1])))
    int_rhoatm = multilinear_interpolator(
        jpoints, rhoatm, wl, extrapolate=True
    )
    int_sphalb = multilinear_interpolator(
        jpoints, sphalb, wl, extrapolate=True
    )
    int_transm_down_dir = multilinear_interpolator(
        jpoints, transm_down_dir, wl, extrapolate=True
    )
    int_transm_down_dif = multilinear_interpolator(
        jpoints, transm_down_dif, wl, extrapolate=True
    )
    int_transm_up_dir = multilinear_interpolator(
        jpoints, transm_up_dir, wl, extrapolate=True
    )
    int_transm_up_dif = multilinear_interpolator(
        jpoints, transm_up_dif, wl, extrapolate=True
    )
    int_dir_dir = multilinear_interpolator(
        jpoints, dir_dir, wl, extrapolate=True
    )
    int_dir_dif = multilinear_interpolator(
        jpoints, dir_dif, wl, extrapolate=True
    )
    int_dif_dir = multilinear_interpolator(
        jpoints, dif_dir, wl, extrapolate=True
    )
    int_dif_dif = multilinear_interpolator(
        jpoints, dif_dif, wl, extrapolate=True
    )

    return {
        'rhoatm': int_rhoatm,
        'sphalb': int_sphalb, 
        'transm_down_dir': int_transm_down_dir,
        'transm_down_dif': int_transm_down_dif,
        'transm_up_dir': int_transm_up_dir,
        'transm_up_dif': int_transm_up_dif,
        'dir_dir': int_dir_dir,
        'dir_dif': int_dir_dif,
        'dif_dir': int_dif_dir,
        'dif_dif': int_dif_dif,
    }


@jax.jit
def check_bounds(vals, minb, maxb):
    return (
        jnp.minimum(
            jnp.maximum(vals, minb),
            maxb,
        )
    )
