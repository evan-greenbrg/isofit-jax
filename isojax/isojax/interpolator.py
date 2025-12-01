import jax
import jax.numpy as jnp
from typing import Sequence, Callable, Tuple, Optional

Array = jnp.ndarray


def prepare_grid(grid):
    """
    Ensure each grid axis is a 1‑D, strictly increasing JAX array,
    and also pre‑compute the reciprocal of the spacing (for speed).

    Assumes:
        Each grid axis is 1-d
        Grid axes are strictly increasing

    I took out these checks because the boolean operator can't be done on traced object
    """
    axes = []
    inv_spacings = []
    for g in grid:
        g = jnp.asarray(g)
        axes.append(g)
        spacing = g[1:] - g[:-1]
        inv_spacings.append(1.0 / spacing)
    return axes, inv_spacings


def search_lower_indices(x, axis, inv_spacing):
    """
    For a *single* point coordinate `x` (scalar) and a given axis,
    return the lower index `i` and the fractional weight `t` in the cell.
    The index is clipped to the valid range [0, len(axis)-2].
    """
    # `searchsorted` returns index of the *right* side, so we subtract 1.
    i = jnp.searchsorted(axis, x, side="right") - 1
    i = jnp.clip(i, 0, axis.shape[0] - 2)  # clamp to valid range

    # Fractional distance within the cell
    t = (x - axis[i]) * inv_spacing[i]
    t = jnp.clip(t, 0.0, 1.0)
    return i, t


def interp_point(xi, axes, inv_spacings, values, wl, init_value=0.):
    """
    Linear interpolation for *one* query point `xi` (shape (ndim,)).
    Will be vmapped over array.
    """
    ndim = len(axes)

    # Cache the lower indexes and their weights
    lower_idx = []
    t = []
    for dim in range(ndim):
        i, w = search_lower_indices(xi[dim], axes[dim], inv_spacings[dim])
        lower_idx.append(i)
        t.append(w)
    lower_idx = jnp.stack(lower_idx)
    t = jnp.stack(t)

    def body(corner, acc):
        """
        corner: 
            (int) The dim-bit advancing value.
        acc:
            (array) The running sum of weighted value
        """
        # bit-shift to find low or high binary position for the 2**n corners
        mask = (corner >> jnp.arange(ndim)) & 1

        # Use cached lower idx to turn binary position into grid index
        idx = lower_idx + mask

        # (t if mask == 1 else (1‑t))
        w = jnp.prod(jnp.where(mask == 1, t, 1.0 - t))

        # Gather the value at that corner
        val = values[tuple(idx)]
        return acc + w * val

    # Loop over each 2**n corner: bit-shift captures lower (0) or upper (1) positions
    result = jax.lax.fori_loop(
        0, 1 << ndim, 
        body, 
        jnp.array([init_value for i in wl])
    )
    return result


def multilinear_interpolator(
    grid_axes,
    values,
    wl,
    extrapolate=False,
    fill_value=jnp.nan
):
    """
    JIT‑compiled multi-linear regular‑grid interpolator.

    Edited from GPT-OSS (High reasoning).

    Parameters
    ----------
    grid_axes:
        List of 1‑D arrays specifying the grid points along each axis,
    values:
        N‑dimensional array. len(values at index, i) == len(grid[i]).
    extrapolate :
        If ``False`` (default) points outside the convex hull return ``nan``.
        If ``True`` the interpolation is performed using the nearest visible cell
        (i.e. clipping to the domain).
    Returns
    -------
    interpolator : Function 
    """
    axes, inv_spacings = prepare_grid(grid_axes)

    @jax.jit
    @jax.vmap
    def batched_interp(xi):
        if not extrapolate:
            out_of_bounds = jnp.any(
                (xi < jnp.array([a[0] for a in axes])) |
                (xi > jnp.array([a[-1] for a in axes]))
            )
            # Return fill for the entire point if it's outside.
            return jnp.where(
                out_of_bounds,
                fill_value,
                interp_point(xi, axes, inv_spacings, values, wl)
            )
        else:
            # With extrapolation we simply clamp the coordinate to the domain,
            # which is already done inside `search_lower_indices` via clipping.
            # TODO check extrapolation behavior
            return interp_point(xi, axes, inv_spacings, values, wl)

    return batched_interp


def batch_interp1d(grid):
    axes, inv_spacings = prepare_grid(grid[jnp.newaxis, :])

    @jax.jit
    @jax.vmap
    def batched_interp(xi, values):
        ndim = len(axes)
        i, w = search_lower_indices(xi, axes[0], inv_spacings[0])
        return jnp.sum(jnp.array([
            (1 - w) * values[i],
            w * values[i + 1],
        ]))

    return batched_interp


def interp1d(grid, values):
    axes, inv_spacings = prepare_grid(grid[jnp.newaxis, :])

    def interp(xi):
        ndim = len(axes)
        i, w = search_lower_indices(xi, axes[0], inv_spacings[0])
        return jnp.sum(jnp.array([
            (1 - w) * values[i],
            w * values[i + 1],
        ]))

    return interp
