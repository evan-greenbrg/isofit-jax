import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import tree_util
import numpy as np

from isofit.core.common import eps
from isofit.radiative_transfer.radiative_transfer import RadiativeTransfer
from isofit.radiative_transfer import luts
from isofit.radiative_transfer.engines import Engines


class VectorInterpolator:
    def __init__(
        self,
        grid_input,
        data_input,
    ):
        # Determine if this a singular unique value, if so just return that directly
        val = data_input[(0,) * data_input.ndim]
        # if np.isnan(val) and np.isnan(data_input).all() or np.all(data_input == val):
        #     self.method = -1
        #     self.value = val
        #     return

        self.single_point_data = None

        # Lists and arrays are mutable, so copy first
        grid = grid_input.copy()
        data = data_input.copy()

        # Check if we are using a single grid point. If so, store the grid input.
        if np.prod(list(map(len, grid))) == 1:
            self.single_point_data = data
        self.n = data.shape[-1]

        # None to disable, 0 for unlimited, negatives == 1
        self.cache_size = 0

        self.gridtuples = [np.array(t) for t in grid]
        self.gridarrays = data
        self.binwidth = [
            t[1:] - t[:-1] for t in self.gridtuples
        ]  # binwidth arrays for each dimension
        self.maxbaseinds = np.array([len(t) - 1 for t in self.gridtuples])

    def __call__(self, *args):
        return self._multilinear_grid(
            *args,
            gridtuples=self.gridtuples,
            maxbaseinds=self.maxbaseinds,
            binwidth=self.binwidth,
            gridarrays=jnp.array(self.gridarrays),
            nwl=self.gridarrays.shape[-1]
        )

def multilinear_grid(points, gridtuples, maxbaseinds, binwidth, 
                      gridarrays, nwl=285):

    deltas = jnp.zeros(points.shape, dtype=jnp.int32)
    # (nelements, 2) e.g. (2, 2)
    idxs = jnp.zeros((points.size, 2), dtype=jnp.int32)

    for i, point in enumerate(points):
        delts, slices = lookup(
            i, point, 
            jnp.array(gridtuples[i]), 
            maxbaseinds[i], 
            jnp.array(binwidth[i])
        )

        idxs = idxs.at[i].set(slices)
        deltas = deltas.at[i].set(delts)


    dynamic_slice = jax.jit(
        dynamic_slice,
        static_argnames=('nwl',)
    )
    cube = dynamic_slice(gridarrays, idxs, nwl)

    # Only linear interpolate sliced dimensions
    for i, idx in enumerate(idxs):
        cube = cube.at[0].multiply(1 - deltas.at[i].get())
        cube = cube.at[1].multiply(deltas.at[i].get())
        cube = cube.at[0].add(cube.at[1].get())
        cube = cube.at[0].get()

    return cube

def lookup(i, point, gridtuples, maxbaseinds, binwidth):
    j = jnp.searchsorted(gridtuples[:-1], point) - 1
    delta = jnp.divide(jnp.subtract(point, gridtuples[j]), binwidth[j])

    lower = jnp.max(
        jnp.array([jnp.min(jnp.array([maxbaseinds, j])), 0])
    )
    upper = jnp.max(
        jnp.array([jnp.min(jnp.array([maxbaseinds + 2, j + 2])), 2])
    )

    slices = jnp.array([(lower, upper) for i in gridtuples])
    slices = slices.at[0].set([lower, lower])
    slices = slices.at[-1].set([upper - 1, upper - 1])

    delts = jnp.array([delta for i in gridtuples])
    delts = delts.at[0].set(np.nan)
    delts = delts.at[-1].set(np.nan)

    k = jnp.argmin(jnp.abs(jnp.subtract(point, gridtuples)))

    return delts.at[k].get(), slices.at[k].get()

@staticmethod
def dynamic_slice(gridarrays, idxs, nwl):
    size1 = jnp.subtract(
        idxs.at[(0, 1)].get(),
        idxs.at[(0, 0)].get()
    )

    cube = jax.lax.dynamic_slice(
        gridarrays,
        [idxs.at[(0, 0)].get(), idxs.at[(1, 0)].get(), 0],
        [2, 2, nwl],
    )

    return cube

def multilinear_interp(
jax.scipy.interpolate.RegularGridInterpolator(
    jpoints, jnp.array(transm), method='linear'
)


x = points[:, 0]
y = points[:, 1]
X, Y = np.meshgrid(x, y, indexing='ij')

interp = scipy.interpolate.RegularGridInterpolator((x, y), sphalb)

grid = points
gridtuples = jnp.array(grid)
gridarrays = jnp.array(sphalb)

points = jnp.array([.15, 2.5])
grid = jnp.array(points)

j = jnp.argmin(jnp.linalg.norm(jnp.subtract(grid, point), axis=1))

maxbaseinds = np.array([len(t) - 1 for t in gridtuples])

binwidth = [
    t[1:] - t[:-1] for t in gridtuples
]

lut_names = ['AOT550', 'H2OSTR']
ds = lut.unstack("point")
# Make sure its in expected order, wl at the end
ds = ds.transpose(*lut_names, "wl")
grid = [ds[key].data for key in lut_names]
fgridtuples = [np.array(t) for t in grid]
fbinwidth = [
    t[1:] - t[:-1] for t in fgridtuples
]  # binwidth arrays for each dimension
fmaxbaseinds = np.array([len(t) - 1 for t in fgridtuples])

@jax.jit
def calc_delta(x, grid, bw): 
    j = jnp.argmin(jnp.linalg.norm(jnp.subtract(grid, x), axis=0))
    delta = jnp.divide(jnp.subtract(x, grid[j]), bw[j])

    return delta

point = [.2, 3]
delta = tree_util.tree_map(calc_delta, list(point), gridtuples, binwidth)


jax.make_jaxpr(delta)(point[0], gridtuples[0], binwidth[0])

@jax.jit
def lookup(point, gridtuples, maxbaseinds, binwidth):
    j = jnp.searchsorted(gridtuples[:-1], point) - 1
    delta = jnp.divide(jnp.subtract(point, gridtuples[j]), binwidth[j])

    lower = jnp.max(
        jnp.array([jnp.min(jnp.array([maxbaseinds, j])), 0])
    )
    upper = jnp.max(
        jnp.array([jnp.min(jnp.array([maxbaseinds + 2, j + 2])), 2])
    )

    slices = jnp.array([(lower, upper) for i in gridtuples])
    slices = slices.at[0].set([lower, lower])
    slices = slices.at[-1].set([upper - 1, upper - 1])

    delts = jnp.array([delta for i in gridtuples])
    delts = delts.at[0].set(np.nan)
    delts = delts.at[-1].set(np.nan)

    k = jnp.argmin(jnp.abs(jnp.subtract(point, gridtuples)))

    return delts.at[k].get(), slices.at[k].get()


def multilinear_grid(points, gridtuples, maxbaseinds, binwidth, 
                      gridarrays, nwl=285):

    deltas = jnp.zeros(points.shape, dtype=jnp.int32)
    # (nelements, 2) e.g. (2, 2)
    idxs = jnp.zeros((points.size, 2), dtype=jnp.int32)

    look = tree_util.tree_map(
        lookup, 
        list(point), 
        fgridtuples, 
        list(fmaxbaseinds), 
        fbinwidth
    )
    deltas = deltas.at[:].set([look[0][0], look[1][0]])
    idxs = idxs.at[:].set([look[0][1], look[1][1]])

    dynamic_slice = jax.jit(
        dynamic_slice,
        static_argnames=('nwl',)
    )
    cube = dynamic_slice(gridarrays, idxs, nwl)

    # Only linear interpolate sliced dimensions
    for i, idx in enumerate(idxs):
        cube = cube.at[0].multiply(1 - deltas.at[i].get())
        cube = cube.at[1].multiply(deltas.at[i].get())
        cube = cube.at[0].add(cube.at[1].get())
        cube = cube.at[0].get()

    return cube
