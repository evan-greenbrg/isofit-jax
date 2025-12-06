import time
import click
import os

import jax.numpy as jnp
from jax import lax
import numpy as np
from spectral import envi

from isofit.configs import configs
from isofit.radiative_transfer.luts import load
from isofit.core.common import envi_header
from isofit.core.fileio import initialize_output

from isojax.lut import lut_grid, check_bounds
from isojax.common import construct_point_list

# Future GPU handling
gpu = False
if not gpu:
    n_cores = os.cpu_count() - 1
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_cores}'

import jax


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
        ### TODO
        # Replace with continuum removal
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



@click.group()
def cli():
    pass


@cli.command('heuristic')
@click.argument("lut_path")
@click.argument("rdn_file")
@click.argument("obs_file")
@click.argument("loc_file")
@click.argument("out_file")
@click.option("--fixed_aod", default=0.1)
@click.option("--batchsize", default=1000)
def heuristic(lut_path, rdn_file, obs_file, loc_file, out_file,
         fixed_aod=0.1, batchsize=1000):

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
    point_list = construct_point_list(
        lut,
        obs_flat, loc_flat,
        aod=0.1, h2o=2.0
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
        b1040=b1040,
        nshard=n_cores
    )

    h2os = np.reshape(h2os, (rdn.shape[0], rdn.shape[1]))
    aods = np.ones(h2os.shape) * fixed_aod
    atm = np.concatenate([
        aods[np.newaxis, ...],
        h2os[np.newaxis, ...],
    ], axis=0)
    atm = np.moveaxis(atm, 0, 1)

    # Save file
    output = initialize_output(
        {
            "data type": 4,
            "file type": "ENVI Standard",
            "byte order": 0,
        },
        out_file,
        (rdn.shape[0], 2, rdn.shape[1]),
        lines=rdn.metadata["lines"],
        samples=rdn.metadata["samples"],
        interleave="bil",
        bands="2",
        band_names=["AOD", "H2O"],
        description=("Heuristic Atmosphere"),
    )
    out = envi.open(envi_header(output)).open_memmap(
        interleave="bil", writable=True
    )
    out[...] = atm
    del out


@cli.command('algebraic')
@click.argument("lut_path")
@click.argument("rdn_file")
@click.argument("obs_file")
@click.argument("loc_file")
@click.argument("atm_file")
@click.argument("out_file")
def algebraic(lut_path, rdn_file, obs_file, loc_file, atm_file, out_file):
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

    atm = envi.open(envi_header(atm_file))
    atm_im = atm.open_memmap(interleave='bip')

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

    atm_flat = np.reshape(
        atm_im,
        (rdn_im.shape[0] * rdn_im.shape[1], atm_im.shape[2])
    )

    # Construct "Geom"
    point_list = construct_point_list(
        lut,
        obs_flat, loc_flat,
        aod=atm_flat[:, 0],
        h2o=atm_flat[:, 1]
    )

    x0 = invert_algebraic(
        meas_list,
        jlut['rhoatm'](point_list),
        jlut['dir_dir'](point_list),
        jlut['dir_dif'](point_list),
        jlut['dif_dir'](point_list),
        jlut['dif_dif'](point_list),
        jlut['sphalb'](point_list),
    )
    x0 = np.reshape(x0, (rdn.shape[0], rdn.shape[1], x0.shape[1]))
    x0 = np.moveaxis(x0, -1, 1)

    output = initialize_output(
        {
            "data type": 4,
            "file type": "ENVI Standard",
            "byte order": 0,
        },
        out_file,
        (rdn.shape[0], x0.shape[1], rdn.shape[1]),
        lines=rdn.metadata["lines"],
        samples=rdn.metadata["samples"],
        interleave="bil",
        bands=f"{x0.shape[1]}",
        band_names=rdn.metadata['band names'],
        description=("Algebraic inversion"),
    )
    out = envi.open(envi_header(output)).open_memmap(
        interleave="bil", writable=True
    )
    out[...] = x0
    del out
