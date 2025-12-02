import click
import os

import jax.numpy as jnp
import numpy as np
from spectral import envi

from isofit.radiative_transfer.luts import load
from isofit.core.common import envi_header
from isofit.core.fileio import initialize_output

from isojax.lut import lut_grid, check_bounds
from isojax.inversions import invert_algebraic, heuristic_atmosphere
from isojax.common import construct_point_list, import_jax


# Future GPU handling
gpu = False
n_cores = 1
if not gpu:
    n_cores = import_jax(os.cpu_count() - 1)


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


if __name__ == '__main__':
    cli()
