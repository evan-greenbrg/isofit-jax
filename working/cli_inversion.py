import time
import click
import os

import jax.numpy as jnp
from jax import lax
import numpy as np
from spectral import envi
import optax

from isofit.configs import configs
from isofit.radiative_transfer.luts import load
from isofit.core.common import envi_header
from isofit.core.fileio import initialize_output

from isojax.forward import ForwardModel
from isojax.lut import lut_grid, check_bounds
from isojax.inversions import invert_algebraic, HeuristicAtmosphere, InvertAnalytical, Invert
from isojax.common import construct_point_list, batch_and_shard, stack_arrays

# Future GPU handling
gpu = False
if not gpu:
    n_cores = os.cpu_count() - 1
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_cores}'

import jax



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
              fixed_aod=0.1, batchsize=200):

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

    meas_shards, last_meas = batch_and_shard(jnp.array(meas_list), batchsize, n_cores)
    point_shards, last_point = batch_and_shard(point_list, batchsize, n_cores)

    b865 = np.argmin(abs(wl - 865))
    b945 = np.argmin(abs(wl - 945))
    b1040 = np.argmin(abs(wl - 1040))

    h2o_grid = jnp.array(np.unique(lut.coords['H2OSTR'].data))
    hatm = HeuristicAtmosphere(h2o_grid, jlut, b865, b945, b1040)

    get_ratio_vmap = jax.jit(jax.vmap(hatm.get_ratio_vmap, in_axes=(None, 0, 0)))
    get_ratio_pmap = jax.pmap(get_ratio_vmap, in_axes=(None, 0, 0))

    min1d_vmap = jax.jit(jax.vmap(hatm.min1d, in_axes=(0)))
    min1d_pmap = jax.pmap(min1d_vmap, in_axes=(0))

    h2os = []
    total_start = time.time()
    for i, (meas_shard, point_shard) in enumerate(zip(meas_shards, point_shards)):
        start = time.time()
        print(f"Shard: {i} of {len(meas_shards)}")
        ratios = get_ratio_pmap(h2o_grid, point_shard, meas_shard)
        h2o = min1d_pmap(ratios)
        h2o.block_until_ready()

        h2os.append(h2o)
        end = time.time()
        print(f"Finished_batch: {round(end - start, 2)}s")
    total_end = time.time()
    print(f"Finished_all: {round(total_end - total_start, 2)}s")

    ratios = get_ratio_vmap(h2o_grid, last_point, last_meas)
    h2o = min1d_vmap(ratios)

    h2os = stack_arrays(h2os, h2o)
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


@cli.command('oe')
@click.argument('config_file')
@click.argument('lut_file')
@click.argument('rdn_file')
@click.argument('obs_file')
@click.argument('loc_file')
@click.argument('x0_file')
@click.argument('atm_file')
@click.argument('out_file')
@click.option('--batchsize', default=200)
def invert_oe(
    config_file: str,
    lut_file: str, 
    rdn_file: str,
    obs_file: str,
    loc_file: str,
    x0_file: str,
    atm_file: str,
    out_file: str,
    batchsize: int=200
):
    # Load LUT
    lut = load(lut_file)
    lut_names = list(lut.coords)[2:]
    jlut = lut_grid(lut, lut_names)

    # Initialize fm
    config = configs.create_new_config(config_file)
    fm = ForwardModel(config, jlut)

    rdn = envi.open(envi_header(rdn_file))
    rdn_im = rdn.open_memmap(interleave='bip')
    wl = np.array(rdn.metadata['wavelength']).astype(float)

    obs = envi.open(envi_header(obs_file))
    obs_im = obs.open_memmap(interleave='bip')

    loc = envi.open(envi_header(loc_file))
    loc_im = loc.open_memmap(interleave='bip')

    x0 = envi.open(envi_header(x0_file))
    x0_im = x0.open_memmap(interleave='bip')

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

    x0_list = np.reshape(
        x0_im,
        (rdn_im.shape[0] * rdn_im.shape[1], x0_im.shape[2])
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

    x0_norm = jnp.linalg.norm(x0_list[:, fm.surface.idx_ref], axis=1)
    cis = fm.surface.component(x0_list, x0_norm[:, jnp.newaxis])

    # Batching and sharding
    x0_shards, last_x0 = batch_and_shard(x0_list, batchsize, n_cores)
    meas_shards, last_meas = batch_and_shard(jnp.array(meas_list), batchsize, n_cores)
    point_shards, last_point = batch_and_shard(point_list, batchsize, n_cores)
    ci_shards, last_ci = batch_and_shard(cis, batchsize, n_cores)
    lamb_norm_shards, last_lamb_norm = batch_and_shard(x0_norm, batchsize, n_cores)
    atm_shards, last_atm = batch_and_shard(atm_flat, batchsize, n_cores)
    print(f"Running in {len(meas_shards)} shards")

    # TODO put some hcained constraints to help stability in gradients
    # e.g. optax.clip_by_global_norm(1.0), but values have to be properly scaled
    transforms = {
        'surface': optax.adam(learning_rate=3e-3),
        'aod': optax.adam(learning_rate=2e-3),
        'h2o': optax.adam(learning_rate=1e-1),
    }
    optimizer = optax.multi_transform(
        transforms,
        {'surface': 'surface', 'aod': 'aod', 'h2o': 'h2o'}
    )

    iv = Invert(fm, optimizer, nsteps=500)
    inv_vmap = jax.jit(jax.vmap(iv.run, in_axes=(0, 0, 0, 0, 0)))
    inv_pmap = jax.pmap(inv_vmap, in_axes=(0, 0, 0, 0, 0))

    xs = []
    grads = []
    losses = []
    total_start = time.time()
    for i, (
        x0_shard, 
        meas_shard, 
        point_shard,
        ci_shard,
        lamb_norm_shard,
        atm_shard,
    ) in enumerate(
        zip(
            x0_shards,
            meas_shards,
            point_shards,
            ci_shards,
            lamb_norm_shards,
            atm_shards,
        )
    ):
        print(f"Shard: {i} of {len(meas_shards)}")
        start = time.time()
        params = jnp.concatenate([x0_shard, atm_shard], axis=2)
        (
            x_surface, 
            x_aod, 
            x_h2o, 
            grad_surface, 
            grad_aod, 
            grad_h2o, 
            loss, 
            step
        ) = inv_pmap(
            params,
            meas_shard,
            point_shard,
            ci_shard[..., jnp.newaxis],
            lamb_norm_shard[..., jnp.newaxis],
        )
        loss = loss.block_until_ready()

        xs.append(np.concatenate([
            x_surface, 
            x_aod[..., None],
            x_h2o[..., None]
        ], axis=-1))

        grads.append(np.concatenate([
            grad_surface, 
            grad_aod[..., None],
            grad_h2o[..., None]
        ], axis=-1))

        losses.append(loss)
        end = time.time()
        print(f"Finished_batch: {round(end - start, 2)}s")

    print(f"Last shard: {len(meas_shards)} of {len(meas_shards)}")
    last_params = jnp.concatenate([x0_shard, atm_shard], axis=2)
    (
        x_surface, 
        x_aod, 
        x_h2o, 
        grad_surface, 
        grad_aod, 
        grad_h2o, 
        loss, 
    ) = inv_vmap(
        last_x0,
        last_meas,
        last_point,
        last_ci[..., jnp.newaxis],
        last_lamb_norm[..., jnp.newaxis],
        last_atm,
    )
    total_end = time.time()
    print(f"Finished_all: {round(total_end - total_start, 2)}s")

    # Construct outputs
    x_surfaces = stack_arrays(x_surfaces, x_surface)
    x_aods = stack_arrays(x_aods, x_aod)
    x_h2os = stack_arrays(x_h2os, x_h2o)

    grad_surfaces = stack_arrays(grad_surfaces, grad_surface)
    grad_aods = stack_arrays(grad_aods, grad_aod)
    grad_h2os = stack_arrays(grad_h2os, grad_h2o)

    # Stack into state image
    x = np.concatenate([
        x_surfaces,
        x_aods[:, None],
        x_h2os[:, None]
    ], axis=1)

    x_im = np.reshape(x, (rdn.shape[0], rdn.shape[1], x.shape[-1]))
    x_im = np.moveaxis(x_im, -1, 1)

    output = initialize_output(
        {
            "data type": 4,
            "file type": "ENVI Standard",
            "byte order": 0,
        },
        out_file,
        (x_im.shape[0], x_im.shape[1], x_im.shape[2]),
        lines=rdn.metadata["lines"],
        samples=rdn.metadata["samples"],
        interleave="bil",
        bands=f"{x_im.shape[1]}",
        band_names=[rdn.metadata['band names']] + ['AOD', 'H2O'],
        description=("OE inversion"),
    )
    out = envi.open(envi_header(output)).open_memmap(
        interleave="bil", writable=True
    )
    out[...] = x_im
    del out


@cli.command('analytical')
@click.argument('config_file')
@click.argument('lut_file')
@click.argument('rdn_file')
@click.argument('obs_file')
@click.argument('loc_file')
@click.argument('x0_file')
@click.argument('atm_file')
@click.argument('out_file')
@click.argument('sub_file')
@click.option('--index_file')
@click.option('--batchsize', default=200)
def analytical_line(
    config_file: str,
    lut_file: str, 
    rdn_file: str,
    obs_file: str,
    loc_file: str,
    x0_file: str,
    atm_file: str,
    out_file: str,
    sub_file: str,
    index_file: str=None,
    batchsize: int=200
):

    # Load LUT
    lut = load(lut_file)
    lut_names = list(lut.coords)[2:]
    jlut = lut_grid(lut, lut_names)

    # Initialize fm
    config = configs.create_new_config(config_file)
    fm = ForwardModel(config, jlut)

    rdn = envi.open(envi_header(rdn_file))
    rdn_im = rdn.open_memmap(interleave='bip')
    wl = np.array(rdn.metadata['wavelength']).astype(float)

    obs = envi.open(envi_header(obs_file))
    obs_im = obs.open_memmap(interleave='bip')

    loc = envi.open(envi_header(loc_file))
    loc_im = loc.open_memmap(interleave='bip')

    x0 = envi.open(envi_header(x0_file))
    x0_im = x0.open_memmap(interleave='bip')

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

    x0_list = np.reshape(
        x0_im,
        (rdn_im.shape[0] * rdn_im.shape[1], x0_im.shape[2])
    )

    atm_flat = np.reshape(
        atm_im,
        (rdn_im.shape[0] * rdn_im.shape[1], atm_im.shape[2])
    )

    # Load sub file
    if index_file:
        # Expects subs file to be shorter than per-pixel
        raise NotImplementedError('Need to add support for indexed subs file')

    else:
        # Expects subs file to be same dim as per-pixel
        sub = envi.open(envi_header(sub_file))
        sub_im = sub.open_memmap(interleave='bip')

        sub_list = np.reshape(
            sub_im,
            (rdn_im.shape[0] * rdn_im.shape[1], sub_im.shape[2])
        )

    # Construct "Geom"
    point_list = construct_point_list(
        lut,
        obs_flat, loc_flat,
        aod=atm_flat[:, 0],
        h2o=atm_flat[:, 1]
    )

    x0_norm = jnp.linalg.norm(x0_list[:, fm.surface.idx_ref], axis=1)
    cis = fm.surface.component(x0_list, x0_norm[:, jnp.newaxis])

    # Get RT up front
    start_time = time.time()
    L_atm = jlut['rhoatm'](point_list)
    L_dir_dir = jlut['dir_dir'](point_list)
    L_dir_dif = jlut['dir_dif'](point_list)
    L_dif_dir = jlut['dif_dir'](point_list)
    L_dif_dif = jlut['dif_dif'](point_list)
    s_alb = jlut['sphalb'](point_list)
    L_tot = (L_dir_dir + L_dir_dif + L_dif_dir + L_dif_dir)
    L_tot = L_tot.block_until_ready()
    end_time = time.time()
    print(f"Finished RT call: {end_time - start_time}")

    # Batching
    indices = np.arange(batchsize, len(meas_list), batchsize)
    x0_batches = np.array_split(
        x0_list, indices, axis=0
    )
    meas_batches = np.array_split(
        meas_list, indices, axis=0
    )
    sub_batches = np.array_split(
        sub_list, indices, axis=0
    )
    point_batches = np.array_split(
        point_list, indices, axis=0
    )
    ci_batches = np.array_split(
        cis, indices, axis=0
    )
    lamb_norm_batches = np.array_split(
        x0_norm[:, jnp.newaxis], indices, axis=0
    )
    L_atm_batches = np.array_split(
        L_atm, indices, axis=0
    )
    L_tot_batches = np.array_split(
        L_tot, indices, axis=0
    )
    s_alb_batches = np.array_split(
        s_alb, indices, axis=0
    )

    # Need to specifically handle uneven blocks
    last_x0 = x0_batches[-1]
    last_meas = meas_batches[-1]
    last_sub = sub_batches[-1]
    last_point = point_batches[-1]
    last_ci = ci_batches[-1]
    last_lamb_norm = lamb_norm_batches[-1]
    last_L_atm = L_atm_batches[-1]
    last_L_tot = L_tot_batches[-1]
    last_s_alb = s_alb_batches[-1]

    x0_batches = jnp.array(x0_batches[:-1])
    meas_batches = jnp.array(meas_batches[:-1])
    sub_batches = jnp.array(sub_batches[:-1])
    point_batches = jnp.array(point_batches[:-1])
    ci_batches = jnp.array(ci_batches[:-1])
    lamb_norm_batches = jnp.array(lamb_norm_batches[:-1])
    L_atm_batches = jnp.array(L_atm_batches[:-1])
    L_tot_batches = jnp.array(L_tot_batches[:-1])
    s_alb_batches = jnp.array(s_alb_batches[:-1])

    # Sharding
    indices = np.arange(n_cores, len(x0_batches), n_cores)
    x0_shards = np.array_split(x0_batches, indices, axis=0)
    meas_shards = np.array_split(meas_batches, indices, axis=0)
    sub_shards = np.array_split(sub_batches, indices, axis=0)
    point_shards = np.array_split(point_batches, indices, axis=0)
    ci_shards = np.array_split(ci_batches, indices, axis=0)
    lamb_norm_shards = np.array_split(
        lamb_norm_batches, indices, axis=0
    )
    L_atm_shards = np.array_split(L_atm_batches, indices, axis=0)
    L_tot_shards = np.array_split(L_tot_batches, indices, axis=0)
    s_alb_shards = np.array_split(s_alb_batches, indices, axis=0)

    # JIT and pmap inversion
    invert_analytical = InvertAnalytical(fm)
    inv_vmap = jax.jit(jax.vmap(
        invert_analytical.invert,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0)
    ))
    inv_pmap = jax.pmap(
        inv_vmap,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0)
    )

    # Loop through shards
    xks = []
    total_start = time.time()
    for i, (
        x0_shard, 
        meas_shard, 
        sub_shard, 
        point_shard,
        ci_shard,
        lamb_norm_shard,
        L_atm_shard,
        L_tot_shard,
        s_alb_shard,
    ) in enumerate(
        zip(
            x0_shards,
            meas_shards,
            sub_shards,
            point_shards,
            ci_shards,
            lamb_norm_shards,
            L_atm_shards,
            L_tot_shards,
            s_alb_shards,
        )
    ):

        # Only explicitely place shard on device if len(shard) == jax.devices()
        if len(x0_shard) == jax.devices():
            x0_shard = jax.device_put_sharded(list(x0_shard), jax.devices())
            meas_shard = jax.device_put_sharded(list(meas_shard), jax.devices())
            sub_shard = jax.device_put_sharded(list(sub_shard), jax.devices())
            point_shard = jax.device_put_sharded(list(point_shard), jax.devices())
            ci_shard = jax.device_put_sharded(list(ci_shard), jax.devices())
            lamb_norm_shard = jax.device_put_sharded(list(lamb_norm_shard), jax.devices())
            L_atm_shard = jax.device_put_sharded(list(L_atm_shard), jax.devices())
            L_tot_shard = jax.device_put_sharded(list(L_tot_shard), jax.devices())
            s_alb_shard = jax.device_put_sharded(list(s_alb_shard), jax.devices())

        print(f"Shard: {i} of {len(meas_shards)}")

        start = time.time()
        xk = inv_pmap(
            x0_shard,
            meas_shard,
            point_shard,
            sub_shard,
            s_alb_shard,
            L_tot_shard,
            L_atm_shard,
            ci_shard,
            lamb_norm_shard
        )
        xk = xk.block_until_ready()
        xks.append(xk)
        end = time.time()
        print(f"Finished_batch: {round(end - start, 2)}s")

    total_end = time.time()
    print(f"Finished_all: {round(total_end - total_start, 2)}s")

    # Flatten out shard and batch dims
    xks = np.concatenate(xks, axis=0)
    xks = np.concatenate(xks, axis=0)

    # Solve the last partial batch
    xk = inv_vmap(
        last_x0,
        last_meas,
        last_point,
        last_sub,
        last_s_alb,
        last_L_tot,
        last_L_atm,
        last_ci,
        last_lamb_norm 
    )
    xks = np.concatenate((xks, xk), axis=0)

    # Save
    x = np.reshape(xks, (rdn.shape[0], rdn.shape[1], xks.shape[1]))
    x = np.moveaxis(x, -1, 1)
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
        bands=f"{x.shape[1]}",
        band_names=rdn.metadata['band names'],
        wavelength=rdn.metadata['wavelength'],
        description=("Analytical inversion"),
    )
    out = envi.open(envi_header(output)).open_memmap(
        interleave="bil", writable=True
    )
    out[...] = x
    del out


if __name__ == '__main__':
    cli()



