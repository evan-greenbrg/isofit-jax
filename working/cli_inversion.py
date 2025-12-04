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

from isojax.forward import ForwardModel
from isojax.lut import lut_grid, check_bounds
from isojax.inversions import invert_algebraic, heuristic_atmosphere, InvertAnalytical
from isojax.common import construct_point_list

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


def invert_oe(
    config_file: str,
    lut_file: str, 
    rdn_file: str,
    obs_file: str,
    loc_file: str,
    x0_file: str,
    out_file: str,
    index_file: str=None,
    batchsize: int=200
):
    config_file='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock_6c/config/emit20220818t205752_isofit.json'
    lut_file='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock_6c/lut_full/lut.nc'
    rdn_file='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_rdn_b0106_v01.img'
    obs_file='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_obs_b0106_v01.img'
    loc_file='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_loc_b0106_v01.img'
    x0_file='/Users/bgreenbe/Github/isofit-jax/working/algebraic_rfl'
    out_file='/Users/bgreenbe/Github/isofit-jax/working/analytical_rfl'
    sub_file='/Users/bgreenbe/Github/isofit-jax/working/algebraic_rfl'
    batchsize=200

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


# SCRATCH

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
