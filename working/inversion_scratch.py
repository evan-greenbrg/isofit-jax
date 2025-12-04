from isofit.configs import configs

from isojax.forward import ForwardModel
from isojax.surface import MultiComponentSurface


config_path = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock_6c/config/emit20220818t205752_isofit.json'
config = configs.create_new_config(config_path)
fm = ForwardModel(config)
winidx = retrieve_winidx(config)

lut_path = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock_6c/lut_full/lut.nc'
lut = load(lut_path)
wl = lut.wl.data
points = np.array([point for point in lut.point.data])
lut_names = list(lut.coords)[2:]
jlut = lut_grid(lut, lut_names)
rdn_file = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_rdn_b0106_v01.img'
obs_file = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_obs_b0106_v01.img'
loc_file = '/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_loc_b0106_v01.img'

norm = lambda x, vmin, vmax: (x - vmin) / (vmax - vmin)
rdn = envi.open(envi_header(rdn_file))
rdn_im = rdn.open_memmap(interleave='bip')

wl = np.array(rdn.metadata['wavelength']).astype(float)
bands = [
    np.argmin(np.abs(650 - wl)),
    np.argmin(np.abs(540- wl)),
    np.argmin(np.abs(450- wl)),
]

obs = envi.open(envi_header(obs_file))
obs_im = obs.open_memmap(interleave='bip')

loc = envi.open(envi_header(loc_file))
loc_im = loc.open_memmap(interleave='bip')

# Test batching
meas_list = np.reshape(rdn_im, (rdn_im.shape[0] * rdn_im.shape[1], rdn_im.shape[2]))

# Construct geoms
obs_flat = np.reshape(obs_im, (rdn_im.shape[0] * rdn_im.shape[1], obs_im.shape[2]))
loc_flat = np.reshape(loc_im, (rdn_im.shape[0] * rdn_im.shape[1], loc_im.shape[2]))

test_aod = 0.1
test_h2o = 2.0
point_list = np.zeros((len(obs_flat), 6))
point_list[:, 0] = test_aod
point_list[:, 1] = test_h2o
point_list[:, 2] = obs_flat[:, 2]

delta_phi = np.abs(obs_flat[:, 3] - obs_flat[:, 1])
point_list[:, 3] = np.minimum(delta_phi, 360 - delta_phi)
point_list[:, 4] = obs_flat[:, 4]
point_list[:, 5] = loc_flat[:, 2]

bounds = np.array([
    [np.min(lut.coords['AOT550'].values), np.max(lut.coords['AOT550'].values)],
    [np.min(lut.coords['H2OSTR'].values), np.max(lut.coords['H2OSTR'].values)],
    [np.min(lut.coords['observer_zenith'].values), np.max(lut.coords['observer_zenith'].values)],
    [np.min(lut.coords['relative_azimuth'].values), np.max(lut.coords['relative_azimuth'].values)],
    [np.min(lut.coords['solar_zenith'].values), np.max(lut.coords['solar_zenith'].values)],
    [np.min(lut.coords['surface_elevation_km'].values), np.max(lut.coords['surface_elevation_km'].values)],
])

point_list = check_bounds(
    point_list,
    bounds[:, 0],
    bounds[:, 1]
)

b865 = np.argmin(abs(wl - 865))
b945 = np.argmin(abs(wl - 945))
b1040 = np.argmin(abs(wl - 1040))

fix_aod = 0.1
h2os = heuristic_atmosphere(
    jnp.array(np.unique(lut.coords['H2OSTR'].data)),
    meas_list, 
    point_list, 
    jlut, 
    fix_aod=fix_aod, 
    batchsize=1000, 
    b865=b865,
    b945=b945,
    b1040=b1040,
    nshard=14
)

# h2os = np.reshape(h2os, (rdn.shape[0] * rdn.shape[1]))

point_list = point_list.at[:, 0].set(fix_aod)
point_list = point_list.at[:, 1].set(h2os)
# invert_algebraic = jax.vmap(invert_algebraic)
x0 = invert_algebraic(
    meas_list,
    jlut['rhoatm'](point_list),
    jlut['dir_dir'](point_list),
    jlut['dir_dif'](point_list),
    jlut['dif_dir'](point_list),
    jlut['dif_dif'](point_list),
    jlut['sphalb'](point_list),
)

surface = MultiComponentSurface(config)
x0_norm = jnp.linalg.norm(x0[:, surface.idx_ref], axis=1)
xa = surface.xa(x0, x0_norm[:, jnp.newaxis])
cis = surface.component(x0, x0_norm[:, jnp.newaxis])
sub = x0

h2os_np = np.array(h2os)
aods = np.ones(h2os_np.shape) * 0.1
atms = jnp.array([aods, h2os]).T


@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(self, fm, jlut, winidx, 
                 loglevel='INFO', logfile=None):
        logging.basicConfig(
            format="%(levelname)s:%(asctime)s ||| %(message)s",
            level=loglevel,
            filename=logfile,
            datefmt="%Y-%m-%d,%H:%M:%S",
        )

        self.fm = fm

        # TODO Look into ways to make this pre-compiled functions
        # shared across workers
        self.inv = jax.jit(invert_analytical, static_argnums=0)
        self.jlut = jlut
        self.winidx = winidx

    def run(self, x0_d, meas_d, sub_d, point_d, ci_d, lamb_norm_d):
        seps_d = self.fm.Seps(
            x0_d, meas_d, point_d, self.jlut
        )[:, self.winidx, :][..., self.winidx]

        L_atm_d = self.jlut['rhoatm'](point_d)
        L_dir_dir_d = self.jlut['dir_dir'](point_d)
        L_dir_dif_d = self.jlut['dir_dif'](point_d)
        L_dif_dir_d = self.jlut['dif_dir'](point_d)
        L_dif_dif_d = self.jlut['dif_dif'](point_d)
        s_alb_d = self.jlut['sphalb'](point_d)

        logging.info("Running")
        res = self.inv(
            self.fm,
            x0_d,
            meas_d,
            sub_d,
            point_d,
            ci_d,
            lamb_norm_d, 
            seps_d,
            L_atm_d,
            (L_dir_dir_d + L_dir_dif_d + L_dif_dir_d + L_dif_dif_d),
            s_alb_d,
            self.winidx
        )
        res.block_until_ready()
        logging.info("Finished")
        return res


def analytical_line_multi(x0, meas_list, sub, point_list, cis, x0_norm, 
    batchsize=1000):

    # TODO This is still taking too long.
    # Not sure how to debug it...

    # Set atms
    point_list = point_list.at[:, :2].set(atms)

    # Set up batching
    indices = np.arange(batchsize, len(meas_list), batchsize)
    x0_batches = np.array_split(x0, indices, axis=0)
    meas_batches = np.array_split(meas_list, indices, axis=0)
    sub_batches = np.array_split(sub, indices, axis=0)
    point_batches = np.array_split(point_list, indices, axis=0)
    ci_batches = np.array_split(cis, indices, axis=0)
    lamb_norm_batches = np.array_split(x0_norm[:, jnp.newaxis], indices, axis=0)

    # Priors never change
    # inv = jax.jit(invert_analytical_point, static_argnums=0)
    inv = None
    inv = jax.jit(invert_analytical, static_argnums=0)

    n_workers = 10
    ray.init(num_cpus=n_workers)
    # Put worker args into Ray object
    params = [ray.put(fm), ray.put(inv), ray.put(jlut), ray.put(winidx)]
    workers = ray.util.ActorPool(
        [Worker.remote(*params) for n in range(n_workers)]
    )

    inputs = [
        [x0_d, meas_d, sub_d, point_d, ci_d, lamb_norm_d]
        for x0_d, meas_d, sub_d, point_d, ci_d, lamb_norm_d
        in zip(
            x0_batches,
            meas_batches,
            sub_batches,
            point_batches,
            ci_batches,
            lamb_norm_batches
        )
    ]
    start_time = time.time()
    res = list(
        workers.map_unordered(
            lambda a, b: a.run.remote(*b), inputs
        )
    )
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    inv_ref = ray.put(inv)
    fm_ref = ray.put(fm)

    result_refs = [
        ray_analytical.remote(
            fm_ref,
            inv_ref,
            jlut_ref,
            x0_batches[i],
            meas_batches[i],
            sub_batches[i],
            point_batches[i],
            ci_batches[i],
            lamb_norm_batches[i], 
            L_atm,
            L_dir_dir,
            L_dir_dif,
            L_dif_dir,
            L_dif_dif,
            s_alb,
            winidx
        ) for i in range(len(indices))
    ]
    lp = np.concatenate(ray.get(result_refs), axis=0)

    aoe = []
    total_start = time.time()
    for i, index in enumerate(indices):
        start_time = time.time()
        print(f"Batch: {i} of {len(meas_batches)}")
        L_atm = jlut['rhoatm'](point_batches[i])
        L_dir_dir = jlut['dir_dir'](point_batches[i])
        L_dir_dif = jlut['dir_dif'](point_batches[i])
        L_dif_dir = jlut['dif_dir'](point_batches[i])
        L_dif_dif = jlut['dif_dif'](point_batches[i])
        s_alb = jlut['sphalb'](point_batches[i])

        res = inv(
            fm,
            x0_batches[i],
            meas_batches[i],
            sub_batches[i],
            point_batches[i],
            ci_batches[i],
            lamb_norm_batches[i], 
            L_atm,
            L_dir_dir,
            L_dir_dif,
            L_dif_dir,
            L_dif_dif,
            s_alb,
            winidx
        )
        res.block_until_ready()

        aoe.append(res)
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time}")

    aoe = np.concatenate(aoe)
    total_end = time.time()
    print(f"Total elapsed time: {total_end - total_start}")

