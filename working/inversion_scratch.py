from matplotlib import pyplot as plt
import optax


# SCRATCH

def fit(param, meas, point, xa, Sa, opt_state):
    value, grad = iv.step()(param, meas, point, xa, Sa)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(param, updates)

    return params, opt_state, grad

fit_vmap = jax.jit(jax.vmap(fit))
learning_rate = 3e-3
nsteps=25

cmap_name = 'winter'
cmap = plt.get_cmap(cmap_name)
colors = [cmap(i) for i in np.linspace(0, 1, nsteps)]

transforms = {
    'surface': optax.adam(learning_rate=1e-3),
    'aod': optax.adam(learning_rate=2e-3),
    'h2o': optax.adam(learning_rate=1e-1),
}
optimizer = optax.multi_transform(
    transforms,
    {'surface': 'surface', 'aod': 'aod', 'h2o': 'h2o'}
)

batchsize = 20
indices = np.arange(batchsize, len(meas_list), batchsize)
x0_batches = np.array_split(
    x0_list, indices, axis=0
)
meas_batches = np.array_split(
    meas_list, indices, axis=0
)
point_batches = np.array_split(
    point_list, indices, axis=0
)
ci_batches = np.array_split(
    cis, indices, axis=0
)
lamb_norm_batches = np.array_split(
    x0_norm, indices, axis=0
)
atm_batches = np.array_split(
    atm_flat, indices, axis=0
)

iv = Invert(fm, optimizer)
inv_vmap = jax.jit(jax.vmap(iv.run, in_axes=(0, 0, 0, 0, 0, 0)))
x_surfaces = []
x_aods = []
x_h2os = []
grad_surfaces = []
grad_aods = []
grad_h2os = []
total_start = time.time()
for i in range(4):
    print(f"Shard: {i} of {len(meas_batches)}")
    start = time.time()
    (
        x_surface, 
        x_aod, 
        x_h2o, 
        grad_surface, 
        grad_aod, 
        grad_h2o, 
    ) = inv_vmap(
        x0_batches[i],
        meas_batches[i],
        point_batches[i],
        ci_batches[i][:, jnp.newaxis],
        lamb_norm_batches[i][:, jnp.newaxis],
        atm_batches[i]
    )
    x_surface.block_until_ready()

    x_surfaces.append(x_surface)
    x_aods.append(x_aod)
    x_h2os.append(x_h2o)
    grad_surfaces.append(grad_surface)
    grad_aods.append(grad_aod)
    grad_h2os.append(grad_h2o)
    end = time.time()
    print(f"Finished_batch: {round(end - start, 2)}s")
total_end = time.time()
print(f"Finished_all: {round(total_end - total_start, 2)}s")


# Plot
i = 10
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(wl, x_batch[i, :], color='red', lw=2, ls='--')
axs[0].plot(wl, params['surface'][i, :], color='black', lw=2)
axs[0].set_ylim([-.05, .5])
axs[0].set_title(f"H2O: {params['h2o'][i]}, AOD: {params['aod'][i]}")

axs[1].plot(wl, grads['surface'][i, :], color='black', lw=2)
plt.show()


for i in range(2):
# Set up the inversion for a single pixel
    x = x0_list[i, ...]
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
    opt_state = optimizer.init(param)

    params = []
    grads = []
    for i in range(nsteps):
        param, opt_state, grad = fit(param, meas, point, xa, Sa, opt_state)
        params.append(param)
        grads.append(grad)

    # TODO The gradients are bad. The optimization is wrong.
    x_surfaces = []
    x_aods = []
    x_h2os = []

    grad_surfaces = []
    grad_aods = []
    grad_h2os = []
    for param, grad in zip(params, grads):
        x_surfaces.append(param['surface'])
        x_aods.append(param['aod'])
        x_h2os.append(param['h2o'])

        grad_surfaces.append(grad['surface'])
        grad_aods.append(grad['aod'])
        grad_h2os.append(grad['h2o'])

    x_surfaces = np.array(x_surfaces)
    grad_surfaces = np.array(grad_surfaces)
    x_aods = np.array(x_aods)
    grad_aods = np.array(grad_aods)
    x_h2os = np.array(x_h2os)
    grad_h2os = np.array(grad_h2os)

    fig, axs = plt.subplots(2, 1, sharex=True)
    for i in range(nsteps):
        axs[0].plot(wl, x_surfaces[i,:285], color=colors[i])
        axs[1].plot(wl, grad_surfaces[i,:285], color=colors[i])
    # plt.plot(wl, params[0, :], color='black', ls='--', lw=2)
    # plt.plot(wl, params[-1, :], color='black', lw=2)
    axs[0].set_ylim([-.05, .5])
    # axs[1].set_ylim([-.05, .5])
    plt.show()

    fig, axs = plt.subplots(2, 2, sharex=True)
    axs[0, 0].set_ylabel('AOD')
    axs[0, 0].scatter(
        [i for i in range(nsteps)],
        x_aods
    )
    axs[1, 0].set_ylabel('AOD Grad')
    axs[1, 0].scatter(
        [i for i in range(nsteps)],
        grad_aods
    )
    axs[0, 1].set_ylabel('H2O')
    axs[0, 1].scatter(
        [i for i in range(nsteps)],
        x_h2os
    )
    axs[1, 1].set_ylabel('H2O Grad')
    axs[1, 1].scatter(
        [i for i in range(nsteps)],
        grad_h2os
    )
    plt.show()



