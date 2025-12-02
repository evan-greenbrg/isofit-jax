import os

import numpy as np

from isojax.lut import check_bounds


def import_jax(ncores):
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={ncores}'
    import jax

    return ncores


def largest_divisible_core(divisor, ncore=10):
    largest_divisible = None
    for num in range(1, ncore+ 1):
        if divisor % num == 0:
            if largest_divisible is None or num > largest_divisible:
                largest_divisible = num
    return largest_divisible


def construct_point_list(lut, obs_flat, loc_flat, aod=0.1, h2o=2.0):
    point_list = np.zeros((len(obs_flat), 6))
    point_list[:, 0] = aod
    point_list[:, 1] = h2o
    point_list[:, 2] = obs_flat[:, 2]

    delta_phi = np.abs(obs_flat[:, 3] - obs_flat[:, 1])
    point_list[:, 3] = np.minimum(delta_phi, 360 - delta_phi)
    point_list[:, 4] = obs_flat[:, 4]
    point_list[:, 5] = loc_flat[:, 2]

    bounds = np.array([
        [
            np.min(lut.coords['AOT550'].values),
            np.max(lut.coords['AOT550'].values)
        ],
        [
            np.min(lut.coords['H2OSTR'].values),
            np.max(lut.coords['H2OSTR'].values)
        ],
        [
            np.min(lut.coords['observer_zenith'].values),
            np.max(lut.coords['observer_zenith'].values)
        ],
        [
            np.min(lut.coords['relative_azimuth'].values),
            np.max(lut.coords['relative_azimuth'].values)
        ],
        [
            np.min(lut.coords['solar_zenith'].values),
            np.max(lut.coords['solar_zenith'].values)
        ],
        [
            np.min(lut.coords['surface_elevation_km'].values),
            np.max(lut.coords['surface_elevation_km'].values)
        ],
    ])

    return check_bounds(
        point_list,
        bounds[:, 0],
        bounds[:, 1]
    )
