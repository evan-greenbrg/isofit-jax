import time
from functools import partial
import os
import itertools

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=10'
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

from spectral import envi
import optax

from isofit.radiative_transfer.luts import load
from isofit.core.common import envi_header
from isofit.core.geometry import Geometry
from isofit.core.fileio import IO
from isofit import ray

from interpolator import multilinear_interpolator



def component
