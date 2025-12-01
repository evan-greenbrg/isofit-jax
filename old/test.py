import numpy as np
import jax
jax.config.update('jax_num_cpu_devices', 15)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import AxisType


arr = jnp.arange(32.).reshape(4, 8)
arr.devices()

jax.debug.visualize_array_sharding(arr)

mesh = jax.make_mesh((2, 4), ('x', 'y'))
sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))
print(sharding)

arr_sharded = jax.device_put(arr, sharding)

print(arr_sharded)
jax.debug.visualize_array_sharding(arr_sharded)

@jax.jit
def f_elementwise(x):
  return 2 * jnp.sin(x) + 1

result_sharded = f_elementwise(arr_sharded)
result = f_elementwise(arr)

jax.debug.visualize_array_sharding(result)
jax.debug.visualize_array_sharding(result_sharded)


some_array = np.arange(8)
print(f"JAX-level type of some_array: {jax.typeof(some_array)}")

@jax.jit
def foo(x):
  print(f"JAX-level type of x during tracing: {jax.typeof(x)}")
  return x + x

foo(some_array)

mesh = jax.make_mesh(
    ([8]), 
    (["X"]),
    axis_types=(AxisType.Explicit)
)
sharded_array = jax.device_put(some_array, jax.NamedSharding(mesh, P("X")))
jax.debug.visualize_array_sharding(sharded_array)

@jax.jit
def foo(x):
    return x * 2

result = foo(sharded_array)
jax.debug.visualize_array_sharding(result)

print(f"replicated_array type: {jax.typeof(some_array)}")
print(f"sharded_array type: {jax.typeof(sharded_array)}")


# Manual parallelism - shard_map

mesh = jax.make_mesh((2, 4), ('x', 'y'))
f_test_sharded = jax.experimental.shard_map.shard_map(
    foo,
    mesh=mesh,
    in_specs=P('x', 'y'),
    out_specs=P('x', 'y')
)
arr = jnp.arange(32).reshape(2, 16)
res = f_test_sharded(arr)
jax.debug.visualize_array_sharding(res)
