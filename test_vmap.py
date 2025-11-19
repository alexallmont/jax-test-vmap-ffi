import jax
import jax.numpy as jnp
import pytest

from vmap_cpp import _add

for name, target in _add.registrations().items():
  jax.ffi.register_ffi_target(name, target)


def _add_broadcast_all(x, y):
    call = jax.ffi.ffi_call(
        "add",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )
    return call(x, y)


def _add_sequential(x, y):
    call = jax.ffi.ffi_call(
        "add",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="sequential",
    )
    return call(x, y)


def _ones(n):
    return jnp.ones(n, dtype=jnp.float32)


def _zero_to_one(n):
    return jnp.linspace(0, 1, n, dtype=jnp.float32)


def _ones(n):
    return jnp.ones(n, dtype=jnp.float32)


def _add_native(x, y):
    return x + y


_add_native_vmap = jax.vmap(_add_native)
_add_broadcast_all_vmap = jax.vmap(_add_broadcast_all)
_add_sequential_vmap = jax.vmap(_add_sequential)


def test_shape_native_non_vmap():
    """
    Baseline JAX-native tests for 1D add, 2D add and 1D/2D add
    """
    x1 = _ones(3)
    y1 = _ones(3)
    assert jnp.allclose(_add_native(x1, y1), jnp.array([2, 2, 2]))

    x2 = jnp.tile(x1, (2, 1))
    y2 = jnp.tile(y1, (2, 1))
    assert jnp.allclose(_add_native(x2, y2), jnp.array([[2, 2, 2], [2, 2, 2]]))

    # non-vmap native code broadcasts all
    assert jnp.allclose(_add_native(x1, y2), jnp.array([[2, 2, 2], [2, 2, 2]]))


def test_shape_native_vmap():
    """
    1D/2D shape test as above but using vmap: JAX rejects differing shapes by default
    """
    x1 = _ones(3)
    y1 = _ones(3)
    assert jnp.allclose(_add_native_vmap(x1, y1), jnp.array([2, 2, 2]))

    x2 = jnp.tile(x1, (2, 1))
    y2 = jnp.tile(y1, (2, 1))
    assert jnp.allclose(_add_native_vmap(x2, y2), jnp.array([[2, 2, 2], [2, 2, 2]]))

    # Native vmap behaviour expects arrays with same shape (REVIEW: unless in_axis set TBC)
    with pytest.raises(ValueError) as excinfo:
        _add_native_vmap(x1, y2)
    assert("vmap got inconsistent sizes" in str(excinfo))


def test_shape_cpp_non_vmap_broadcast_all():
    """
    Baseline C++ tests for 1D add, 2D add and 1D/2D add: roughpy_jax rejects shape mismatch
    TODO return number of iterations from C++ (this dispatches 1 calls from JAX, 2 iterations in C++)
    """
    x1 = _ones(3)
    y1 = _ones(3)
    assert jnp.allclose(_add_broadcast_all(x1, y1), jnp.array([2, 2, 2]))

    x2 = jnp.tile(x1, (2, 1))
    y2 = jnp.tile(y1, (2, 1))
    assert jnp.allclose(_add_broadcast_all(x2, y2), jnp.array([[2, 2, 2], [2, 2, 2]]))

    # REVIEW: C++ code rejects different sizes internally, i.e. in RoughPy non-vmap methods enforce vmap constraints.
    with pytest.raises(jax.errors.JaxRuntimeError) as excinfo:
        _add_broadcast_all(x1, y2)
    assert("y must have same dimensions as x" in str(excinfo))


def test_shape_cpp_vmap_broadcast_all():
    """
    1D/2D C++ shape test as above but using vmap: JAX rejects differing shapes by default
    """
    x1 = _ones(3)
    y1 = _ones(3)
    assert jnp.allclose(_add_broadcast_all_vmap(x1, y1), jnp.array([2, 2, 2]))

    x2 = jnp.tile(x1, (2, 1))
    y2 = jnp.tile(y1, (2, 1))
    assert jnp.allclose(_add_broadcast_all_vmap(x2, y2), jnp.array([[2, 2, 2], [2, 2, 2]]))

    with pytest.raises(ValueError) as excinfo:
        _add_broadcast_all_vmap(x1, y2)
    assert("vmap got inconsistent sizes" in str(excinfo))


def test_shape_cpp_non_vmap_sequential():
    """
    Check that sequential behaves same as test_shape_cpp_non_vmap_broadcast_all:
    TODO return number of iterations from (this dispatches 2 separate calls from JAX)
    """
    x1 = _ones(3)
    y1 = _ones(3)
    assert jnp.allclose(_add_sequential(x1, y1), jnp.array([2, 2, 2]))

    x2 = jnp.tile(x1, (2, 1))
    y2 = jnp.tile(y1, (2, 1))
    assert jnp.allclose(_add_sequential(x2, y2), jnp.array([[2, 2, 2], [2, 2, 2]]))

    # REVIEW: C++ code rejects different sizes internally, i.e. in RoughPy non-vmap methods enforce vmap constraints.
    with pytest.raises(jax.errors.JaxRuntimeError) as excinfo:
        _add_sequential(x1, y2)
    assert("y must have same dimensions as x" in str(excinfo))


def test_shape_cpp_vmap_sequential():
    # FIXME similar to test_shape_cpp_vmap_broadcast_all but map data for _add_sequential for profiling
    pass


def test_vmap_deep():
    """
    Check that native vmap of deep arrays is the same native and cpp
    """
    I = 5
    J = 7
    K = 11
    x3 = jnp.tile(_zero_to_one(I), (K, J, 1))
    y3 = jnp.tile(_ones(I), (K, J, 1))

    # Add consistent offsets per rank to test varied output
    offset_k = jnp.linspace(1, K, K, jnp.float32)
    y3 += offset_k[:, None, None]
    offset_j = jnp.linspace(1, J, J, jnp.float32)
    y3 += offset_j[None, :, None]

    expected = _add_native_vmap(x3, y3)
    assert(expected.shape == (11, 7, 5))
    assert(expected[0, 0, 0] == 3) # Sanity check the offsets were applied correctly
    assert(expected[-1, -1, -1] == 20)

    cpp_add = _add_broadcast_all(x3, y3)
    assert(cpp_add.shape == (11, 7, 5))
    assert(jnp.allclose(cpp_add, expected))
