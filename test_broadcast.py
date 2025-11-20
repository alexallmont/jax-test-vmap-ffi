"""
Test that np and jnp broadcast behaviour is the same.
Also confirms where jax vmap differs. See sections of code with `method != "vmap"`

NumPy can broadcast two arrays if:
- Right-align the shapes
- For each dimension: they match OR one is 1
- Missing dimensions are treated as 1
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

def add(shape_a: tuple, shape_b: tuple, method: str):
    """
    Add two shapes of ones together using a particular numpy module method and optionally vmap

    method may be either "np", "jnp" or "vmap", the latter being jax-specific.
    """
    module = np if method == "np" else jnp
    a = module.ones(shape_a)
    b = module.ones(shape_b)
    impl = lambda x, y: x + y
    if method == "vmap":
        impl = jax.vmap(impl)
    return impl(a, b)


def dummy_cpp_add_vmap(shape_a: tuple, shape_b: tuple):
    """
    Simulate calling convention of above add method without doing actual add

    Used to understand exceptional vmap cases, as we don't want to actually call JAX's add method because it has its
    own internal constraints on behaviour, so this gives an idea of what one may do with their own FFI.
    """
    a = jnp.ones(shape_a)
    b = jnp.ones(shape_b)
    impl = jax.vmap(lambda x, _: x)
    return impl(a, b)


@pytest.mark.parametrize("method", ["np", "jnp", "vmap"])
def test_broadcast(method):
    assert(add((1,), (1,), method).shape == (1,))
    assert(add((1, 1, 1), (1,), method).shape == (1, 1, 1))

    # Rule 1 — Compare shapes from the trailing (rightmost) dimension
    #   (5, 4, 3)
    #      (4, 3)
    if method != "vmap":
        assert(add((5, 4, 3, 2), (3, 2), method).shape == (5, 4, 3, 2))
        assert(add((5, 4, 3, 2), (4, 3, 2), method).shape == (5, 4, 3, 2))

    with pytest.raises(Exception):
        add((5, 4, 3, 2), (5, 4), method)
    with pytest.raises(Exception):
        add((5, 4, 3, 2), (4, 3), method)

    # Rule 2 — Dimensions must be equal OR one must be 1
    #   compatible:  Y  Y  Y  N
    #               (3, 3, 1, 5)
    #               (3, 1, 7, 4)
    assert(add((5, 4, 3), (5, 4, 3), method).shape == (5, 4, 3))
    if method != "vmap":
        assert(add((5, 4, 3), (1, 4, 3), method).shape == (5, 4, 3))
    assert(add((5, 4, 3), (5, 1, 3), method).shape == (5, 4, 3))
    assert(add((5, 4, 3), (5, 4, 1), method).shape == (5, 4, 3))

    with pytest.raises(Exception):
        add((5, 4, 3), (2, 4, 3), method)
    with pytest.raises(Exception):
        add((5, 4, 3), (5, 2, 3), method)
    with pytest.raises(Exception):
        add((5, 4, 3), (5, 3, 2), method)

    # Rule 3 — If one array has fewer dimensions, pad with 1s on the left
    #   A: (3, 4)
    #   B: (4,)
    #   B becomes (1, 4)
    if method != "vmap":
        assert(add((3, 4), (4,), method).shape == (3, 4))
        assert(add((2, 3, 4), (4,), method).shape == (2, 3, 4))
        assert(add((2, 3, 4), (3, 4,), method).shape == (2, 3, 4))

    if method != "vmap":
        with pytest.raises(Exception):
            add((2, 3, 4), (2,), method)
    else:
        add((2, 3, 4), (2,), method)

        # Exceptional vmap allowing 2 in leftmost is because it looks like it's only considering first dimension
        dummy_cpp_add_vmap((2, 3), (2, 3))
        dummy_cpp_add_vmap((2, 3), (2, 7))
        dummy_cpp_add_vmap((2, 3, 4), (2, 3, 7))
        dummy_cpp_add_vmap((2, 3, 4), (2, 5, 7)) # Particularly interesting, second dimension completely ignored
        with pytest.raises(Exception):
            dummy_cpp_add_vmap((2, 3, 4), (3, 5, 7)) # Confirm that the first dimension is significant

    with pytest.raises(Exception):
        add((2, 3, 4), (3,), method)
