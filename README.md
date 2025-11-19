# JAX vmap cpp tests

Experimental code to test vmaps for shaping RoughPy API

Docs to follow imminently. TODO: sharding and OMP. For now, test with:

    uv venv
    source .venv/bin/activate
    uv pip install jax pytest
    mkdir build
    cd build
    cmake ..
    make
    cd ..
    PYTHONPATH=build:src pytest
