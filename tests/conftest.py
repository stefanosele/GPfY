import jax
import jax.random as jr
import pytest

from shgp.spherical import NTK

jax.config.update("jax_enable_x64", True)
key = jr.PRNGKey(42)


@pytest.fixture
def kernel_param():
    k = NTK(depth=3, ard=True)
    return k.init(key, input_dim=5)
