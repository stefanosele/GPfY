import jax
import jax.random as jr
import pytest

from shgp.spherical import NTK

jax.config.update("jax_enable_x64", True)
key = jr.PRNGKey(42)


@pytest.fixture
def kernel_param_5d():
    k = NTK(depth=3, ard=True)
    return k, k.init(key, input_dim=5)
