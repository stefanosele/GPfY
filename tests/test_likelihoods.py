import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from shgp.likelihoods import Bernoulli, Gaussian

key = jr.PRNGKey(42)


def test_init_create_correct_variables():
    lik = Gaussian()
    param = lik.init()

    assert "variance" in param.params["likelihood"][lik.name]


@pytest.mark.parametrize("likelihood_class", [Gaussian, Bernoulli])
def test_variational_expectations(likelihood_class):
    """
    Here we make sure that the variational_expectations gives the same result
    as log_prob if the latent function has no uncertainty.
    """
    lik = likelihood_class()
    param = lik.init()
    y = jr.normal(key, (10, 5))
    f = jr.normal(key, (10, 5))
    fvar = jnp.zeros_like(f)

    r1 = lik.log_prob(param, f, y)
    r2 = lik.variational_expectations(param, f, fvar)(y)
    np.testing.assert_allclose(r1, r2)
