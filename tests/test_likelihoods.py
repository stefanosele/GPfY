# Copyright 2023 Stefanos Eleftheriadis, James Hensman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from gpfy.likelihoods import Bernoulli, Gaussian

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
