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

from gpfy.likelihoods import Gaussian
from gpfy.model import GP
from gpfy.spherical import NTK, PolynomialDecay
from gpfy.spherical_harmonics import SphericalHarmonics
from gpfy.variational import VariationalDistributionTriL

key = jr.PRNGKey(42)

dims = [3, 5]
num_independent_processes = [1, 7]


def _create_model_and_variational():
    k = PolynomialDecay()
    k = NTK()
    m = GP(kernel=k)
    sh = SphericalHarmonics(5, 10)
    q = VariationalDistributionTriL()
    return m, sh, q


def test_conditional_model_has_non_empty_tuple():
    m, sh, q = _create_model_and_variational()
    assert not m.conditional_fn  # assert tuple is empty

    m = m.conditional(sh, q)
    assert m.conditional_fn


@pytest.mark.parametrize("num_processes", num_independent_processes)
@pytest.mark.parametrize("input_dim", dims)
def test_predict_mean(input_dim, num_processes):
    m, sh, q = _create_model_and_variational()
    lik = Gaussian()
    param = m.init(
        key,
        input_dim,
        num_processes,
        num_processes,
        likelihood=lik,
        sh_features=sh,
        variational_dist=q,
    )

    x = jr.normal(key, (10, input_dim), dtype=jnp.float64)
    # predict from prior should be 0
    mu = m.mu(param)(x)
    np.testing.assert_allclose(mu, 0)
    np.testing.assert_array_equal(mu.shape, (10, 1))

    # condition on the var features
    m = m.conditional(sh, q)
    param = m.init(
        key,
        input_dim,
        num_processes,
        num_processes,
        likelihood=lik,
        sh_features=sh,
        variational_dist=q,
    )

    mu = m.mu(param)(x)
    np.testing.assert_array_equal(mu.shape, (10, num_processes))


@pytest.mark.parametrize("num_processes", num_independent_processes)
@pytest.mark.parametrize("input_dim", dims)
def test_predict_var(input_dim, num_processes):
    m, sh, q = _create_model_and_variational()
    lik = Gaussian()
    param = m.init(
        key,
        input_dim,
        num_processes,
        num_processes,
        likelihood=lik,
        sh_features=sh,
        variational_dist=q,
    )

    x = jr.normal(key, (10, input_dim), dtype=jnp.float64)
    # predict from prior should be k.K_diag
    var = m.var(param)(x)
    np.testing.assert_allclose(var, m.kernel.K_diag(param, x))
    np.testing.assert_array_equal(var.shape, (10,))

    # condition on the var features
    m = m.conditional(sh, q)
    param = m.init(
        key,
        input_dim,
        num_processes,
        num_processes,
        likelihood=lik,
        sh_features=sh,
        variational_dist=q,
    )

    var = m.var(param)(x)
    np.testing.assert_array_equal(var.shape, (10, num_processes))


@pytest.mark.parametrize("num_processes", num_independent_processes)
@pytest.mark.parametrize("input_dim", dims)
def test_predict_cov(input_dim, num_processes):
    m, sh, q = _create_model_and_variational()
    lik = Gaussian()
    param = m.init(
        key,
        input_dim,
        num_processes,
        num_processes,
        likelihood=lik,
        sh_features=sh,
        variational_dist=q,
    )

    x = jr.normal(key, (10, input_dim), dtype=jnp.float64)
    # predict from prior should be k.K
    cov = m.cov(param)(x)
    np.testing.assert_allclose(cov, m.kernel.K(param, x))
    np.testing.assert_array_equal(cov.shape, (10, 10))

    # condition on the var features
    m = m.conditional(sh, q)
    param = m.init(
        key,
        input_dim,
        num_processes,
        num_processes,
        likelihood=lik,
        sh_features=sh,
        variational_dist=q,
    )

    cov = m.cov(param)(x)
    np.testing.assert_array_equal(cov.shape, (num_processes, 10, 10))


@pytest.mark.parametrize("num_processes", num_independent_processes)
@pytest.mark.parametrize("input_dim", dims)
def test_predict_diag_same_as_mu_and_var(input_dim, num_processes):
    m, sh, q = _create_model_and_variational()
    m = m.conditional(sh, q)

    lik = Gaussian()
    x = jr.normal(key, (10, input_dim), dtype=jnp.float64)

    param = m.init(
        key,
        input_dim,
        num_processes,
        num_processes,
        likelihood=lik,
        sh_features=sh,
        variational_dist=q,
    )

    mu = m.mu(param)(x)
    var = m.var(param)(x)
    mu2, var2 = m.predict_diag(param, x)
    np.testing.assert_allclose(mu, mu2)
    np.testing.assert_allclose(var, var2)


@pytest.mark.parametrize("num_processes", num_independent_processes)
@pytest.mark.parametrize("input_dim", dims)
def test_predict_same_as_mu_and_cov(input_dim, num_processes):
    m, sh, q = _create_model_and_variational()
    m = m.conditional(sh, q)

    lik = Gaussian()
    x = jr.normal(key, (10, input_dim), dtype=jnp.float64)

    param = m.init(
        key,
        input_dim,
        num_processes,
        num_processes,
        likelihood=lik,
        sh_features=sh,
        variational_dist=q,
    )

    mu = m.mu(param)(x)
    cov = m.cov(param)(x)
    mu2, cov2 = m.predict(param, x)
    np.testing.assert_allclose(mu, mu2)
    np.testing.assert_allclose(cov, cov2)
