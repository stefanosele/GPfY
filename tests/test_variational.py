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

from gpfy.spherical_harmonics import SphericalHarmonics
from gpfy.variational import (
    VariationalDistributionFullCovariance,
    VariationalDistributionTriL,
)

key = jr.PRNGKey(42)

num_inducing_features_list = [2, 5, 10]
num_independent_processes_list = [1, 7]
var_classes = [VariationalDistributionFullCovariance, VariationalDistributionTriL]


def test_init_fails_when_passing_param_with_wrong_size_of_inducing_features():
    sh = SphericalHarmonics(5, 10)
    sh_param = sh.init(key, input_dim=4)
    num_inducing = sh.num_inducing(sh_param)
    q = VariationalDistributionTriL()
    with pytest.raises(ValueError):
        _ = q.init(num_inducing_features=num_inducing + 1, param=sh_param)


@pytest.mark.parametrize("num_inducing", num_inducing_features_list)
@pytest.mark.parametrize("num_processes", num_independent_processes_list)
@pytest.mark.parametrize("var_class", var_classes)
def test_init_create_correct_variables(num_inducing, num_processes, var_class):
    q = var_class()
    param = q.init(num_inducing_features=num_inducing, num_independent_processes=num_processes)

    assert "mu" in param.params["variational"][q.name]
    np.testing.assert_array_equal(q.mu(param).shape, (num_inducing, num_processes))
    assert "Sigma" in param.params["variational"][q.name]
    np.testing.assert_array_equal(
        q.cov_part(param).shape, (num_processes, num_inducing, num_inducing)
    )


@pytest.mark.parametrize("num_inducing", num_inducing_features_list)
@pytest.mark.parametrize("num_processes", num_independent_processes_list)
@pytest.mark.parametrize("var_class", var_classes)
def test_log_det(num_inducing, num_processes, var_class):
    q = var_class()
    param = q.init(num_inducing_features=num_inducing, num_independent_processes=num_processes)
    num_processes = q.mu(param).shape[-1]

    # initialised to identity so logdet should be 0.
    np.testing.assert_array_equal(q.logdet(param), jnp.zeros((num_processes,)))


@pytest.mark.parametrize("num_inducing", num_inducing_features_list)
@pytest.mark.parametrize("num_processes", num_independent_processes_list)
@pytest.mark.parametrize("var_class", var_classes)
def test_trace(num_inducing, num_processes, var_class):
    q = var_class()
    param = q.init(num_inducing_features=num_inducing, num_independent_processes=num_processes)
    num_inducing, num_processes = q.mu(param).shape

    # initialised to identity so trace should be num_inducing.
    np.testing.assert_array_equal(q.trace(param), num_inducing * jnp.ones((num_processes,)))


@pytest.mark.parametrize("num_inducing", num_inducing_features_list)
@pytest.mark.parametrize("num_processes", num_independent_processes_list)
@pytest.mark.parametrize("var_class", var_classes)
def test_projection_shapes(num_inducing, num_processes, var_class):
    q = var_class()
    param = q.init(num_inducing_features=num_inducing, num_independent_processes=num_processes)
    num_inducing, num_processes = q.mu(param).shape

    x = jr.normal(key, (num_inducing, 13))

    proj = q.project_mean(param, x)
    np.testing.assert_array_equal(proj.shape, (x.shape[-1], num_processes))

    proj = q.project_variance(param, x)
    np.testing.assert_array_equal(proj.shape, (num_processes, x.shape[-1], x.shape[-1]))

    proj = q.project_diag_variance(param, x)
    np.testing.assert_array_equal(proj.shape, (x.shape[-1], num_processes))
