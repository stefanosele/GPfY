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

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.special import gegenbauer as sci_gegenbauer

from gpfy.gegenbauer import GegenbauerLookupTable, gegenbauer

key = jr.PRNGKey(42)


def _get_input_from_shape(shape):
    return jr.uniform(key, shape, minval=-1.0, maxval=1.0, dtype=jnp.float64)


alpha_list = [0.5, 1.5, 3.0, 10.0]
freq_list = [0, 1, 2, 5, 10]
shape_list = [(), (1, 1), (3, 1), (100, 7)]


@pytest.mark.parametrize("n", freq_list)
@pytest.mark.parametrize("alpha", alpha_list)
@pytest.mark.parametrize("shape", shape_list)
def test_gegenbauer_against_scipy(n, alpha, shape):
    """test gegenbauer while-loop vs scipy."""
    angle = _get_input_from_shape(shape)
    sci_geg = sci_gegenbauer(n, alpha)
    jnp.array_equal(sci_geg(angle), gegenbauer(n, alpha, angle))


@pytest.mark.parametrize("n", freq_list)
@pytest.mark.parametrize("alpha", alpha_list)
@pytest.mark.parametrize("shape", shape_list)
def test_gegenbauer_lookup_table_against_while_loop(n, alpha, shape):
    """test gegenbauer lookup table vs while-loop."""
    angle = _get_input_from_shape(shape)
    geg = GegenbauerLookupTable(max_level=n, alpha=alpha, grid_resolution=100_000)
    np.testing.assert_allclose(geg(n, alpha, angle), gegenbauer(n, alpha, angle), rtol=1e-5)


@pytest.mark.parametrize("n", freq_list)
@pytest.mark.parametrize("alpha", alpha_list)
def test_gegenbauer_gradient(n, alpha):
    """test gegenbauer gradient."""
    angle = _get_input_from_shape(shape=())
    geg = GegenbauerLookupTable(max_level=n, alpha=alpha, grid_resolution=100_000)
    lookup_grad = jax.grad(geg, argnums=2)(n, alpha, angle)
    gegenbauer_grad = jax.grad(gegenbauer, argnums=2)(n, alpha, angle)
    np.testing.assert_allclose(lookup_grad, gegenbauer_grad, rtol=1e-5)


def test_raises_when_initialise_lookuptable_with_grid_evaluations():
    """test we can't initialise grid_evaluations for LookupTable"""
    with pytest.raises(TypeError):
        GegenbauerLookupTable(2, 0.5, grid_evaluations=42)


def test_raises_when_initialise_lookuptable_with_grid_grad_evaluations():
    """test we can't initialise grid_grad_evaluations for LookupTable"""
    with pytest.raises(TypeError):
        GegenbauerLookupTable(2, 0.5, grid_grad_evaluations=42)
