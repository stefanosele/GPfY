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

from gpfy.param import Param, identity, positive
from gpfy.spherical import NTK, ArcCosine, PolynomialDecay

key = jr.PRNGKey(42)

kernels = [ArcCosine, NTK, PolynomialDecay]
dims = [2, 3, 5, 10]
order_list = [0, 1, 2]
ards = [True, False]
depths = [1, 2, 5]
projection_dims = [None, 2]
shape_list = [(1, 3), (10, 7)]


def _get_input_from_shape(shape):
    return jr.normal(key, shape, dtype=jnp.float64)


def _create_kernel(kernel, order, ard, depth):
    if kernel.__name__ == "ArcCosine":
        return kernel(order=order, ard=ard, depth=depth)
    elif kernel.__name__ == "NTK":
        return kernel(ard=ard, depth=depth)
    else:
        return kernel(ard=ard)


@pytest.mark.parametrize("kernel", kernels)
@pytest.mark.parametrize("order", order_list)
@pytest.mark.parametrize("ard", ards)
@pytest.mark.parametrize("depth", depths)
@pytest.mark.parametrize("input_dim", dims)
@pytest.mark.parametrize("projection_dim", projection_dims)
def test_kernel_init(kernel, order, ard, depth, input_dim, projection_dim):
    """Test the kernel initialisation."""
    if kernel.__name__ != "ArcCosine" and order != 1:
        return
    if kernel.__name__ == "PolynomialDecay" and depth != 1:
        return
    k = _create_kernel(kernel, order, ard, depth)
    param = k.init(key, input_dim=input_dim, projection_dim=projection_dim)
    collection = k.name  # the kernel name

    assert collection in param.params
    assert collection in param._bijectors
    assert collection in param._trainables
    assert "sphere" in param.constants

    variables = ["weight_variances", "bias_variance", "variance"]
    for v in variables:
        assert v in param.params[collection].keys()
        gt_bij = identity if v == "weight_variances" and projection_dim else positive
        assert isinstance(param._bijectors[collection][v], gt_bij)

    if collection == "PolynomialDecay":
        assert "beta" in param.params[collection].keys()
        assert isinstance(param._bijectors[collection]["beta"], positive)

    if projection_dim:
        assert param.params[collection]["weight_variances"].shape == (input_dim, projection_dim)
    else:
        if ard:
            assert param.params[collection]["weight_variances"].shape == (input_dim,)
        else:
            assert param.params[collection]["weight_variances"].shape == ()

    if kernel.__name__ == "PolynomialDecay":
        "gegenbauer_lookup_table" in param.constants["sphere"]
    else:
        "eigenvalues" in param.constants["sphere"]


@pytest.mark.parametrize("kernel", kernels)
def test_raises_for_inappropriate_order(kernel):
    """Test we error when we pass incompatible order value given the kernel class."""
    if kernel.__name__ == "ArcCosine":
        with pytest.raises(ValueError):
            kernel(order=4)
    else:
        with pytest.raises(TypeError):
            kernel(order=2)


@pytest.mark.parametrize("kernel", kernels)
@pytest.mark.parametrize("shape", shape_list)
@pytest.mark.parametrize("projection_dim", projection_dims)
def test_to_sphere_returns_unit_length_vector(kernel, shape, projection_dim):
    """Test the unit length of the projected vectors."""
    x = _get_input_from_shape(shape)
    dim = x.shape[-1]
    k = _create_kernel(kernel, 1, True, 3)

    param = k.init(key, input_dim=dim, projection_dim=projection_dim)

    x_sphere, radius = k.to_sphere(param, x)

    assert x_sphere.shape[-1] == param.constants["sphere"]["sphere_dim"] + 1
    np.testing.assert_allclose(jnp.sum(x_sphere**2, -1), jnp.ones_like(radius))

    x_scaled = k._scale_X(param, x)
    np.testing.assert_allclose(jnp.sum(x_scaled**2, -1), radius**2 - 1)  # bias initialised to 1


@pytest.mark.parametrize("kernel", kernels)
@pytest.mark.parametrize("shape", shape_list)
@pytest.mark.parametrize("projection_dim", projection_dims)
def test_scaled_X_returns_expected_shape(kernel, shape, projection_dim):
    """Test the shape of the projected vectors."""
    x = _get_input_from_shape(shape)
    dim = x.shape[-1]
    k = _create_kernel(kernel, 1, True, 3)

    param = k.init(key, input_dim=dim, projection_dim=projection_dim)
    x_scaled = k._scale_X(param, x)
    assert x_scaled.shape[-1] == (projection_dim or dim)


@pytest.mark.parametrize("kernel", kernels[:-1])
def test_eigenvalues_return_zero_if_not_precomputed(kernel):
    """Test we return 0 for non-precomputed eigvals."""
    k = _create_kernel(kernel, 1, True, 3)
    dummy_param = Param()
    np.testing.assert_allclose(k.eigenvalues(dummy_param, jnp.arange(10)), 0.0)


def test_eigenvalues_are_normalised():
    """Test the eigvals are normalised for the PolynomialDecay kernel."""
    k = _create_kernel(PolynomialDecay, 1, True, 3)
    param = k.init(key, input_dim=5)

    sphere_dim = param.constants["sphere"]["sphere_dim"]
    geg = param.constants["sphere"]["gegenbauer_lookup_table"]
    const_factor = k._compute_eigvals(sphere_dim, gegenbauer_lookup_table=geg)

    np.testing.assert_allclose(jnp.sum(k.eigenvalues(param, jnp.arange(10)) / const_factor), 1.0)


@pytest.mark.parametrize("kernel", kernels)
@pytest.mark.parametrize("order", order_list)
@pytest.mark.parametrize("depth", depths)
def test_shape_function_is_normalised(kernel, order, depth):
    """Test the shape_function is normalised so that κ(1) = 1."""
    k = _create_kernel(kernel, order, True, depth)
    param = k.init(key, input_dim=3)

    np.testing.assert_allclose(k.shape_function(param, jnp.ones((1, 1))), 1.0)


@pytest.mark.parametrize("kernel", kernels)
@pytest.mark.parametrize("order", order_list)
@pytest.mark.parametrize("ard", ards)
@pytest.mark.parametrize("depth", depths)
@pytest.mark.parametrize("projection_dim", projection_dims)
def test_K_vs_K_diag(kernel, order, ard, depth, projection_dim):
    """Test that k.K(X, X) and k.K_diag(X) give the same answer for the diagonal entries."""
    if kernel.__name__ != "ArcCosine" and order != 1:
        return
    if kernel.__name__ == "PolynomialDecay" and depth != 1:
        return

    x = _get_input_from_shape((10, 5))
    k = _create_kernel(kernel, order, ard, depth)
    param = k.init(key, input_dim=x.shape[-1], projection_dim=projection_dim)

    np.testing.assert_allclose(jnp.diag(k.K(param, x)), k.K_diag(param, x))


@pytest.mark.parametrize("kernel", kernels)
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("ard", ards)
@pytest.mark.parametrize("depth", depths)
@pytest.mark.parametrize("projection_dim", projection_dims)
def test_K(kernel, order, ard, depth, projection_dim):
    """Test that k.K(X) and k.K(X, X) give the same answer."""
    if kernel.__name__ != "ArcCosine" and order != 1:
        return
    if kernel.__name__ == "PolynomialDecay" and depth != 1:
        return

    x = _get_input_from_shape((10, 5))
    k = _create_kernel(kernel, order, ard, depth)
    param = k.init(key, input_dim=x.shape[-1], projection_dim=projection_dim)

    np.testing.assert_allclose(k.K(param, x), k.K(param, x, x))
    np.testing.assert_allclose(k.K(param, x), k.K2(param, x))
    np.testing.assert_allclose(k.K(param, x), k.K2(param, x, x))
