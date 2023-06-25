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


# Borrowed and adapted from the excellent implementation of GPflow (and later on of GPJax)


from typing import Callable

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

"""The number of Gauss-Hermite points to use for quadrature"""
DEFAULT_NUM_GAUSS_HERMITE_POINTS = 20
gh_points, gh_weights = np.polynomial.hermite.hermgauss(DEFAULT_NUM_GAUSS_HERMITE_POINTS)


def gauss_hermite_quadrature(
    fun: Callable[[Float[Array, "N ... D"], Float[Array, "N ... D"]], Float[Array, "N ..."]],
    mean: Float[Array, "N D"],
    stddev: Float[Array, "N D"],
    *args,
    **kwargs,
) -> Callable[[Float[Array, "N D"]], Float[Array, " N"]]:
    """
    Compute Gaussian-Hermite quadrature for a given function. The quadrature
    points are adjusted through the supplied mean and variance arrays.
    Args:
        fun (Callable): The function for which quadrature should be applied to.
        mean (Float[Array, "N D"]): The mean of the Gaussian distribution that
            is used to shift quadrature points.
        sd (Float[Array, "N D"]): The standard deviation of the Gaussian
            distribution that is used to scale quadrature points.
    Returns:
        Float[Array, "N"]: The evaluated integrals value.
    """
    # [N, num_gh, D]

    X = mean[..., None, :] + jnp.sqrt(2.0) * stddev[..., None, :] * gh_points[..., None]
    W = gh_weights / jnp.sqrt(jnp.pi)  # [gh,]
    return lambda y: jnp.sum(fun(X, y[..., None, :]) * W, axis=-1)
