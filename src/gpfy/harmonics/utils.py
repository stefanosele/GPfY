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

from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float
from scipy.special import loggamma, roots_legendre

from gpfy.gegenbauer import gegenbauer

legendre_x, legendre_w = roots_legendre(1000)
legendre_x = jnp.array(legendre_x, dtype=jnp.float64)
legendre_w = jnp.array(legendre_w, dtype=jnp.float64)


def weight_func(t: Float[Array, " N"], d: int) -> Float[Array, " N"]:
    """
    The function which orthogonalises Gegenbauer polynomials in [-1, 1].

    Args:
        t: The input to the weight function and the Gegenbauer polynomial.
        d: The dimensionality of the input on the Sᵈ⁻¹ sphere.

    Returns:
        The function evaluation at `t`.
    """
    return (1.0 - jnp.square(t)) ** ((d - 3) / 2)


def funk_hecke_lambda(
    fn: Callable[[Float[Array, " N"]], Float[Array, " N"]], n: int, d: int
) -> Float[Array, ""]:
    """
    The general Funk-Hecke formula that gives the `n`th eigenvalue of function `fn` on Sᵈ⁻¹.

    It computes the integral::
        λₙ = ω / Cᵅₙ(1) ∫₋₁¹ f(t) Cᵅₙ(t) (1 - t²)ʳ dt,
    where::
        α = ⁽ᵈ⁻²⁾⁄₂, r = α - ¹⁄₂
    and::
        ω = |Sᵈ⁻²| / |Sᵈ⁻¹|

    Args:
        fn: The function which we integrate.
        n: The order of the eigenvalue
        d: The dimensionality of the input on the Sᵈ⁻¹ sphere.

    Returns:
        The nth eigenvalue.
    """
    alpha = (d - 2.0) / 2.0
    C1 = gegenbauer(n, alpha, jnp.array(1, dtype=jnp.float64))
    solid_angle_ratio = jnp.exp(loggamma(d / 2) - loggamma((d - 1) / 2)) / jnp.sqrt(jnp.pi)

    def integrand(x):
        return gegenbauer(n, alpha, x) * fn(x) * weight_func(x, d)

    integral = jnp.dot(legendre_w, integrand(legendre_x))
    return solid_angle_ratio / C1 * integral
