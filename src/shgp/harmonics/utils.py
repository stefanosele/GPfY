# Copyright 2023 Stefanos Eleftheriadis
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
from jax import Array
from jaxtyping import ArrayLike
from scipy.special import comb, gamma, roots_legendre

from shgp.gegenbauer import gegenbauer

legendre_x, legendre_w = roots_legendre(1000)
legendre_x = jnp.array(legendre_x, dtype=jnp.float64)
legendre_w = jnp.array(legendre_w, dtype=jnp.float64)


def weight_func(t: ArrayLike, d: int):
    return (1.0 - jnp.square(t)) ** ((d - 3) / 2)


def funk_hecke_lambda(fun: Callable[[Array], Array], n: int, d: int) -> float:
    alpha = (d - 2.0) / 2.0
    C1 = gegenbauer(n, alpha, jnp.array(1, dtype=jnp.float64))
    solid_angle = gamma(d / 2) / gamma((d - 1) / 2) / jnp.sqrt(jnp.pi)

    def integrand(x):
        return gegenbauer(n, alpha, x) * fun(x) * weight_func(x, d)

    integral = jnp.sum(legendre_w * integrand(legendre_x))
    return solid_angle / C1 * integral
