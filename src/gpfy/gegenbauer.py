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

from typing import Tuple, Union

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import custom_vjp
from jax.tree_util import Partial
from jaxtyping import Array, Bool, Float

from gpfy.utils import dataclass, field


@jax.jit
def _gegenbauer(level: int, alpha: Union[float, Float[Array, ""]], x: Array) -> Array:
    """
    Compute the gegenbauer polynomial Cᵅₙ(x) recursively.

    Cᵅ₀(x) = 1
    Cᵅ₁(x) = 2αx
    Cᵅₙ(x) = (2x(n + α - 1) Cᵅₙ₋₁(x) - (n + 2α - 2) Cᵅₙ₋₂(x)) / n

    Args:
        level: The order of the polynomial.
        alpha: The hyper-sphere constant given by (d - 2) / 2 for the Sᵈ⁻¹ sphere.
        x: Input array.

    Returns:
        The Gegenbauer polynomial evaluated at `x`.
    """
    C_0 = jnp.ones_like(x, dtype=x.dtype)
    C_1 = 2 * alpha * x

    def step(Cs_and_n: Tuple[Array, Array, Float[Array, ""]]) -> Tuple[Array, Array, Array]:
        C, C_prev, n = Cs_and_n
        C, C_prev = (2 * x * (n + alpha - 1) * C - (n + 2 * alpha - 2) * C_prev) / n, C
        return C, C_prev, n + 1

    def cond(Cs_and_n: Tuple[Array, Array, Float[Array, ""]]) -> Bool[Array, ""]:
        n = Cs_and_n[2]
        return n <= level

    return jax.lax.cond(
        level == 0,
        lambda: C_0,
        lambda: jax.lax.while_loop(cond, step, (C_1, C_0, jnp.array(2, jnp.float64)))[0],
    )


@custom_vjp
def gegenbauer(level, alpha, x):
    """Gegenbauer function with custom gradient."""
    return _gegenbauer(level, alpha, x)


# @jax.jit
def gegenbauer_fwd(level, alpha, x):
    """The forward-mode differentiation for the Gegenbauer polynomial wrt the input `x`."""
    geg = _gegenbauer(level, alpha, x)
    g_fn = lambda: 2 * alpha * _gegenbauer(level - 1, alpha + 1, x)
    grad = jax.lax.cond(level == 0, lambda: jnp.zeros_like(x), g_fn)
    return geg, grad


# @jax.jit
def gegenbauer_bwd(res, cotangents):
    """The backward pass for the custom gradient of Gegenbauer polynomial wrt the input `x`."""
    return (None, None, res * cotangents)


# declare the custom vjp
gegenbauer.defvjp(gegenbauer_fwd, gegenbauer_bwd)


@dataclass
class GegenbauerLookupTable:
    """
    Gegenbauer polynomial via interpolation from a lookup table.

    The Gegenbauer polynomial are defined on the interval [-1, 1]. So we first evaluate offline on a
    dense grid the gegenbauer and its gradient on the interval. Then during training we interpolate
    from the lookup table.

    Attributes:
        max_level: The maximum order of the polynomial up to which we will evaluate on a grid.
        alpha: The hyper-sphere constant given by (d - 2) / 2 for the Sᵈ⁻¹ sphere.
        grid_resolution: The number of points in the linspace of [-1, 1]. Defautls to `100_000`.
        grid_evaluations: The grid evaluation of all orders of the Gegenbauer.
        grid_grad_evaluations: The grid evaluation of all orders of the gradient of Gegenbauer.
    """

    max_level: int = field(default_factory=int, pytree_node=False)
    alpha: float = field(default_factory=float, pytree_node=False)
    grid_resolution: int = field(default=100_000, pytree_node=False)
    grid_evaluations: jax.Array = field(init=False, pytree_node=False)
    grid_grad_evaluations: jax.Array = field(init=False, pytree_node=False)

    def __post_init__(self):
        """Evaluate the Gegenbauer and the gradients and save it to the relevant lookup tables."""
        x_grid = jnp.linspace(-1, 1, self.grid_resolution)

        # grid evaluation for the Gegenbauer
        part_gegenbauer = Partial(_gegenbauer, alpha=self.alpha, x=x_grid)
        grid_evaluations = jax.vmap(part_gegenbauer)(
            jnp.arange(self.max_level + 1, dtype=jnp.int32)
        )

        # grid evaluation for the gradient of Gegenbauer
        part_gegenbauer = Partial(_gegenbauer, alpha=self.alpha + 1, x=x_grid)
        grid_grad_evaluations = (
            2 * self.alpha * jax.vmap(part_gegenbauer)(jnp.arange(self.max_level, dtype=jnp.int32))
        )
        grid_grad_evaluations = jnp.stack([jnp.zeros_like(x_grid), *grid_grad_evaluations])
        grid_grad_evaluations = grid_grad_evaluations

        object.__setattr__(self, "grid_evaluations", grid_evaluations)
        object.__setattr__(self, "grid_grad_evaluations", grid_grad_evaluations)

    def __call__(self, level: int, alpha: Union[float, Float[Array, ""]], x: Array) -> Array:
        """
        Compute the Gegenbauer polynomial via linear interpolation from the lookup table.

        NOTE: The `alpha` argument is ignored as the Gegenbauer has been already computed with a
        user defined alpha.

        Args:
            level: The order of the polynomial.
            alpha: The hyper-sphere constant given by (d - 2) / 2 for the Sᵈ⁻¹ sphere.
            x: Input array.

        Returns:
            The interpolated Gegenbauer polynomial evaluated at the input `x`.
        """
        return gegenbauer_from_lookup(level, self.grid_evaluations, self.grid_grad_evaluations, x)


@custom_vjp
@jax.jit
def gegenbauer_from_lookup(
    level: float, grid_eval: Float[Array, "N M"], grid_grad_eval: Float[Array, "N M"], x: Array
) -> Array:
    """
    Evaluate the Gegenbauer polynomial and the custom gradient via interpoloation on a grid.

    Args:
        level: The order of the polynomial.
        grid_eval: The lookup table with the Gegenbauer evaluation.
        grid_grad_eval: The lookup table with the gradient of Gegenbauer.
        x: Input array.

    Returns:
        The interpolated Gegenbauer polynomial evaluated at the input `x`.
    """
    return tfp.math.interp_regular_1d_grid(x, -1.0, 1.0, grid_eval[jnp.array(level, jnp.int32)])


def gegenbauer_from_lookup_fwd(
    level: float, grid_eval: Float[Array, "N M"], grid_grad_eval: Float[Array, "N M"], x: Array
) -> Tuple[Array, Array]:
    """The forward-mode custom gradient wrt the input `x`."""
    return (
        gegenbauer_from_lookup(level, grid_eval, grid_grad_eval, x),
        tfp.math.interp_regular_1d_grid(x, -1.0, 1.0, grid_grad_eval[jnp.array(level, jnp.int32)]),
    )


def gegenbauer_from_lookup_bwd(res, cotangents):
    """The backward pass for the custom gradient wrt the input `x`."""
    return (None, None, None, res * cotangents)


# declare the custom vjp
gegenbauer_from_lookup.defvjp(gegenbauer_from_lookup_fwd, gegenbauer_from_lookup_bwd)


# @jax.custom_jvp
# def _evaluate2(level: int, grid_eval: Array, grid_grad_eval: Array, x: Array):
#     return tfp.math.interp_regular_1d_grid(x, -1.0, 1.0, grid_eval[int(level)])
#
#
# @_evaluate2.defjvp
# def _evaluate2_jvp(primals, tangents):
#     level, grid_eval, grad_eval, x = primals
#     *_, x_dot = tangents
#     dC_dx = tfp.math.interp_regular_1d_grid(x, -1.0, 1.0, grad_eval[int(level)])
#
#     primal_out = _evaluate2(*primals)
#     return primal_out, dC_dx * x_dot
#
#
# @jax.custom_gradient
# def _evaluate3(level: float, grid_eval: Array, grid_grad_eval: Array, x: Array):
#     def grad(g):
#         dC_dx = tfp.math.interp_regular_1d_grid(x, -1.0, 1.0, grid_grad_eval[int(level)])
#         return (None, None, None, dC_dx * g)
#
#     return tfp.math.interp_regular_1d_grid(x, -1.0, 1.0, grid_eval[int(level)]), grad
