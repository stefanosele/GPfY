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

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import Array, custom_gradient, custom_jvp, custom_vjp
from jax.experimental import checkify
from jax.tree_util import Partial
from jaxtyping import ArrayLike, Float

from shgp.utils import dataclass, field


@jax.jit
def gegenbauer2(level: int, alpha: Float[ArrayLike, ""], x: Array):
    """gegenbauer implementation with jax.lax.scan"""
    alpha = jnp.array(alpha, dtype=jnp.float64)
    l = jnp.array(level, dtype=jnp.float64)

    C_0 = jnp.ones_like(x, dtype=jnp.float64)
    C_1 = 2 * alpha * x
    C_up_to_1 = jnp.where(l == 0, C_0, C_1)

    def step(
        Cs: Tuple[Array, Array, Array], iter_num: Optional[Array]
    ):  # -> Tuple[jax.Array, jax.Array]:
        C_n_minus1, C_n_minus2, n = Cs
        # n = jnp.array(iter_num, dtype=jnp.float64)
        C_next = (
            2 * x * (n + alpha - 1) * C_n_minus1 - (n + 2 * alpha - 2) * C_n_minus2
        ) / n
        return (C_next, C_n_minus1, n + 1), n

    # iters = jnp.arange(2, level + 1).astype(jnp.float64)
    C = jnp.where(
        l < 2,
        C_up_to_1,
        jax.lax.scan(step, (C_1, C_0, jnp.array(2, jnp.float64)), None, 3)[0][0],
    )
    # C = jnp.where(level < 2, C_up_to_1, jax.lax.scan(step, (C_1, C_0, level), iters)[0][0])
    return C


@jax.jit
def _gegenbauer(level: int, alpha: Float[ArrayLike, ""], x: Array):
    """gegenbauer implementation with jax.lax.while_loop"""
    alpha = jnp.array(alpha, dtype=jnp.float64)

    C_0 = jnp.ones_like(x, dtype=jnp.float64)
    C_1 = 2 * alpha * x

    def step(Cs_and_n: Tuple[Array, Array, Array]) -> Tuple[Array, Array, Array]:
        C, C_prev, n = Cs_and_n
        # n = jnp.array(n, dtype=jnp.float64)
        C, C_prev = (2 * x * (n + alpha - 1) * C - (n + 2 * alpha - 2) * C_prev) / n, C
        return C, C_prev, n + 1

    def cond(Cs_and_n: Tuple[Array, Array, int]) -> bool:
        n = Cs_and_n[2]
        return n <= level

    C, C_prev, _ = jax.lax.while_loop(cond, step, (C_1, C_0, jnp.array(2, jnp.float64)))  # type: ignore
    return jnp.where(level == 0, C_prev, C)


def geg(level: int, alpha: float, x: float):
    """gegenbauer implementation for one element used with jax.vmap"""
    # alpha = jnp.array(alpha, dtype=jnp.float64)

    C_0 = 1  # jnp.ones_like(x, dtype=jnp.float64)
    C_1 = 2 * alpha * x

    def step(Cs_and_n: Tuple[float, float, float]) -> Tuple[float, float, float]:
        C, C_prev, n = Cs_and_n
        # n = jnp.array(n, dtype=jnp.float64)
        C, C_prev = (2 * x * (n + alpha - 1) * C - (n + 2 * alpha - 2) * C_prev) / n, C
        return C, C_prev, n + 1

    def cond(Cs_and_n: Tuple[Array, Array, int]) -> bool:
        n = Cs_and_n[2]
        return n <= level

    C, C_prev, _ = jax.lax.while_loop(cond, step, (C_1, C_0, 2))  # type: ignore
    return jnp.where(level == 0, C_prev, C)


# @custom_gradient
# def gegenbauer(level, alpha, x):
#     alpha = jnp.array(alpha, dtype=jnp.float64)
#     c = _gegenbauer(level, alpha, x)
#
#     def grad(upstream):
#         g = 2 * alpha * _gegenbauer(level - 1, alpha + 1, x)
#         g = g * upstream
#         jnp.where(level == 0, jnp.zeros_like(g), g)
#         return (None, None, g)
#
#     return c, grad


@custom_vjp
def gegenbauer(level, alpha, x):
    # alpha = jnp.array(alpha, dtype=jnp.float64)
    # return gegenbauer2(level, alpha, x)
    return _gegenbauer(level, alpha, x)


@jax.jit
def gegenbauer_fwd(level, alpha, x):
    # g = 2 * alpha * gegenbauer2(level - 1, alpha + 1, x)
    # return (gegenbauer2(level, alpha, x), jnp.where(level == 0, jnp.zeros_like(g), g))
    g = 2 * alpha * _gegenbauer(level - 1, alpha + 1, x)
    return (_gegenbauer(level, alpha, x), jnp.where(level == 0, jnp.zeros_like(g), g))


@jax.jit
def gegenbauer_bwd(res, cotangents):
    return (None, None, res * cotangents)


gegenbauer.defvjp(gegenbauer_fwd, gegenbauer_bwd)


@dataclass
class GegenbauerLookupTable:
    max_level: int = field(default_factory=int, pytree_node=False)
    alpha: float = field(default_factory=float, pytree_node=False)
    grid_resolution: int = field(default=100_000, pytree_node=False)
    grid_evaluations: jax.Array = field(init=False, pytree_node=False)
    grid_grad_evaluations: jax.Array = field(init=False, pytree_node=False)

    def __post_init__(self):
        x_grid = jnp.linspace(-1, 1, self.grid_resolution)

        part_gegenbauer = Partial(_gegenbauer, alpha=self.alpha, x=x_grid)
        grid_evaluations = jax.vmap(part_gegenbauer)(jnp.arange(self.max_level + 1))

        # part_gegenbauer = Partial(gegenbauer, alpha=self.alpha + 1, x=x_grid)
        part_gegenbauer = (
            lambda n: 2 * self.alpha * _gegenbauer(n, alpha=self.alpha + 1, x=x_grid)
        )
        grid_grad_evaluations = jax.vmap(part_gegenbauer)(jnp.arange(self.max_level))
        grid_grad_evaluations = jnp.stack(
            [jnp.zeros_like(x_grid), *grid_grad_evaluations]
        )
        grid_grad_evaluations = grid_grad_evaluations

        object.__setattr__(self, "grid_evaluations", grid_evaluations)
        object.__setattr__(self, "grid_grad_evaluations", grid_grad_evaluations)

    def __call__(self, level: int, alpha: Float[ArrayLike, ""], x: Array):
        # checkify.check(alpha == self.alpha, "provided `alpha` is not the same as in the lookup")
        # assert alpha == self.alpha, "provided `alpha` is not the same as in the lookup"
        @custom_vjp
        def _gegenbauer_from_lookup(level: float, x: Array):
            return tfp.math.interp_regular_1d_grid(
                x, -1.0, 1.0, self.grid_evaluations[jnp.array(level, int)]
            )

        def _gegenbauer_from_lookup_fwd(level: float, x: Array):
            return (
                _gegenbauer_from_lookup(level, x),
                tfp.math.interp_regular_1d_grid(
                    x, -1.0, 1.0, self.grid_grad_evaluations[jnp.array(level, int)]
                ),
            )

        def _gegenbauer_from_lookup_bwd(res, cotangents):
            return (None, res * cotangents)

        _gegenbauer_from_lookup.defvjp(
            _gegenbauer_from_lookup_fwd, _gegenbauer_from_lookup_bwd
        )

        return _gegenbauer_from_lookup(level, x)

        # return gegenbauer_from_lookup(
        #     level, self.grid_evaluations, self.grid_grad_evaluations, x
        # )


@custom_vjp
def gegenbauer_from_lookup(
    level: float, grid_eval: Array, grid_grad_eval: Array, x: Array
):
    # resolution = grid_eval[jnp.array(level, int)].shape[0]
    # grid = jnp.linspace(-1.0, 1.0, resolution, dtype=jnp.float64)
    # return jnp.interp(x, grid, grid_eval[jnp.array(level, int)])
    return tfp.math.interp_regular_1d_grid(
        x, -1.0, 1.0, grid_eval[jnp.array(level, int)]
    )


def gegenbauer_from_lookup_fwd(
    level: float, grid_eval: Array, grid_grad_eval: Array, x: Array
):
    # resolution = grid_eval[jnp.array(level, int)].shape[0]
    # grid = jnp.linspace(-1.0, 1.0, resolution, dtype=jnp.float64)
    return (
        gegenbauer_from_lookup(level, grid_eval, grid_grad_eval, x),
        tfp.math.interp_regular_1d_grid(
            x, -1.0, 1.0, grid_grad_eval[jnp.array(level, int)]
        )
        # jnp.interp(x, grid, grid_grad_eval[jnp.array(level, int)])
    )


def gegenbauer_from_lookup_bwd(res, cotangents):
    return (None, None, None, res * cotangents)


gegenbauer_from_lookup.defvjp(gegenbauer_from_lookup_fwd, gegenbauer_from_lookup_bwd)


@custom_jvp
def _evaluate2(level: int, grid_eval: Array, grid_grad_eval: Array, x: Array):
    return tfp.math.interp_regular_1d_grid(x, -1.0, 1.0, grid_eval[int(level)])


@_evaluate2.defjvp
def _evaluate2_jvp(primals, tangents):
    level, grid_eval, grad_eval, x = primals
    *_, x_dot = tangents
    dC_dx = tfp.math.interp_regular_1d_grid(x, -1.0, 1.0, grad_eval[int(level)])

    primal_out = _evaluate2(*primals)
    return primal_out, dC_dx * x_dot


@custom_gradient
def _evaluate3(level: float, grid_eval: Array, grid_grad_eval: Array, x: Array):
    def grad(g):
        dC_dx = tfp.math.interp_regular_1d_grid(
            x, -1.0, 1.0, grid_grad_eval[int(level)]
        )
        return (None, None, None, dC_dx * g)

    return tfp.math.interp_regular_1d_grid(x, -1.0, 1.0, grid_eval[int(level)]), grad
