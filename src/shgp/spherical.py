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

from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import Array, random
from jax.experimental import checkify
from jax.tree_util import Partial
from jaxtyping import ArrayLike, Float, Int

from shgp.gegenbauer import GegenbauerLookupTable
from shgp.harmonics.utils import funk_hecke_lambda
from shgp.param import Param, positive
from shgp.utils import dataclass, field

ActiveDims = Union[slice, list]
PRNG = Union[random.PRNGKeyArray, jax.Array]


def _slice_or_list(value: Optional[ActiveDims] = None):
    if value is None:
        return slice(None, None, None)
    if not isinstance(value, slice):
        return jnp.array(value, dtype=int)


@dataclass
class Spherical:
    order: int = field(default=1, pytree_node=False)
    ard: bool = field(default=True, pytree_node=False)
    active_dims: Optional[ActiveDims] = field(
        default_factory=_slice_or_list, pytree_node=False
    )
    name: str = field(default="Spherical", pytree_node=False)

    def __init_subclass__(cls):
        dataclass(cls)  # pytype: disable=wrong-arg-types

    def init(
        self, key: PRNG, input_dim: int, projection_dim: Optional[int] = None
    ) -> Param:
        bias_variance = jnp.array(1.0, dtype=jnp.float64)
        variance = jnp.array(1.0, dtype=jnp.float64)
        bijectors = {}
        if not projection_dim:
            if self.ard:
                weight_variances = jnp.ones(input_dim, dtype=jnp.float64)
            else:
                weight_variances = jnp.array(1.0, dtype=jnp.float64)
        else:
            weight_variances = random.normal(
                key, (input_dim, projection_dim), dtype=jnp.float64
            )
            bijectors = {"weight_variances": tfp.bijectors.Identity()}

        params = {
            "weight_variances": weight_variances,
            "bias_variance": bias_variance,
            "variance": variance,
        }
        collection = self.name
        sphere_dim = (projection_dim or input_dim) + 1
        alpha = (sphere_dim - 2.0) / 2.0
        constants = {}
        constants["sphere"] = {"sphere_dim": sphere_dim, "alpha": alpha}
        constants[self.name] = {}

        # constants["sphere_dim"] = sphere_dim
        if isinstance(self, PolynomialDecay):
            params["beta"] = jnp.array(1.0, dtype=jnp.float64)
            # alpha = (sphere_dim - 2.) / 2.
            gegenbauer_lookup = GegenbauerLookupTable(self.truncation_level, alpha)
            constants["sphere"]["gegenbauer_lookup_table"] = gegenbauer_lookup
        else:
            # sphere_dim = (projection_dim or input_dim) + 1
            eigenvalues = self._compute_eigvals(sphere_dim, max_num_eigvals=50)
            constants[self.name]["eigenvalues"] = eigenvalues

        # The default bijector is positive, so we only need to pass the bijector for the weights
        # in case it's Identity due to the linear projection
        return Param(
            params={collection: params},
            _bijectors={collection: bijectors},
            constants=constants,
        )

    def _compute_eigvals(
        self,
        sphere_dim: int,
        *,
        max_num_eigvals: Optional[int] = None,
        gegenbauer_lookup_table: Optional[GegenbauerLookupTable] = None,
    ):
        # checkify.check(
        #     bool(max_num_eigvals) ^ bool(gegenbauer_lookup_table),
        #     "One of `max_num_eigvals` and `gegenbauer_lookup_table` should be provided."
        # )
        if not isinstance(self, PolynomialDecay) and gegenbauer_lookup_table:
            raise ValueError("Lookup table is only used with `PolyDecay` kernel.")
        max_num_eigvals = max_num_eigvals or 50

        # _truncation_level = self.__dict__.get("truncation_level", max_num_eigvals)
        # gegenbauer_lookup_table = gegenbauer_lookup_table or GegenbauerLookupTable()

        # def _compute():
        # part_shape_function = Partial(self.shape_function, param=None)
        dummy_param = Param()
        part_shape_function = lambda x: self.shape_function(dummy_param, x=x)
        part_funk_hecke = lambda n: funk_hecke_lambda(
            part_shape_function, n, sphere_dim
        )
        return jax.vmap(part_funk_hecke)(jnp.arange(max_num_eigvals))

        # def _from_lookup():
        #     alpha = (sphere_dim - 2.) / 2.
        #     # gegenbauer_lookup_table: GegenbauerLookupTable  # helps mypy
        #     geg = lambda n: gegenbauer_lookup_table(n, alpha, jnp.array(1., dtype=jnp.float64))
        #     C_1 = jax.vmap(geg)(jnp.arange(_truncation_level))
        #     n = jnp.array(_truncation_level, dtype=jnp.float64)
        #     return alpha / (n + alpha) / C_1

        # return jax.lax.cond(max_num_eigvals is None, _from_lookup, _compute)

    def eigenvalues(
        self, param: Param, levels: Int[ArrayLike, "n"]
    ) -> Float[ArrayLike, "n"]:
        if self.name in param.constants and "eigenvalues" in param.constants[self.name]:
            eigval = param.constants[self.name]["eigenvalues"]
            levels = jnp.clip(levels, 0, len(eigval) - 1)
        else:
            assert isinstance(self, PolynomialDecay)  # helps mypy
            n = jnp.arange(self.truncation_level, dtype=jnp.float64)
            beta = param.params[self.name]["beta"]
            sphere_dim = param.constants["sphere"]["sphere_dim"]
            geg = param.constants["sphere"]["gegenbauer_lookup_table"]
            decay = (1 + n) ** (-beta)
            const_factor = self._compute_eigvals(
                sphere_dim, gegenbauer_lookup_table=geg
            )
            eigval = decay * const_factor / jnp.sum(decay)
            levels = jnp.clip(levels, 0, self.truncation_level - 1)
        return eigval[levels]

    def _scale_X(self, param: Param, X: Float[ArrayLike, "N D"]):
        weight_variances = param.params[self.name]["weight_variances"]
        if len(weight_variances.shape) == 2:
            return jnp.matmul(X, weight_variances)
        return X * jnp.sqrt(weight_variances)

    def to_sphere(self, param: Param, X: Float[ArrayLike, "N D"]):
        scaled_X = self._scale_X(param, X)
        bias_shape = scaled_X.shape[:-1] + (1,)
        b = param.params[self.name]["bias_variance"]
        bias = jnp.ones(bias_shape, dtype=jnp.float64) * jnp.sqrt(b)
        X_with_bias = jnp.concatenate([scaled_X, bias], axis=-1)
        r = jnp.sqrt(jnp.sum(jnp.square(X_with_bias), axis=-1))
        return X_with_bias / r[..., None], r

    # def shape_function(self, param: Param, x: ArrayLike) -> ArrayLike:
    def shape_function(self, param: Param, x: Array) -> Array:
        raise NotImplementedError

    def _squash(self, x: Array):
        eps = 1e-15
        return jnp.array((1 - eps) * x, dtype=jnp.float64)

    def kappa(self, u: Array, order: int):
        if order == 0:
            return (jnp.pi - jnp.arccos(u)) / jnp.pi
        elif order == 1:
            return (
                u * (jnp.pi - jnp.arccos(u)) + jnp.sqrt(1.0 - jnp.square(u))
            ) / jnp.pi
        elif order == 2:
            return (
                (1.0 + 2.0 * jnp.square(u)) / 3 * (jnp.pi - jnp.arccos(u))
                + jnp.sqrt(1.0 - jnp.square(u))
            ) / jnp.pi
        else:
            raise NotImplementedError()

    def K2(
        self,
        param: Param,
        X: Float[Array, "N D"],
        X2: Optional[Float[Array, "M D"]] = None,
    ) -> Float[Array, "N M"]:
        X_sphere, rad1 = self.to_sphere(param, X)
        if X2 is None:
            X_sphere2 = X_sphere
            rad2 = rad1
            K = jax.vmap(lambda x: jax.vmap(lambda y: jnp.dot(x, y))(X_sphere2))(
                X_sphere
            )
            i, j = jnp.diag_indices(K.shape[-1])
            K = K.at[..., i, j].set(1.0)
        else:
            X_sphere2, rad2 = self.to_sphere(param, X2)
            K = jax.vmap(lambda x: jax.vmap(lambda y: jnp.dot(x, y))(X_sphere2))(
                X_sphere
            )

        K = self.shape_function(param, K)
        r = jax.vmap(lambda x: jax.vmap(lambda y: jnp.dot(x, y))(rad2))(rad1)
        variance = param.params[self.name]["variance"]
        return variance * K * (r**self.order)

    def K(
        self,
        param: Param,
        X: Float[Array, "N D"],
        X2: Optional[Float[Array, "M D"]] = None,
    ) -> Float[Array, "N M"]:
        X2 = X if X2 is None else X2

        def _k_func(x1, x2):
            x_sphere, rad1 = self.to_sphere(param, x1)
            x_sphere2, rad2 = self.to_sphere(param, x2)
            k = jnp.dot(x_sphere, x_sphere2)
            k = self.shape_function(param, k)
            r = rad1 * rad2
            variance = param.params[self.name]["variance"]
            return variance * k * (r**self.order)

        return jax.vmap(lambda x: jax.vmap(lambda y: _k_func(x, y))(X2))(X)

    def K_diag(self, param: Param, X: Float[Array, "N D"]) -> Float[Array, "N"]:
        _, rad = self.to_sphere(param, X)
        variance = param.params[self.name]["variance"]
        return variance * rad ** (2 * self.order)

    def __add__(self, other: "Spherical") -> "Spherical":
        return Sum(kernels=[self, other])  # type: ignore

    def __mul__(self, other: "Spherical") -> "Spherical":
        return Product(kernels=[self, other])  # type: ignore


class Combination(Spherical):
    kernels: Sequence[Spherical] = field(default_factory=list, pytree_node=False)
    _reduce_fn: Optional[Callable] = field(default=None, pytree_node=False)
    name: str = field(default="Combination", pytree_node=False)

    def __post_init__(self):
        if not all(isinstance(k, Spherical) for k in self.kernels):
            TypeError("We can only combine Spherical kernels.")

        kernels: Sequence[Spherical] = []
        # cursors = {self.__class__.__name__: 0}
        for k in self.kernels:
            # prefix = k.__class__.__name__
            # suffix = cursors.get(prefix, 0)
            # k_name = f"{prefix}_{suffix}"
            # k = k.replace(name=k_name)
            # cursors[prefix] = suffix + 1

            if isinstance(k, self.__class__):
                kernels.extend(k.kernels)
            else:
                kernels.append(k)

        object.__setattr__(self, "kernels", kernels)
        # object.__setattr__(self, "name", f"{self.__class__.__name__}_0")


class Sum(Combination):
    _reduce_fn: Optional[Callable] = field(default=jnp.sum, pytree_node=False)
    name: str = field(default="Sum", pytree_node=False)


class Product(Combination):
    _reduce_fn: Optional[Callable] = field(default=jnp.prod, pytree_node=False)
    name: str = field(default="Product", pytree_node=False)


class ArcCosine(Spherical):
    depth: int = field(default=1, pytree_node=False)
    name: str = field(default="ArcCosine", pytree_node=False)

    def __post_init__(self):
        if self.order not in {0, 1, 2}:
            raise ValueError("Requested order is not implemented.")

    def shape_function(self, param: Param, x: Array) -> Array:
        # x = self._squash(x)

        def step(carry: Array, dummy: Optional[Array] = None) -> Tuple[Array, Array]:
            y = self.kappa(carry, self.order)
            return y, y

        y, _ = jax.lax.scan(step, x, xs=None, length=self.depth)
        return y


class NTK(Spherical):
    order: int = field(init=False, pytree_node=False)
    depth: int = field(default=1, pytree_node=False)
    name: str = field(default="NTK", pytree_node=False)

    def __post_init__(self):
        object.__setattr__(self, "order", 1)

    def shape_function(self, param: Param, x: Array) -> Array:
        x = self._squash(x)

        def step(
            carry: Tuple[Array, Array], dummy: Optional[Array] = None
        ) -> Tuple[Tuple[Array, Array], Array]:
            x, y = carry
            x, y = self.kappa(x, 1), y * self.kappa(x, 0) + self.kappa(x, 1)
            carry = x, y
            return carry, y

        _, y = jax.lax.scan(step, (x, x), xs=None, length=self.depth)
        return y[-1] / (self.depth + 1)


class PolynomialDecay(Spherical):
    truncation_level: int = field(default=10, pytree_node=False)

    order: int = field(init=False, pytree_node=False)
    name: str = field(default="PolyDecay", pytree_node=False)

    # def _compute_eigvals(
    #         self,
    #         sphere_dim: int,
    #         max_num_eigvals: Optional[int] = None,
    #         gegenbauer_lookup_table: Optional[GegenbauerLookupTable] = None,
    # ):
    #     alpha = (sphere_dim - 2.) / 2.
    #     gegenbauer = lambda n: self.gegenbauer(n, alpha, jnp.array(1., dtype=jnp.float64))
    #     C_1 = jax.vmap(gegenbauer)(jnp.arange(self.truncation_level))
    #     n = jnp.array(self.truncation_level, dtype=jnp.float64)
    #     return alpha / (n + alpha) / C_1
    def _compute_eigvals(
        self,
        sphere_dim: int,
        *,
        max_num_eigvals: Optional[int] = None,
        gegenbauer_lookup_table: Optional[GegenbauerLookupTable] = None,
    ):
        # checkify.check(
        #     gegenbauer_lookup_table is None,
        #     "The `gegenbauer_lookup_table` should be provided."
        # )
        if not gegenbauer_lookup_table:
            raise ValueError("Lookup table should be provided with `PolyDecay` kernel.")

        # _truncation_level = self.__dict__.get("truncation_level", max_num_eigvals)
        # gegenbauer_lookup_table = gegenbauer_lookup_table or GegenbauerLookupTable()

        alpha = (sphere_dim - 2.0) / 2.0
        geg = lambda n: gegenbauer_lookup_table(
            n, alpha, jnp.array(1.0, dtype=jnp.float64)
        )
        C_1 = jax.vmap(geg)(jnp.arange(self.truncation_level))
        n = jnp.arange(self.truncation_level, dtype=jnp.float64)
        return alpha / (n + alpha) / C_1

    def shape_function(self, param: Param, x: Array) -> Array:
        # x = self._squash(x)

        alpha = param.constants["sphere"]["alpha"]
        # (param.constants[self.name]["sphere_dim"]- 2.) / 2.
        gegenbauer_lookup = param.constants["sphere"]["gegenbauer_lookup_table"]
        levels = jnp.arange(self.truncation_level)

        const_factor = self.eigenvalues(param, levels)
        eigvals = const_factor * (levels.astype(jnp.float64) + alpha) / alpha
        C_n_x = jax.vmap(lambda n: gegenbauer_lookup(n, alpha, x))(levels)
        return jnp.sum(jax.vmap(lambda n, e: n * e)(C_n_x, eigvals), 0)
