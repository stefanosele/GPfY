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

from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import Array, random
from jaxtyping import Float

from shgp.param import Param, positive
from shgp.quadrature import gauss_hermite_quadrature
from shgp.utils import dataclass, field

ActiveDims = Union[slice, list]
PRNG = Union[random.PRNGKeyArray, jax.Array]


@dataclass
class Likelihood:
    name: str = field(default="Likelihood", pytree_node=False)

    def __init_subclass__(cls):
        dataclass(cls)  # pytype: disable=wrong-arg-types

    def init(self, param: Optional[Param] = None) -> Param:
        return param or Param()

    def log_prob(
        self, param: Param, F: Float[Array, "N D"], Y: Float[Array, "N D"]
    ) -> Float[Array, "N"]:
        return jnp.sum(self.conditional_distribution(param, F).log_prob(Y), -1)

    def conditional_distribution(
        self, param: Param, F: Float[Array, "N D"]
    ) -> tfp.distributions.Distribution:
        raise NotImplementedError()

    def variational_expectations(
        self, param: Param, Fmu: Float[Array, "N D"], Fvar: Float[Array, "N D"]
    ) -> Callable[[Float[Array, "N D"]], Float[Array, "N"]]:
        raise NotImplementedError()


class Gaussian(Likelihood):
    name: str = field(default="Gaussian", pytree_node=False)

    def init(self, param: Optional[Param] = None) -> Param:
        lik_params = {"variance": jnp.array(1.0, dtype=jnp.float64)}
        lik_bijectors = {"variance": positive()}

        collection = "likelihood"
        all_params, all_trainables, all_bijectors, all_constants, constrained = (
            {},
            {collection: {}},
            {},
            {},
            True,
        )
        if param:
            all_params = param.params
            all_bijectors = param._bijectors
            all_trainables = param._trainables
            all_constants = param.constants
            constrained = param._constrained

        all_params[collection], all_bijectors[collection] = {}, {}
        all_params[collection][self.name] = lik_params
        all_bijectors[collection][self.name] = lik_bijectors
        return Param(
            params=all_params,
            _trainables=all_trainables,
            _bijectors=all_bijectors,
            constants=all_constants,
            _constrained=constrained,
        )

    def conditional_distribution(
        self, param: Param, F: Float[Array, "N D"]
    ) -> tfp.distributions.Distribution:
        variance = param.params["likelihood"][self.name]["variance"]
        return tfp.distributions.Normal(
            loc=F.astype(jnp.float64), scale=jnp.sqrt(variance)
        )

    def variational_expectations(
        self, param: Param, Fmu: Float[Array, "N D"], Fvar: Float[Array, "N D"]
    ) -> Callable[[Float[Array, "N D"]], Float[Array, "N"]]:
        variance = param.params["likelihood"][self.name]["variance"]

        def _var_exp(Y: Float[Array, "N D"]):
            return jnp.sum(
                -0.5 * jnp.log(2 * jnp.pi)
                - 0.5 * jnp.log(variance)
                - 0.5 * ((Y - Fmu) ** 2 + Fvar) / variance,
                axis=-1,
            )

        return _var_exp


class Bernoulli(Likelihood):
    name: str = field(default="Bernoulli", pytree_node=False)

    def conditional_distribution(
        self, param: Param, F: Float[Array, "N D"]
    ) -> tfp.distributions.Distribution:
        return tfp.distributions.Bernoulli(probs=inv_probit(F))

    def variational_expectations(
        self, param: Param, Fmu: Float[Array, "N D"], Fvar: Float[Array, "N D"]
    ) -> Callable[[Float[Array, "N D"]], Float[Array, "N"]]:
        log_prob = lambda f, y: self.log_prob(param, f, y)
        return gauss_hermite_quadrature(log_prob, Fmu, jnp.sqrt(Fvar))


def inv_probit(x: Float[Array, "N D"]) -> Float[Array, "N D"]:
    """Compute the inverse probit function.
    Args:
        x (Float[Array, "N 1"]): A vector of values.
    Returns:
        Float[Array, "N 1"]: The inverse probit of the input vector.
    """
    jitter = 1e-3  # To ensure output is in interval (0, 1).
    return (
        0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0))) * (1 - 2 * jitter)
        + jitter
    )
