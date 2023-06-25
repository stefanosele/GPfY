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

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxtyping import Array, Float

from gpfy.param import Param, positive
from gpfy.quadrature import gauss_hermite_quadrature
from gpfy.typing import BijectorDict, ConstantDict, TrainableDict, VariableDict
from gpfy.utils import dataclass, field


@dataclass
class Likelihood:
    """
    The base class for a Likelihood object.

    We rely on tensorflow_probability for the heavy work.

    Currently only a Gaussian and a Bernoulli distribution are supported.
    Each derived class should implement the following abstract methods:
        * conditional_distribution(param, ...)
        * variational_expectations(param, ...)

    Attributes:
        name: The name for the object. Defautls to the class name.
    """

    name: str = field(default="Likelihood", pytree_node=False)

    def __init_subclass__(cls):
        """Make sure inherting classes are dataclasses."""
        dataclass(cls)  # pytype: disable=wrong-arg-types

    def init(self, param: Optional[Param] = None) -> Param:
        """
        Iinitialise the parameters, if any, of the likelihood.

        Args:
            param: An already initialised collection of parameters from another object, i.e., a
                kernel or spherical harmonics. If it is provided we extend the `Param` with the
                additional collection. Otherwise we return a new `Param`. Defaults to None.

        Returns:
            The `Param` object with all variables.
        """
        # basic set-up
        return param or Param()

    def log_prob(
        self, param: Param, F: Float[Array, "N D"], Y: Float[Array, "N D"]
    ) -> Float[Array, " N"]:
        """
        The log probability density log p(Y|F), where we sum over the independent output dimensions.

        Args:
            param: A `Param` initialised with the likelihood.
            F: The ranodm vector, which is usually a sample obtained from the GP.
            Y: The outputs we want to match via the distribution.

        Returns:
            The log probability density.
        """
        return jnp.sum(self.conditional_distribution(param, F).log_prob(Y), -1)

    def conditional_distribution(
        self, param: Param, F: Float[Array, "N D"]
    ) -> tfp.distributions.Distribution:
        """
        The distribution of the outputs conditioned on a latent function evaluation, i.e., p(Y|F).

        Args:
            param: A `Param` initialised with the likelihood.
            F: The ranodm vector, which is usually a sample obtained from the GP.

        Returns:
            The conditioanl distribution.
        """
        raise NotImplementedError()

    def variational_expectations(
        self, param: Param, Fmu: Float[Array, "N D"], Fvar: Float[Array, "N D"]
    ) -> Callable[[Float[Array, "N D"]], Float[Array, " N"]]:
        """
        Define the expected log density, given a Gaussian distribution for the function values.

        In variational inference we have a variational posterior over functions
            q(f) = 𝒩(Fmu, Fvar).

        Then for a liklihood

            p(y|f)

        we compute

           ∫ log(p(y=Y|f)) q(f) df.

        Returns:
            The expected log density function that can be applied on some data `Y`.
        """
        raise NotImplementedError()


class Gaussian(Likelihood):
    """
    The Gaussian likelihood parameterised by the noise variance.

    Attributes:
        name: The name for the object. Defautls to the class name.
    """

    name: str = field(default="Gaussian", pytree_node=False)

    def init(self, param: Optional[Param] = None) -> Param:
        """
        Initialise the parameters of the Gaussian likelihood and return a `Param` object.

        Args:
            param: An already initialised collection of parameters from another object, i.e., a
                kernel or spherical harmonics. If it is provided we extend the `Param` with the
                additional collection. Otherwise we return a new `Param`. Defaults to None.

        Returns:
            The `Param` object with all variables.
        """
        lik_params = {"variance": jnp.array(1.0, dtype=jnp.float64)}
        lik_bijectors = {"variance": positive()}

        # initialise the collection
        collection = "likelihood"
        all_params: VariableDict = {}
        all_trainables: TrainableDict = {collection: {}}
        all_bijectors: BijectorDict = {}
        all_constants: ConstantDict = {}
        constrained = True

        # get the collections in case we have a provided Param
        if param:
            all_params = param.params
            all_bijectors = param._bijectors
            all_trainables = param._trainables
            all_constants = param.constants
            constrained = param._constrained

        # extend the collection and return a new param
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
        """
        The Normal distribution p(Y|F) = 𝒩(F, σ²).

        Args:
            param: A `Param` initialised with the likelihood.
            F: The ranodm vector, which is usually a sample obtained from the GP.

        Returns:
            The conditioanl distribution.
        """
        variance = param.params["likelihood"][self.name]["variance"]
        return tfp.distributions.Normal(loc=F.astype(jnp.float64), scale=jnp.sqrt(variance))

    def variational_expectations(
        self, param: Param, Fmu: Float[Array, "N D"], Fvar: Float[Array, "N D"]
    ) -> Callable[[Float[Array, "N D"]], Float[Array, " N"]]:
        """
        Define the expected log density, given a Gaussian distribution for the function values.

        In variational inference we have a variational posterior over functions
            q(f) = 𝒩(Fmu, Fvar).

        Then for a liklihood

            p(y|f)

        we compute

           ∫ log(p(y=Y|f)) q(f) df.

        NOTE: For the Guassian likelihood the variational expectations can be computed analytically.

        Returns:
            The expected log density function that can be applied on some data `Y`.
        """
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
    """
    The Bernoulli likelihood for classification.

    It is parameterised by a latent function (the GP), which is mapped via the inverse probit to the
    probability space [0, 1].

    Attributes:
        name: The name for the object. Defautls to the class name.
    """

    name: str = field(default="Bernoulli", pytree_node=False)

    def conditional_distribution(
        self, param: Param, F: Float[Array, "N D"]
    ) -> tfp.distributions.Distribution:
        """
        The distribution of the outputs conditioned on a latent function evaluation, i.e., p(Y|F).

        NOTE: The inverse probit is used as a link function to map the continuous latent function
        to the probability space [0, 1].

        Args:
            param: A `Param` initialised with the likelihood.
            F: The ranodm vector, which is usually a sample obtained from the GP.

        Returns:
            The conditioanl distribution.
        """
        return tfp.distributions.Bernoulli(probs=inv_probit(F))

    def variational_expectations(
        self, param: Param, Fmu: Float[Array, "N D"], Fvar: Float[Array, "N D"]
    ) -> Callable[[Float[Array, "N D"]], Float[Array, " N"]]:
        """
        Define the expected log density, given a Gaussian distribution for the function values.

        In variational inference we have a variational posterior over functions
            q(f) = 𝒩(Fmu, Fvar).

        Then for a liklihood

            p(y|f)

        we compute

           ∫ log(p(y=Y|f)) q(f) df.

        NOTE: For the Bernoulli likelihood the variational expectations are estimated via
        Gauss-Hermite quadrature integration.

        Returns:
            The expected log density function that can be applied on some data `Y`.
        """
        log_prob = lambda f, y: self.log_prob(param, f, y)
        return gauss_hermite_quadrature(log_prob, Fmu, jnp.sqrt(Fvar))


def inv_probit(x: Float[Array, "N D"]) -> Float[Array, "N D"]:
    """
    Compute the inverse probit function.

    Args:
        x: A vector of values.

    Returns:
        The inverse probit of the input vector.
    """
    jitter = 1e-3  # To ensure output is in interval (0, 1).
    return 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0))) * (1 - 2 * jitter) + jitter
