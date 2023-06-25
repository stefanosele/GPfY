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

from typing import Optional

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax.tree_util import tree_leaves, tree_map
from jaxtyping import Array, Float

from gpfy.param import Param, identity
from gpfy.typing import BijectorDict, ConstantDict, TrainableDict, VariableDict
from gpfy.utils import dataclass, field


@dataclass
class VariationalDistribution:
    """
    Base dataclass for a variational distribution.

    Currently only Gaussian distributions are supported. Each derived class should implement the
    following abstract methods:
        * project_mean(param, ...)
        * project_diag_variance(param, ...)
        * project_variance(param, ...)
        * logdet(param, ...)
        * trace(param, ...)
        * _define_covariance_bijector()

    Attributes:
        name: The name for the object. Defautls to the class name.
    """

    name: str = field(default="VariationalDistribution", pytree_node=False)

    def __init_subclass__(cls):
        """Make sure inherting classes are dataclasses."""
        dataclass(cls)  # pytype: disable=wrong-arg-types

    def _define_covariance_bijector(self) -> tfp.bijectors.Bijector:
        """Defines the bijector to transform the covariance parameter."""
        raise NotImplementedError()

    def init(
        self,
        num_inducing_features: int,
        num_independent_processes: int = 1,
        param: Optional[Param] = None,
    ) -> Param:
        """
        Initialise the parameters of the variational distribution and return a `Param` object.

        Args:
            num_inducing_features: The number of inducing features to learn.
            num_independent_processes: The number of independent processes we need to learn.
                Used for independently modelling multiple outputs. Defaults to 1.
            param: An already initialised collection of parameters from another object, i.e., a
                kernel or spherical harmonics. If it is provided we extend the `Param` with the
                additional collection. Otherwise we return a new `Param`. Defaults to None.

        Raises:
            ValueError: Raises if the number of inducing features doesn't match the total number of
                harmonics in case we have already initialised a `Param` with `SphericalHarmonics`.

        Returns:
            The `Param` object with all variables.
        """
        inducing_features = tree_leaves(param.params.get("variational")) if param else None
        if inducing_features:
            if num_inducing_features != sum(tree_map(lambda x: x.shape[0], inducing_features)):
                raise ValueError("`param` object contains different number of inducing features.")

        # initialise the collection
        collection = "variational"
        mu = jnp.zeros((num_inducing_features, num_independent_processes), dtype=jnp.float64)
        cov = jax.vmap(lambda _: jnp.eye(num_inducing_features, dtype=jnp.float64))(
            jnp.arange(num_independent_processes)
        )
        var_params = {"mu": mu, "Sigma": cov}

        # Define the bijectors
        var_bijectors = {
            "mu": identity(),
            "Sigma": self._define_covariance_bijector(),
        }

        all_params: VariableDict = {collection: {}}
        all_trainables: TrainableDict = {collection: {}}
        all_bijectors: BijectorDict = {collection: {}}
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
        all_params[collection][self.name] = var_params
        all_bijectors[collection][self.name] = var_bijectors
        return Param(
            params=all_params,
            _trainables=all_trainables,
            _bijectors=all_bijectors,
            constants=all_constants,
            _constrained=constrained,
        )

    def mu(self, param: Param) -> Float[Array, "M L"]:
        """
        Helper function to easily access the mean of the distribution from the provided `Param`.

        Args:
            param: A `Param` initialised with the variational distribution.

        Returns:
            The mean of the distribution.
        """
        return param.params["variational"][self.name]["mu"]

    def cov_part(self, param: Param) -> Float[Array, "L M M"]:
        """
        Helper function to easily access the covariance (part) of the distribution from the `Param`.

        Args:
            param: A `Param` initialised with the variational distribution.

        Returns:
            The covariance parameterisation of the distribution.
        """
        return param.params["variational"][self.name]["Sigma"]

    def project_mean(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "N L"]:
        """
        Project the mean of the distribution given a projection matrix.

        In standard sparse GP inference the mean of the posterior process is given by::
            m(x) = KₙₘKₘₘ⁻¹ μ,
        where Kₘₘ⁻¹Kₘₙ is regarded to be the projection.

        Args:
            param: A `Param` initialised with the variational distribution.
            A: A projection, normally coming from the model/process side, that is applied to the
                mean of the distribution.

        Returns:
            The projected mean of the distribution.
        """
        raise NotImplementedError()

    def project_diag_variance(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "N L"]:
        """
        Project the diagonal variance of the distribution given a projection matrix.

        In standard sparse GP inference the covariance of the posterior process is given by::
            var(x) = Kₙₙ - KₙₘKₘₘ⁻¹Kₘₙ,
        where Kₘₘ⁻¹Kₘₙ is regarded to be the projection.

        Args:
            param: A `Param` initialised with the variational distribution.
            A: A projection, normally coming from the model/process side, that is applied to the
                covariance of the distribution.

        Returns:
            The diagonal of projected covariance of the distribution.
        """
        raise NotImplementedError()

    def project_variance(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "L N N"]:
        """
        Project the variance of the distribution given a projection matrix.

        In standard sparse GP inference the covariance of the posterior process is given by::
            var(x) = Kₙₙ - KₙₘKₘₘ⁻¹Kₘₙ,
        where Kₘₘ⁻¹Kₘₙ is regarded to be the projection.

        Args:
            param: A `Param` initialised with the variational distribution.
            A: A projection, normally coming from the model/process side, that is applied to the
                covariance of the distribution.

        Returns:
            The projected covariance of the distribution.
        """
        raise NotImplementedError()

    def logdet(self, param: Param) -> Float[Array, " L"]:
        """
        Evaluate the log determinant of the covariance of the distribution.

        Args:
            param: A `Param` initialised with the variational distribution.

        Returns:
            The log determinant of the covariance of the distribution.
        """
        raise NotImplementedError()

    def trace(self, param: Param) -> Float[Array, " L"]:
        """
        Evaluate the trace of the covariance of the distribution.

        Args:
            param: A `Param` initialised with the variational distribution.

        Returns:
            The trace of the covariance of the distribution.
        """
        raise NotImplementedError()

    def prior_KL(self, param: Param) -> Float[Array, ""]:
        """
        Evaluate the KL divergence `KL(q||p)` between the variational distribution and some prior.

        NOTE: We assume we are always working on the whitened prior case, so p(u) = 𝒩(0, I).

        Returns:
            The KL divergence `KL(q||p)`.
        """
        # we are always in whitened case
        KL = -0.5 * jnp.array(jnp.size(self.mu(param)), jnp.float64)  # constant term
        KL -= 0.5 * jnp.sum(self.logdet(param))  # logdet term
        KL += 0.5 * jnp.sum(jnp.square(self.mu(param)))  # mahalanobis term
        KL += 0.5 * jnp.sum(self.trace(param))  # trace term

        return KL


class VariationalDistributionTriL(VariationalDistribution):
    """
    Variational distribution with a lower triangular parameterisation for the half-covariance.

    The distribution has the form of::
        q(u) = 𝒩(μ, LLᵀ),

    where Σ = LLᵀ.

    Attributes:
        name: The name for the object. Defautls to the class name.
    """

    name: str = field(default="VariationalDistributionTriL", pytree_node=False)

    def _define_covariance_bijector(self) -> tfp.bijectors.Bijector:
        """Return the lower triangular bijector."""
        return tfp.bijectors.FillTriangular()

    def logdet(self, param: Param) -> Float[Array, " L"]:
        """
        Evaluate the log determinant of the covariance of the distribution.

        Args:
            param: A `Param` initialised with the variational distribution.

        Returns:
            The log determinant of the covariance of the distribution.
        """
        L = self.cov_part(param)
        return jnp.sum(jnp.log(jnp.square(jax.vmap(jnp.diag)(L))), axis=-1)

    def trace(self, param: Param) -> Float[Array, " L"]:
        """
        Evaluate the trace of the covariance of the distribution.

        Args:
            param: A `Param` initialised with the variational distribution.

        Returns:
            The trace of the covariance of the distribution.
        """
        L = self.cov_part(param)
        return jnp.sum(jnp.square(L), axis=[-1, -2])

    def project_mean(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "N L"]:
        """
        Project the mean of the distribution given a projection matrix.

        In standard sparse GP inference the mean of the posterior process is given by::
            m(x) = KₙₘKₘₘ⁻¹ μ,
        where Kₘₘ⁻¹Kₘₙ is regarded to be the projection.

        Args:
            param: A `Param` initialised with the variational distribution.
            A: A projection, normally coming from the model/process side, that is applied to the
                mean of the distribution.

        Returns:
            The projected mean of the distribution.
        """
        mu = self.mu(param)
        return jnp.matmul(A.swapaxes(-1, -2), mu)

    def project_diag_variance(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "N L"]:
        """
        Project the diagonal variance of the distribution given a projection matrix.

        In standard sparse GP inference the covariance of the posterior process is given by::
            var(x) = Kₙₙ - KₙₘKₘₘ⁻¹Kₘₙ,
        where Kₘₘ⁻¹Kₘₙ is regarded to be the projection.

        Args:
            param: A `Param` initialised with the variational distribution.
            A: A projection, normally coming from the model/process side, that is applied to the
                covariance of the distribution.

        Returns:
            The diagonal of projected covariance of the distribution.
        """
        L = self.cov_part(param)

        A = A[..., None, :, :]  # match any leading dims
        tmp = jnp.matmul(L.swapaxes(-1, -2), A)  # [L, M, N]
        projected_diag_variance = jnp.sum(jnp.square(tmp), -2)  # [L, N]
        return projected_diag_variance.swapaxes(-1, -2)

    def project_variance(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "L N N"]:
        """
        Project the variance of the distribution given a projection matrix.

        In standard sparse GP inference the covariance of the posterior process is given by::
            var(x) = Kₙₙ - KₙₘKₘₘ⁻¹Kₘₙ,
        where Kₘₘ⁻¹Kₘₙ is regarded to be the projection.

        Args:
            param: A `Param` initialised with the variational distribution.
            A: A projection, normally coming from the model/process side, that is applied to the
                covariance of the distribution.

        Returns:
            The projected covariance of the distribution.
        """
        L = self.cov_part(param)

        A = A[..., None, :, :]  # match any leading dims
        tmp = jnp.matmul(L.swapaxes(-1, -2), A)
        projected_variance = jnp.matmul(tmp.swapaxes(-1, -2), tmp)
        return projected_variance


class VariationalDistributionFullCovariance(VariationalDistribution):
    """
    Variational distribution with a full covariance parameterisation.

    The distribution has the form of::
        q(u) = 𝒩(μ, Σ).

    NOTE: There is no guarantee that the covariance will remain positive-definite if we use it in an
    optimisation loop.

    Attributes:
        name: The name for the object. Defautls to the class name.
    """

    name: str = field(default="VariationalDistributionFullCovariance", pytree_node=False)

    def _define_covariance_bijector(self) -> tfp.bijectors.Bijector:
        """Return the identity bijector."""
        return identity()

    def logdet(self, param: Param) -> Float[Array, " L"]:
        """
        Evaluate the log determinant of the covariance of the distribution.

        Args:
            param: A `Param` initialised with the variational distribution.

        Returns:
            The log determinant of the covariance of the distribution.
        """
        S = self.cov_part(param)
        L = jnp.linalg.cholesky(S)
        return jnp.sum(jnp.log(jnp.square(jax.vmap(jnp.diag)(L))), axis=-1)

    def trace(self, param: Param) -> Float[Array, " L"]:
        """
        Evaluate the trace of the covariance of the distribution.

        Args:
            param: A `Param` initialised with the variational distribution.

        Returns:
            The trace of the covariance of the distribution.
        """
        S = self.cov_part(param)
        return jnp.trace(S, axis1=-2, axis2=-1)

    def project_mean(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "N L"]:
        """
        Project the mean of the distribution given a projection matrix.

        In standard sparse GP inference the mean of the posterior process is given by::
            m(x) = KₙₘKₘₘ⁻¹ μ,
        where Kₘₘ⁻¹Kₘₙ is regarded to be the projection.

        Args:
            param: A `Param` initialised with the variational distribution.
            A: A projection, normally coming from the model/process side, that is applied to the
                mean of the distribution.

        Returns:
            The projected mean of the distribution.
        """
        mu = self.mu(param)
        return jnp.matmul(A.swapaxes(-1, -2), mu)

    def project_diag_variance(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "N L"]:
        """
        Project the diagonal variance of the distribution given a projection matrix.

        In standard sparse GP inference the covariance of the posterior process is given by::
            var(x) = Kₙₙ - KₙₘKₘₘ⁻¹Kₘₙ,
        where Kₘₘ⁻¹Kₘₙ is regarded to be the projection.

        Args:
            param: A `Param` initialised with the variational distribution.
            A: A projection, normally coming from the model/process side, that is applied to the
                covariance of the distribution.

        Returns:
            The diagonal of projected covariance of the distribution.
        """
        S = self.cov_part(param)

        A = A[..., None, :, :]  # match any leading dims
        tmp = jnp.matmul(S, A)
        projected_diag_variance = jnp.sum(tmp * A, -2)
        return projected_diag_variance.swapaxes(-1, -2)

    def project_variance(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "L N N"]:
        """
        Project the variance of the distribution given a projection matrix.

        In standard sparse GP inference the covariance of the posterior process is given by::
            var(x) = Kₙₙ - KₙₘKₘₘ⁻¹Kₘₙ,
        where Kₘₘ⁻¹Kₘₙ is regarded to be the projection.

        Args:
            param: A `Param` initialised with the variational distribution.
            A: A projection, normally coming from the model/process side, that is applied to the
                covariance of the distribution.

        Returns:
            The projected covariance of the distribution.
        """
        S = self.cov_part(param)

        A = A[..., None, :, :]  # match any leading dims
        tmp = jnp.matmul(S, A)
        projected_variance = jnp.matmul(A.swapaxes(-1, -2), tmp)
        return projected_variance
