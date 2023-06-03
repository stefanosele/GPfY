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

from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import Array, random
from jax.tree_util import tree_leaves, tree_map
from jaxtyping import ArrayLike, Float, Int

from shgp.param import Param, identity
from shgp.spherical import Spherical
from shgp.utils import dataclass, field

ActiveDims = Union[slice, list]
PRNG = Union[random.PRNGKeyArray, jax.Array]


@dataclass
class VariationalDistribution:
    name: str = field(default="VariationalDistribution", pytree_node=False)

    def __init_subclass__(cls):
        dataclass(cls)  # pytype: disable=wrong-arg-types

    def _define_covariance_bijector(self) -> tfp.bijectors.Bijector:
        raise NotImplementedError()

    def init(
        self,
        num_inducing_features: int,
        num_independent_processes: int = 1,
        param: Optional[Param] = None,
    ) -> Param:
        inducing_features = tree_leaves(param.params.get("variational")) if param else None
        if inducing_features:
            if num_inducing_features != sum(tree_map(lambda x: x.shape[0], inducing_features)):
                raise ValueError("`param` object contains different number of inducing features.")

        collection = "variational"
        mu = jnp.zeros((num_inducing_features, num_independent_processes), dtype=jnp.float64)
        cov = jax.vmap(lambda _: jnp.eye(num_inducing_features, dtype=jnp.float64))(
            jnp.arange(num_independent_processes)
        )
        var_params = {"mu": mu, "Sigma": cov}
        var_bijectors = {
            "mu": identity(),
            "Sigma": self._define_covariance_bijector(),
        }

        all_params, all_trainables, all_bijectors, all_constants, constrained = (
            {collection: {}},
            {collection: {}},
            {collection: {}},
            {},
            True,
        )
        if param:
            all_params = param.params
            all_bijectors = param._bijectors
            all_trainables = param._trainables
            all_constants = param.constants
            constrained = param._constrained

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
        return param.params["variational"][self.name]["mu"]

    def cov_part(self, param: Param) -> Float[Array, "L M M"]:
        return param.params["variational"][self.name]["Sigma"]

    def project_mean(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "N L"]:
        raise NotImplementedError()

    def project_diag_variance(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "N L"]:
        raise NotImplementedError()

    def project_variance(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "L N N"]:
        raise NotImplementedError()

    def logdet(self, param: Param) -> Float[Array, "L M"]:
        raise NotImplementedError()

    def trace(self, param: Param) -> Float[Array, "L"]:
        raise NotImplementedError()

    def prior_KL(self, param: Param) -> Float[Array, ""]:
        # we are always in whitened case
        KL = -0.5 * jnp.array(jnp.size(self.mu(param)), jnp.float64)  # constant term
        KL -= 0.5 * jnp.sum(self.logdet(param))  # logdet term
        KL += 0.5 * jnp.sum(jnp.square(self.mu(param)))  # mahalanobis term
        KL += 0.5 * jnp.sum(self.trace(param))  # trace term

        return KL


class VariationalDistributionTriL(VariationalDistribution):
    name: str = field(default="VariationalDistributionTriL", pytree_node=False)

    def _define_covariance_bijector(self) -> tfp.bijectors.Bijector:
        return tfp.bijectors.FillTriangular()

    def logdet(self, param: Param) -> Float[Array, "L M"]:
        L = param.params["variational"][self.name]["Sigma"]
        return jnp.sum(jnp.log(jnp.square(jax.vmap(jnp.diag)(L))), axis=-1)

    def trace(self, param: Param) -> Float[Array, "L"]:
        L = param.params["variational"][self.name]["Sigma"]
        return jnp.sum(jnp.square(jax.vmap(jnp.diag)(L)), axis=[-1, -2])

    def project_mean(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "N L"]:
        mu = param.params["variational"][self.name]["mu"]
        return jnp.matmul(A.swapaxes(-1, -2), mu)

    def project_diag_variance(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "N L"]:
        L = param.params["variational"][self.name]["Sigma"]

        A = A[..., None, :, :]  # match any leading dims
        tmp = jnp.matmul(L.swapaxes(-1, -2), A)
        projected_diag_variance = jnp.sum(jnp.square(tmp), -2)
        return projected_diag_variance.swapaxes(-1, -2)

    def project_variance(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "L N N"]:
        L = param.params["variational"][self.name]["Sigma"]

        A = A[..., None, :, :]  # match any leading dims
        tmp = jnp.matmul(L.swapaxes(-1, -2), A)
        projected_variance = jnp.matmul(tmp.swapaxes(-1, -2), tmp)
        return projected_variance


class VariationalDistributionFullCovariance(VariationalDistribution):
    name: str = field(default="VariationalDistributionFullCovariance", pytree_node=False)

    def _define_covariance_bijector(self) -> tfp.bijectors.Bijector:
        return identity()

    def logdet(self, param: Param) -> Float[Array, "L M"]:
        S = param.params["variational"][self.name]["Sigma"]
        L = jnp.linalg.cholesky(S)
        return jnp.sum(jnp.log(jnp.square(jax.vmap(jnp.diag)(L))), axis=-1)

    def trace(self, param: Param) -> Float[Array, "L"]:
        S = param.params["variational"][self.name]["Sigma"]
        M = S.shape[-1]
        i, j = jnp.diag_indices(M)
        return S.at[..., i, j].get().sum(-1)

    def project_mean(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "N L"]:
        mu = param.params["variational"][self.name]["mu"]
        return jnp.matmul(A.swapaxes(-1, -2), mu)

    def project_diag_variance(self, param: Param, A: Float[Array, "M N"]) -> Float[Array, "N L"]:
        S = param.params["variational"][self.name]["Sigma"]

        A = A[..., None, :, :]  # match any leading dims
        tmp = jnp.matmul(S, A)
        projected_diag_variance = jnp.sum(tmp * A, -2)
        return projected_diag_variance.swapaxes(-1, -2)

    def project_mean_and_variance(
        self, param: Param, A: Float[Array, "M N"]
    ) -> Float[Array, "L N N"]:
        S = param.params["variational"][self.name]["Sigma"]

        A = A[..., None, :, :]  # match any leading dims
        tmp = jnp.matmul(S, A)
        projected_variance = jnp.matmul(A.swapaxes(-1, -2), tmp)
        return projected_variance
