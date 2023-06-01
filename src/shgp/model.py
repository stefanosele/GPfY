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

from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import Array, random
from jaxtyping import Float
from spherical import Spherical

from shgp.likelihoods import Likelihood
from shgp.param import Param
from shgp.spherical import Spherical
from shgp.spherical_harmonics import SphericalHarmonics
from shgp.utils import dataclass, field
from shgp.variational import VariationalDistribution

PRNG = Union[random.PRNGKeyArray, jax.Array]


@dataclass
class GP:
    kernel: Spherical = field(pytree_node=False)
    conditional_fun: Tuple[Callable, Callable, Callable] = field(
        default_factory=tuple, pytree_node=False
    )
    name: str = field(default="GP", pytree_node=False)

    def __init_subclass__(cls):
        dataclass(cls)  # pytype: disable=wrong-arg-types

    def init(
        self,
        key: PRNG,
        input_dim: int,
        num_independent_processes: int = 1,
        projection_dim: Optional[int] = None,
        likelihood: Optional[Likelihood] = None,
        sh_features: Optional[SphericalHarmonics] = None,
        variational_dist: Optional[VariationalDistribution] = None,
    ) -> Param:
        key, subkey = jax.random.split(key)
        param = self.kernel.init(subkey, input_dim, projection_dim)
        param = likelihood.init(param) if likelihood else param
        if variational_dist and sh_features:
            param = sh_features.init(key, projection_dim or input_dim, param)
            param = variational_dist.init(
                sh_features.num_inducing(param), num_independent_processes, param
            )

        return param

    def conditional(
        self,
        sh: SphericalHarmonics,
        q: VariationalDistribution,
    ) -> "GP":
        proj_fun, cond_cov, cond_var = sh.conditional_fun(self.kernel)
        cond_mean_fun = lambda param, x: q.project_mean(param, proj_fun(param, x))
        cond_cov_fun = lambda param, x: cond_cov(param, x) + q.project_variance(
            param, proj_fun(param, x)
        )
        cond_var_fun = lambda param, x: cond_var(param, x)[
            ..., None
        ] + q.project_diag_variance(param, proj_fun(param, x))
        conditional_fun = (
            jax.jit(cond_mean_fun),
            jax.jit(cond_cov_fun),
            jax.jit(cond_var_fun),
        )
        return GP(self.kernel, conditional_fun)

    def mu(self, param: Param) -> Callable[[Float[Array, "N D"]], Float[Array, "N D"]]:
        if self.conditional_fun:
            return lambda x: self.conditional_fun[0](param, x)
        return lambda x: jnp.zeros_like(x, dtype=jnp.float64)

    def var(self, param: Param) -> Callable[[Float[Array, "N D"]], Float[Array, "N"]]:
        if self.conditional_fun:
            return lambda x: self.conditional_fun[2](param, x)
        return lambda x: self.kernel.K_diag(param, x)

    def cov(self, param: Param) -> Callable[[Float[Array, "N D"]], Float[Array, "N N"]]:
        if self.conditional_fun:
            return lambda x: self.conditional_fun[1](param, x)
        return lambda x: self.kernel.K(param, x)

    def predict_diag(self, param: Param, X: Float[Array, "N D"]):
        return self.mu(param)(X), self.var(param)(X)

    def predict(self, param: Param, X: Float[Array, "N D"]):
        return self.mu(param)(X), self.cov(param)(X)


@jax.jit
def elbo(
    param: Param,
    m: GP,
    q: VariationalDistribution,
    lik: Likelihood,
    train_data: Tuple,
    dataset_size: int = -1,
):
    X, Y = train_data
    fmu, fvar = m.predict_diag(param, X)
    var_exp = lik.variational_expectations(param, fmu, fvar)(Y)
    scale = jax.lax.cond(
        dataset_size > 0, lambda: dataset_size, lambda: jnp.shape(X)[0]
    )
    KL = q.prior_KL(param)
    return scale * jnp.mean(var_exp, -1) - KL