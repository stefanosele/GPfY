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

from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import Array, random
from jaxtyping import Float

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
    conditional_fun: Tuple[Callable, Callable, Callable, Callable] = field(
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

        # cond_var = lambda param, x, proj: self.kernel.K_diag(param, x) - jnp.sum(
        #     jnp.square(proj), -2
        # )
        # cond_cov = lambda param, x, proj: self.kernel.K(param, x) - jnp.matmul(
        #     proj.swapaxes(-1, -2), proj
        # )

        cond_mean_fun = lambda param, proj: q.project_mean(param, proj)
        cond_cov_fun = lambda param, x, proj: cond_cov(param, x, proj) + q.project_variance(
            param, proj
        )
        cond_var_fun = lambda param, x, proj: cond_var(param, x, proj)[
            ..., None
        ] + q.project_diag_variance(param, proj)
        conditional_fun = (
            (proj_fun),
            (cond_mean_fun),
            (cond_cov_fun),
            (cond_var_fun),
        )
        return GP(self.kernel, conditional_fun)

    def mu(self, param: Param) -> Callable[[Float[Array, "N D"]], Float[Array, "N D"]]:
        if self.conditional_fun:
            proj_fun, cond_mean_fun, _, __ = self.conditional_fun
            return lambda x: cond_mean_fun(param, proj_fun(param, x))
        return lambda x: jnp.zeros_like(x, dtype=jnp.float64)

    def var(self, param: Param) -> Callable[[Float[Array, "N D"]], Float[Array, "N"]]:
        if self.conditional_fun:
            proj_fun, cond_mean_fun, _, cond_var_fun = self.conditional_fun
            return lambda x: cond_var_fun(param, x, proj_fun(param, x))
        return lambda x: self.kernel.K_diag(param, x)

    def cov(self, param: Param) -> Callable[[Float[Array, "N D"]], Float[Array, "N N"]]:
        if self.conditional_fun:
            proj_fun, cond_mean_fun, cond_cov_fun, _ = self.conditional_fun
            return lambda x: cond_cov_fun(param, x, proj_fun(param, x))
        return lambda x: self.kernel.K(param, x)

    def predict_diag(self, param: Param, X: Float[Array, "N D"]):
        if self.conditional_fun:
            proj_fun, cond_mean_fun, _, cond_var_fun = self.conditional_fun

            proj = proj_fun(param, X)
            mu = cond_mean_fun(param, proj)
            var = cond_var_fun(param, X, proj)
        else:
            mu = jnp.zeros_like(X, dtype=jnp.float64)
            var = self.kernel.K_diag(param, X)
        return mu, var

    def predict(self, param: Param, X: Float[Array, "N D"]):
        if self.conditional_fun:
            proj_fun, cond_mean_fun, cond_cov_fun, _ = self.conditional_fun

            proj = proj_fun(param, X)
            mu = cond_mean_fun(param, proj)
            var = cond_cov_fun(param, X, proj)
        else:
            mu = jnp.zeros_like(X, dtype=jnp.float64)
            var = self.kernel.K(param, X)
        return mu, var


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
    scale = jax.lax.cond(dataset_size > 0, lambda: dataset_size, lambda: jnp.shape(X)[0])
    KL = q.prior_KL(param)
    return scale * jnp.mean(var_exp, -1) - KL
