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
from jax.tree_util import tree_leaves, tree_map
from jaxtyping import ArrayLike, Float, Int
from spherical import Spherical

from shgp.likelihoods import Likelihood
from shgp.param import Param, identity
from shgp.spherical import PolynomialDecay, Spherical
from shgp.spherical_harmonics import SphericalHarmonics
from shgp.utils import dataclass, field
from shgp.variational import VariationalDistribution, VariationalDistributionTriL

PRNG = Union[random.PRNGKeyArray, jax.Array]


@dataclass
class RandomProcess:
    kernel: Spherical = field(pytree_node=False)
    conditional_fun: Optional[Tuple[Callable, Callable, Callable]] = field(
        default=None, pytree_node=False
    )
    name: str = field(default="RandomProcess", pytree_node=False)

    def __init_subclass__(cls):
        dataclass(cls)  # pytype: disable=wrong-arg-types

    def conditional(
        self,
        param: Param,
        sh: SphericalHarmonics,
        q: VariationalDistribution,
    ) -> "RandomProcess":
        proj_fun, cond_cov, cond_var = sh.conditional_fun(param, self.kernel)
        cond_mean_fun = lambda x: q.project_mean(param, proj_fun(x))
        cond_cov_fun = lambda x: cond_cov(x) + q.project_variance(param, proj_fun(x))
        cond_var_fun = lambda x: cond_var(x)[..., None] + q.project_diag_variance(
            param, proj_fun(x)
        )
        conditional_fun = (cond_mean_fun, cond_cov_fun, cond_var_fun)
        return RandomProcess(self.kernel, conditional_fun)

    def mu(self, param: Param) -> Callable[[Float[Array, "N D"]], Float[Array, "N D"]]:
        if self.conditional_fun:
            return self.conditional_fun[0]
        return lambda x: jnp.zeros_like(x, dtype=jnp.float64)

    def var(self, param: Param) -> Callable[[Float[Array, "N D"]], Float[Array, "N"]]:
        if self.conditional_fun:
            return self.conditional_fun[2]
        return lambda x: self.kernel.K_diag(param, x)
        # return jax.vmap(lambda x: self.kernel.K_diag(param, x))

    def cov(self, param: Param) -> Callable[[Float[Array, "N D"]], Float[Array, "N N"]]:
        if self.conditional_fun:
            return self.conditional_fun[1]
        return lambda x: self.kernel.K(param, x)
        # def _cov(x: Float[Array, "N D"], y: Optional[Float[Array, "M D"]] = None):
        #     return self.kernel.K(param, x, y)

        # return _cov

    def predict_diag(self, param: Param, X: Float[Array, "N D"]):
        return self.mu(param)(X), self.var(param)(X)

    def predict(self, param: Param, X: Float[Array, "N D"]):
        return self.mu(param)(X), self.cov(param)(X)
