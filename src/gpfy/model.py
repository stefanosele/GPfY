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

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from gpfy.likelihoods import Likelihood
from gpfy.param import Param
from gpfy.scan import vscan
from gpfy.spherical import Spherical
from gpfy.spherical_harmonics import SphericalHarmonics
from gpfy.training import TrainState
from gpfy.typing import PRNG, TrainStepFn
from gpfy.utils import dataclass, field
from gpfy.variational import VariationalDistribution


@dataclass
class GP:
    """
    A Gaussian process model parameterised by a mean and a covariance function.

    Currently, mean funciton is not implemented so we assume all models to have zero mean.

    Atrirbutes:
        kernel: A spherical kernel acting as the covariance function of the GP.
        conditional_fn: A tuple of four functions that define how to do conditioning for the GP via:
            1. The projection function that computes the projection operation Kₘₘ⁻¹Kₘₙ.
            2. The conditional mean that computes the predictive mean given the projection.
            3. The conditional variance that computes the diagonal variances given the projection.
            4. The conditional covariance that computes the covariance given the porjection.

            It defaults to an empty tuple, which implies predicting from the prior GP.
        name: The name for the object. Defautls to the class name.

    Returns:
        A GP model.
    """

    kernel: Spherical = field(pytree_node=False)
    conditional_fn: Tuple[Callable, Callable, Callable, Callable] = field(
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
        """
        Initialise the parameters of a sparse sphercial GP.

        Args:
            key: A random key for initialising the kernel and the spherical harmonics.
            input_dim: The input dimension. We then append an extra dimension for bias.
            num_independent_processes: The number of independent processes we need to learn.
                Used for independently modelling multiple outputs. Defaults to 1.
            projection_dim: If specified it denotes a projection of the `input_dim` to
                `projection_dim`. Defaults to None.
            likelihood: The likelihood to model the data. Defaults to None.
            sh_features: The spherical harmonics features for sparse inference. Defaults to None.
            variational_dist: The variational distribution to approximate the posterior process.
                Defaults to None.

        Returns:
            The `Param` object with all variables.
        """
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
        """
        Condition the process on the specified pseudo observations, i.e., the inducing features.

        Args:
            sh: The spherical harmonic features.
            q: The variational distribution.

        Returns:
            A new GP model with populated `self.conditional_fn` to make predictions from.
        """
        proj_fun, cond_cov, cond_var = sh.conditional_fun(self.kernel)

        # cond_var = lambda param, x, proj: self.kernel.K_diag(param, x) - jnp.sum(
        #     jnp.square(proj), -2
        # )
        # cond_cov = lambda param, x, proj: self.kernel.K(param, x) - jnp.matmul(
        #     proj.swapaxes(-1, -2), proj
        # )

        # conditional mean function
        cond_mean_fun = lambda param, proj: q.project_mean(param, proj)

        # conditional covariance function
        cond_cov_fun = lambda param, x, proj: cond_cov(param, x, proj) + q.project_variance(
            param, proj
        )

        # conditional diagonal variance function
        cond_var_fun = lambda param, x, proj: cond_var(param, x, proj)[
            ..., None
        ] + q.project_diag_variance(param, proj)
        conditional_fn = (
            (proj_fun),
            (cond_mean_fun),
            (cond_cov_fun),
            (cond_var_fun),
        )
        return GP(self.kernel, conditional_fn)

    def mu(self, param: Param) -> Callable[[Float[Array, "N Din"]], Float[Array, "N Dout"]]:
        """
        The mean function of the GP.

        If the `self.conditional_fn` tuple is defined, then it returns the specified predictive
        mean function. Otherwise it returns the prior mean function.

        Args:
            param: A `Param` initialised with the model.

        Returns:
            The conditional mean function or the prior mean.
        """
        if self.conditional_fn:
            proj_fun, cond_mean_fun, _, __ = self.conditional_fn
            return lambda x: cond_mean_fun(param, proj_fun(param, x))
        return lambda x: jnp.zeros((x.shape[0], 1), dtype=jnp.float64)

    def var(self, param: Param) -> Callable[[Float[Array, "N Din"]], Float[Array, "N Dout"]]:
        """
        The diagonal variance function of the GP.

        If the `self.conditional_fn` tuple is defined, then it returns the specified diagonal
        predictive variance function. Otherwise it returns the prior variance.

        Args:
            param: A `Param` initialised with the model.

        Returns:
            The conditional variance function or the prior variance.
        """
        if self.conditional_fn:
            proj_fun, _, __, cond_var_fun = self.conditional_fn
            return lambda x: cond_var_fun(param, x, proj_fun(param, x))
        return lambda x: self.kernel.K_diag(param, x)

    def cov(self, param: Param) -> Callable[[Float[Array, "N D"]], Float[Array, "N N"]]:
        """
        The covariance function of the GP.

        If the `self.conditional_fn` tuple is defined, then it returns the specified predictive
        covariance function. Otherwise it returns the prior covariance.

        Args:
            param: A `Param` initialised with the model.

        Returns:
            The conditional covariance function or the prior covariance.
        """
        if self.conditional_fn:
            proj_fun, _, cond_cov_fun, __ = self.conditional_fn
            return lambda x: cond_cov_fun(param, x, proj_fun(param, x))
        return lambda x: self.kernel.K(param, x)

    def predict_diag(
        self, param: Param, X: Float[Array, "N Din"]
    ) -> Tuple[Float[Array, "N Dout"], Float[Array, "N Dout"]]:
        """
        Predict the mean and the diagonal variance of the GP, given some input `X`.

        If the `self.conditional_fn` tuple is defined, then it predicts from the equivalent
        conditional distribution. Otherwise it predicts from the prior process.

        NOTE: We explicitly duplicate code here instead of directly calling `self.mu` and aself.var`
        to avoid unecessary duplicate operations.

        Args:
            param: A `Param` initialised with the model.
            X: Input array.

        Returns:
            The predictive mean and diagonal variance of the (conditional) process.
        """
        if self.conditional_fn:
            proj_fun, cond_mean_fun, _, cond_var_fun = self.conditional_fn

            proj = proj_fun(param, X)
            mu = cond_mean_fun(param, proj)
            var = cond_var_fun(param, X, proj)
        else:
            mu = jnp.zeros((X.shape[0], 1), dtype=jnp.float64)
            var = self.kernel.K_diag(param, X)[..., None]
        return mu, var

    def predict(
        self, param: Param, X: Float[Array, "N Din"]
    ) -> Tuple[Float[Array, "N Dout"], Float[Array, "N N"]]:
        """
        Predict the mean and the covariance of the GP, given some input `X`.

        If the `self.conditional_fn` tuple is defined, then it predicts from the equivalent
        conditional distribution. Otherwise it predicts from the prior process.

        NOTE: We explicitly duplicate code here instead of directly calling `self.mu` and aself.var`
        to avoid unecessary duplicate operations.

        Args:
            param: A `Param` initialised with the model.
            X: Input array.

        Returns:
            The predictive mean and covariance of the (conditional) process.
        """
        if self.conditional_fn:
            proj_fun, cond_mean_fun, cond_cov_fun, _ = self.conditional_fn

            proj = proj_fun(param, X)
            mu = cond_mean_fun(param, proj)
            var = cond_cov_fun(param, X, proj)
        else:
            mu = jnp.zeros((X.shape[0], 1), dtype=jnp.float64)
            var = self.kernel.K(param, X)
        return mu, var

    def fit(
        self,
        param: Param,
        train_step: TrainStepFn,
        optimizer: optax.GradientTransformation,
        num_iters: int,
        progress_bar: bool = True,
    ):
        """
        Optimise the parameters to fit the GP model on the training data.

        Args:
            param: A `Param` initialised with the model.
            train_step: A training step function that is called inside a loop during optimisation.
            optimizer: The optimizer to use.
            num_iters: The number of iterations.
            progress_bar: Flag to indicate if we want to print a progress bar during optimisation.
                Defaults to True.

        Returns:
            A tuple of 3 objects containing the optimised `Param`, the final `TrainingState` and
            an `Array` containing the value of the negative `elbo` during the optimisation steps.
        """
        # Get the unconstrained params
        param_free = param.unconstrained()

        # Get the non-trainable params and set the optimizer to zero_grad
        frozen = jax.tree_map(lambda x: not (x), param_free._trainables)
        tx = optax.chain(optimizer, optax.masked(optax.set_to_zero(), frozen))

        # initialise a train state with the free parameters and an `apply_fn` that replaces the
        # VariableDict in the free `Param`.
        state = TrainState.create(
            apply_fn=lambda p: param_free.replace(params=p), params=param_free.params, tx=tx
        )

        scan: Callable = vscan if progress_bar else jax.lax.scan  # type: ignore
        state, loss_val = scan(train_step, state, None, num_iters)

        param_new = state.apply_fn(state.params).constrained()

        return param_new, state, loss_val
