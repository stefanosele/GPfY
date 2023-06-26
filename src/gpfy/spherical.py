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

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import random
from jaxtyping import Array, Float, Int

from gpfy.gegenbauer import GegenbauerLookupTable
from gpfy.harmonics.utils import funk_hecke_lambda
from gpfy.param import Param
from gpfy.typing import PRNG, ActiveDims, ConstantDict
from gpfy.utils import dataclass, field


def _slice_or_list(value: Optional[ActiveDims] = None) -> Union[ActiveDims, Array]:
    """
    Retrun a slice for indexing dimensions of inputs.

    Args:
        value: The slice to the dimensions we want to index. Defaults to None.

    Returns:
        The slice.
    """
    if value is None:
        return slice(None, None, None)
    if isinstance(value, slice):
        return value
    return jnp.array(value, dtype=int)


@dataclass
class Spherical:
    """
    Abstract base class for spherical kernels that mimic the behaviour of neural networks.

    Implementations of `ArcCosine`, `NTK` and a custom `PolynomialDecay` kernel of continuous depth.

    Key reference is
    ::
        @article{cho2009kernel,
            title={Kernel Methods for Deep Learning},
            author={Cho, Youngmin and Saul, Lawrence},
            journal={Advances in Neural Information Processing Systems},
            volume={22},
            year={2009}
        }

    NOTE: All classess inherting from `Spherical` need to implement the `shape_function(...)`.

    Attributes:
        order: The order of the spherical kernel that specifies the activation function of the
            equivalent neural network. The function is a monmial of the specified order.
            Defaults to 1.
        ard: Flag to indicate if we want to model separate weights per input dimension.
            Defaults to True.
        active_dims: Optional slice to specify the active dimensions for the kernel.
            Currently, not supported.
        name: The name for the object. Defautls to `"Spherical"`.
    """

    order: int = field(default=1, pytree_node=False)
    ard: bool = field(default=True, pytree_node=False)
    active_dims: Optional[ActiveDims] = field(default_factory=_slice_or_list, pytree_node=False)
    name: str = field(default="Spherical", pytree_node=False)

    def __init_subclass__(cls):
        """Make sure inherting classes are dataclasses."""
        dataclass(cls)  # pytype: disable=wrong-arg-types

    def init(
        self,
        key: PRNG,
        input_dim: int,
        projection_dim: Optional[int] = None,
        max_num_eigvals: Optional[int] = None,
    ) -> Param:
        """
        Initialise the parameters of the kernel and return a `Param` object.

        Args:
            key: A random key for initialising the weights.
            input_dim: The input dimension. We then append an extra dimension for bias.
            projection_dim: If specified it denotes a projection of the `input_dim` to
                `projection_dim`. Defaults to None.
            max_num_eigvals: If specified it denotes the higher order of frequency up to which we
                compute the eigenvalues. Defaults to None.

        Returns:
            The `Param` object with all variables.
        """
        bias_variance = jnp.array(1.0, dtype=jnp.float64)
        variance = jnp.array(1.0, dtype=jnp.float64)
        bijectors = {}
        if not projection_dim:
            if self.ard:
                weight_variances = jnp.ones(input_dim, dtype=jnp.float64)
            else:
                weight_variances = jnp.array(1.0, dtype=jnp.float64)
        else:
            weight_variances = random.normal(key, (input_dim, projection_dim), dtype=jnp.float64)
            bijectors = {"weight_variances": tfp.bijectors.Identity()}

        params = {
            "weight_variances": weight_variances,
            "bias_variance": bias_variance,
            "variance": variance,
        }
        # set the collection of the parameters to the name of the kernel.
        collection = self.name
        # Compute constants regarding the sphere and the spectral properties of the kernel.
        # Add an extra dimension for the bias.
        sphere_dim = projection_dim or input_dim
        dim = sphere_dim + 1
        alpha = (dim - 2.0) / 2.0
        constants: ConstantDict = {}
        constants["sphere"] = {"sphere_dim": sphere_dim, "alpha": alpha}
        constants[self.name] = {}

        if isinstance(self, PolynomialDecay):
            # parameter to control the decay
            params["beta"] = jnp.array(1.0, dtype=jnp.float64)
            gegenbauer_lookup = GegenbauerLookupTable(self.truncation_level, alpha)
            constants["sphere"]["gegenbauer_lookup_table"] = gegenbauer_lookup
        else:
            # Precompute the eigenvalues of the kernel
            eigenvalues = self._compute_eigvals(sphere_dim, max_num_eigvals=max_num_eigvals)
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
    ) -> Float[Array, " N"]:
        """
        Compute the eigenvalues of the spherical kernel.

        Args:
            sphere_dim: The dimensionality of the sphere.
            max_num_eigvals: The maximum number of eigenvalues to precompute. Defaults to None,
                which will compute eigenvalues up to the 20th frequency.
            gegenbauer_lookup_table: A precomputed lookup table to evaluate the Gegenbauer
                polynomial. Defaults to None.

        Returns:
            Array with the eigenvalues of the kernel.
        """
        max_num_eigvals = max_num_eigvals or 20
        # The dim of the inputs
        dim = sphere_dim + 1
        # shape function expects a param input arg, but it is not required for ArcCosine and NTK.
        dummy_param = Param()
        part_shape_function = lambda x: self.shape_function(dummy_param, x=x)
        part_funk_hecke = lambda n: funk_hecke_lambda(part_shape_function, n, dim)
        return jax.vmap(part_funk_hecke)(jnp.arange(max_num_eigvals, dtype=jnp.int32))

    def eigenvalues(self, param: Param, levels: Int[Array, " N"]) -> Float[Array, " N"]:
        """
        Get the eigenvalues from the provided `param` object, or return 0s if not precomputed.

        Args:
            param: A `Param` initialised with the kernel.
            levels: An array specifying up to which order de we need eigenvalues.

        Returns:
            The eigenvalues.
        """
        zeros = jnp.zeros((len(levels),), dtype=jnp.float64)
        eigval = param.constants.get(self.name, {}).get("eigenvalues", zeros)
        return eigval[levels]

        # if self.name in param.constants and "eigenvalues" in param.constants[self.name]:
        #     # The kernel is initalised and we have already precomputed the eigenvalues
        #     eigval = param.constants[self.name]["eigenvalues"]
        #     levels = jnp.clip(levels, 0, len(eigval) - 1)
        # else:
        #     # we have a PolynomailDecay kernel and we need to compute the eigenvalues on the spot.
        #     assert isinstance(self, PolynomialDecay)  # helps mypy
        #     n = jnp.arange(self.truncation_level, dtype=jnp.float64)
        #     beta = param.params[self.name]["beta"]
        #     sphere_dim = param.constants["sphere"]["sphere_dim"]
        #     geg = param.constants["sphere"]["gegenbauer_lookup_table"]
        #     decay = (1 + n) ** (-beta)
        #     const_factor = self._compute_eigvals(sphere_dim, gegenbauer_lookup_table=geg)
        #     eigval = decay * const_factor / jnp.sum(decay)
        #     levels = jnp.clip(levels, 0, self.truncation_level - 1)
        # return eigval[levels]

    def _scale_X(self, param: Param, X: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """
        Scale the input X by the weights.

        Args:
            param: The `Param` initialised with the kernel.
            X: Input Array already projected to the unit sphere.

        Returns:
            The scaled input.
        """
        weight_variances = param.params[self.name]["weight_variances"]
        if len(weight_variances.shape) == 2:
            return jnp.matmul(X, weight_variances)
        return X * jnp.sqrt(weight_variances)

    def to_sphere(
        self, param: Param, X: Float[Array, "N D"]
    ) -> Tuple[Float[Array, "N DSphere"], Float[Array, " N"]]:
        """
        Normalise the input X and project it on the unit sphere.

        Args:
            param: The `Param` initialised with the kernel.
            X: Input Array.

        Returns:
            A tuple containing the normalised input and the norm (radius) of the vector.
        """
        scaled_X = self._scale_X(param, X)
        bias_shape = scaled_X.shape[:-1] + (1,)
        b = param.params[self.name]["bias_variance"]
        bias = jnp.ones(bias_shape, dtype=jnp.float64) * jnp.sqrt(b)
        X_with_bias = jnp.concatenate([scaled_X, bias], axis=-1)
        r = jnp.sqrt(jnp.sum(jnp.square(X_with_bias), axis=-1))
        return X_with_bias / r[..., None], r

    def shape_function(self, param: Param, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """
        The shape function that determines the equivalent kernel to a deep network architecture.

        Args:
            param: The `Param` initialised with the kernel.
            x: Input Array.

        Raises:
            NotImplementedError: All derived classes need to implement this method.
        """
        raise NotImplementedError

    def _squash(self, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """
        Ensure that the normalised input is squashed between `[-1, 1]`.

        Due to numerical precision the input can be larger than 1, which causes NaNs when evaluating
        the Gegenbauer polynomial.

        Args:
            x: Input Array.

        Returns:
            The squashed input.
        """
        return jnp.clip(x, -1.0, 1.0)

    def kappa(self, u: Float[Array, " N"], order: int) -> Float[Array, " N"]:
        """
        Compute the analytic integral of the angular part for angles with cosine `u`.

        This is essentially the kernel evaluation.

        NOTE: We have only implemented the first 3 orders which correspond to:
            0 -> step function
            1 -> ReLU
            2 -> Similar to elu

        NOTE: It ignores any specified order higher than 2 and clips it back to 2.

        Args:
            u: The cosine of the angle between normalised vectors on the unit sphere.
            order: The order of the kernel which determines the equivalent activation function.

        Returns:
            The shape function evaluated at the specified angles.
        """

        def _zero_order():
            return (jnp.pi - jnp.arccos(u)) / jnp.pi

        def _first_order():
            return (u * (jnp.pi - jnp.arccos(u)) + jnp.sqrt(1.0 - jnp.square(u))) / jnp.pi

        def _second_order():
            return (
                (1.0 + 2.0 * jnp.square(u)) / 3 * (jnp.pi - jnp.arccos(u))
                + jnp.sqrt(1.0 - jnp.square(u))
            ) / jnp.pi

        return jax.lax.switch(order, (_zero_order, _first_order, _second_order))

    def K(
        self,
        param: Param,
        X: Float[Array, "N D"],
        X2: Optional[Float[Array, "M D"]] = None,
    ) -> Float[Array, "N M"]:
        """
        Evaluates the covariances element-wise between the all pairs in `X` and `X2`.

        Similar to `self.K2` but with broadcasting instead of vmap.
        NOTE: If `X2` is not specified, the method evaluates the covariances between pairs of `X`.

        Args:
            param: The `Param` initialised with the kernel.
            X: Input Array.
            X2: Input Array.

        Returns:
            The covariance between the specified intpus.
        """
        # project inputs
        X_sphere, rad1 = self.to_sphere(param, X)
        if X2 is None:  # we compute the covariance between pairs of X, so copy X2 <- X
            X_sphere2 = X_sphere
            rad2 = rad1
            K = jax.vmap(lambda x: jax.vmap(lambda y: jnp.dot(x, y))(X_sphere2))(X_sphere)
            i, j = jnp.diag_indices(K.shape[-1])
            K = K.at[..., i, j].set(1.0)  # make sure diagonal is 1.
        else:  # proceed as normal and compute covariance between X, X2
            # project X2 on the sphere
            X_sphere2, rad2 = self.to_sphere(param, X2)
            K = jax.vmap(lambda x: jax.vmap(lambda y: jnp.dot(x, y))(X_sphere2))(X_sphere)

        K = self.shape_function(param, K)
        r = jax.vmap(lambda x: jax.vmap(lambda y: jnp.dot(x, y))(rad2))(rad1)
        variance = param.params[self.name]["variance"]
        return variance * K * (r**self.order)

    def K2(
        self,
        param: Param,
        X: Float[Array, "N D"],
        X2: Optional[Float[Array, "M D"]] = None,
    ) -> Float[Array, "N M"]:
        """
        Evaluates the covariances element-wise between the all pairs in `X` and `X2`.

        NOTE: If `X2` is not specified, the method evaluates the covariances between pairs of `X`.

        Args:
            param: The `Param` initialised with the kernel.
            X: Input Array.
            X2: Input Array.

        Returns:
            The covariance between the specified intpus.
        """
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

    def K_diag(self, param: Param, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        """
        Evaluates the diagonal of the kernel, i.e., the variances of the input.

        NOTE: We don't need to evaluate the shape function, as the cosine for the diagonal elements
        is 1 and the shape function is normalised so that `κ(1) = 1`.

        Args:
            param: The `Param` initialised with the kernel.
            X: Input Array.

        Returns:
            The diagonal variances of specified intpus.
        """
        _, rad = self.to_sphere(param, X)
        variance = param.params[self.name]["variance"]
        return variance * rad ** (2 * self.order)

    # def __add__(self, other: "Spherical") -> "Spherical":
    #     return Sum(kernels=[self, other])  # type: ignore

    # def __mul__(self, other: "Spherical") -> "Spherical":
    #     return Product(kernels=[self, other])  # type: ignore


class ArcCosine(Spherical):
    """
    The ArcCosine kernel.

    Attributes:
        depth: the depth of the equivalent neural network, which corresponds to nested computations
            of the `shape_function`. Defautls to 1.
        name: The name of the kernel. Defautls to `"ArcCosine"`.

    Raises:
        ValueError: if specified order is not in `{0, 1, 2}`.

    Returns:
        The kernel.
    """

    depth: int = field(default=1, pytree_node=False)
    name: str = field(default="ArcCosine", pytree_node=False)

    def __post_init__(self):
        """
        Check after the creation of the dataclass if we have provided a valid `order`.

        Raises:
            ValueError: if specified order is not in `{0, 1, 2}`.
        """
        if self.order not in {0, 1, 2}:
            raise ValueError("Requested order is not implemented.")

    def shape_function(self, param: Param, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """
        The shape function that determines the equivalent kernel to a deep network architecture.

        Args:
            param: The `Param` initialised with the kernel.
            x: Input Array.
        """
        x = self._squash(x)

        # step function for 1 layer
        def step(carry: Array, dummy: Optional[Array] = None) -> Tuple[Array, Array]:
            y = self.kappa(carry, self.order)
            return y, y

        # run scan for going deep.
        _, y = jax.lax.scan(step, x, xs=None, length=self.depth)
        return y[-1]


class NTK(Spherical):
    """
    The neural tangent kernel.

    Attributes:
        depth: the depth of the equivalent neural network, which corresponds to nested computations
            of the `shape_function`. Defautls to 1.
        name: The name of the kernel. Defautls to `"NTK"`.

    Returns:
        _description_
    """

    order: int = field(init=False, pytree_node=False)
    depth: int = field(default=1, pytree_node=False)
    name: str = field(default="NTK", pytree_node=False)

    def __post_init__(self):
        """Hard-wire the `order` to be 1."""
        object.__setattr__(self, "order", 1)

    def shape_function(self, param: Param, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """
        The shape function that determines the equivalent kernel to a deep network architecture.

        Args:
            param: The `Param` initialised with the kernel.
            x: Input Array.
        """
        x = self._squash(x)

        # step function for 1 layer
        def step(
            carry: Tuple[Array, Array], dummy: Optional[Array] = None
        ) -> Tuple[Tuple[Array, Array], Array]:
            x, y = carry
            x, y = self.kappa(x, 1), y * self.kappa(x, 0) + self.kappa(x, 1)
            carry = x, y
            return carry, y

        # run scan for going deep.
        _, y = jax.lax.scan(step, (x, x), xs=None, length=self.depth)
        return y[-1] / (self.depth + 1)


class PolynomialDecay(Spherical):
    """
    The polynomial decay kernel which parameterises the decay rate of the eigenvalues.

    The functional form is:

        K(u) = Σₙ n⁻ᵝ Cᵅₙ(u) (n + α) / α,

        where n is the ordinal frequency, α is a constant on the sphere and β the decay rate.

    Attributes:
        truncation_level: the order at which we truncate the frequencies.
        name: The name for the object. Defautls to `"PolynomialDecay"`.
    """

    truncation_level: int = field(default=10, pytree_node=False)
    order: int = field(init=False, pytree_node=False)
    name: str = field(default="PolynomialDecay", pytree_node=False)

    def __post_init__(self):
        """Hard-wire the `order` to be 1."""
        object.__setattr__(self, "order", 1)

    def _compute_eigvals(
        self,
        sphere_dim: int,
        *,
        max_num_eigvals: Optional[int] = None,
        gegenbauer_lookup_table: Optional[GegenbauerLookupTable] = None,
    ):
        """
        Compute the eigenvalues of the PolynomialDecay kernel.

        Without an initialised parameter we can only compute the constant part of the eigenvalues
            α / (n + α) / Cᵅₙ(1)

        Args:
            sphere_dim: The dimensionality of the sphere.
            max_num_eigvals: The maximum number of eigenvalues to precompute.
                This is ignored as the kernel uses the `self.truncation_level` instead.
            gegenbauer_lookup_table: A precomputed lookup table to evaluate the Gegenbauer
                polynomial. Defaults to None.

        Raises:
            ValueError: If `gegenbauer_lookup_table` is specified when we don't use a
                `PolynomialDecay` kernel.

        Returns:
            Array with the eigenvalues of the kernel.
        """
        if not gegenbauer_lookup_table:
            raise ValueError("Lookup table should be provided with `PolyDecay` kernel.")

        dim = sphere_dim + 1
        alpha = (dim - 2.0) / 2.0
        geg = lambda n: gegenbauer_lookup_table(n, alpha, jnp.array(1.0, dtype=jnp.float64))
        C_1 = jax.vmap(geg)(jnp.arange(self.truncation_level, dtype=jnp.int32))
        n = jnp.arange(self.truncation_level, dtype=jnp.float64)
        return alpha / (n + alpha) / C_1

    def eigenvalues(self, param: Param, levels: Int[Array, " N"]) -> Float[Array, " N"]:
        """
        Get the eigenvalues of the PolynomialDecay kernel for the specified levels.

        For the nth frequency we have::
            λₙ = (n + 1)⁻ᵝ α / (n + α) / Cᵅₙ(1) / Σₘ(m + 1)⁻ᵝ

        NOTE: we clip the provided levels to the `self.truncation_level`.

        Args:
            param: A `Param` initialised with the kernel.
            levels: An array specifying up to which order de we need eigenvalues.

        Returns:
            The eigenvalues.
        """
        n = jnp.arange(self.truncation_level, dtype=jnp.float64)
        beta = param.params[self.name]["beta"]
        sphere_dim = param.constants["sphere"]["sphere_dim"]
        geg = param.constants["sphere"]["gegenbauer_lookup_table"]
        decay = (1 + n) ** (-beta)
        const_factor = self._compute_eigvals(sphere_dim, gegenbauer_lookup_table=geg)
        eigval = decay * const_factor / jnp.sum(decay)
        levels = jnp.clip(levels, 0, self.truncation_level - 1)
        return eigval[levels]

    def shape_function(self, param: Param, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """
        The shape function that determines the equivalent kernel to a deep network architecture.

        Args:
            param: The `Param` initialised with the kernel.
            x: Input Array.

        Raises:
            NotImplementedError: All derived classes need to implement this method.
        """
        x = self._squash(x)

        alpha = param.constants["sphere"]["alpha"]
        gegenbauer_lookup = param.constants["sphere"]["gegenbauer_lookup_table"]
        levels = jnp.arange(self.truncation_level, dtype=jnp.int32)

        const_factor = self.eigenvalues(param, levels)
        eigvals = const_factor * (levels.astype(jnp.float64) + alpha) / alpha
        C_n_x = jax.vmap(lambda n: gegenbauer_lookup(n, alpha, x))(levels)
        return jnp.sum(jax.vmap(lambda n, e: n * e)(C_n_x, eigvals), 0)
