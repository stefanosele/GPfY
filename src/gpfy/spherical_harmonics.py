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

from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax._src.lax.linalg import triangular_solve
from jax.tree_util import tree_leaves, tree_map
from jaxtyping import Array, Float
from scipy.special import comb

from gpfy.gegenbauer import GegenbauerLookupTable, gegenbauer
from gpfy.harmonics.fund_set import fundamental_set_loader
from gpfy.param import Param, identity
from gpfy.spherical import Spherical
from gpfy.typing import PRNG, BijectorDict, ConstantDict, TrainableDict, VariableDict
from gpfy.utils import dataclass, field


def _num_harmonics(dim: int, frequency: int) -> int:
    """
    Compute the number of spherical harmonic functions in a given frequency.

    Args:
        dim: The dimensionality of the input on the Sᵈ⁻¹ sphere.
        frequency: The order of the frequency.

    Returns:
        The number of spherical harmonic functions for the the nth frequency.
    """
    if frequency == 0:
        return 1
    else:
        c1 = comb(frequency + dim - 3, frequency - 1)
        c2 = comb(frequency + dim - 2, frequency)
    return int(c1 + c2)


@dataclass
class SphericalHarmonics:
    """
    Spherical harmonics inducing features for sparse inference in Gaussian processes.

    The spherical harmonics, Yₙᵐ(·) of frequency n and phase m are eigenfunctions on the sphere and,
    as such, they form an orthogonal basis.

    To construct the harmonics, we use a a fundamental set of points on the sphere {vᵢ}ᵢ and compute
    b = {Cᵅₙ(<vᵢ, x>)}ᵢ. b now forms a complete basis on the sphere and we can orthogoalise it via
    a Cholesky decomposition. However, we only need to run the Cholesky decomposition once during
    initialisation.

    NOTE: If we truncate the harmonics up to a maximum phase, then we need to learn the fundamental
    set of points {vᵢ}ᵢ from data. This requires to run a Cholesky decomposition at every iteration.

    Attributes:
        num_frequencies: The number of frequencies, up to which, we compute the harmonics.
        phase_truncation: The number for truncating the phases/harmonics in every frequency.
            Defaults to max(int), which essentially means no truncation.

    Returns:
        An instance of the spherical harmonics features.
    """

    num_frequencies: int = field(default_factory=10, pytree_node=False)
    phase_truncation: int = field(default=2**31 - 1, pytree_node=False)

    @property
    def levels(self):
        return jnp.arange(self.num_frequencies, dtype=jnp.int32)

    def init(self, key: PRNG, input_dim: int, param: Optional[Param] = None) -> Param:
        """
        Initialise the parameters of the spherical harmonic features and return a `Param` object.

        Args:
            key: A random key for initialising the fundamental set in case we truncate the phases.
            input_dim: The input dimension. We then append an extra dimension for bias.
            param: An already initialised collection of parameters from another object, i.e., a
                kernel. If it is provided we extend the `Param` with the additional collection.
                Otherwise we return a new `Param`. Defaults to None.

        Raises:
            ValueError: if a `param` is provided but the `sphere_dim`s do not match.

        Returns:
            The `Param` object with all variables.
        """
        sphere_dim = input_dim
        dim = sphere_dim + 1

        # check if we have a sphere initialised in our param object and fail if it doesn't match.
        sphere = param.constants.get("sphere") if param else {}
        if sphere and sphere_dim != sphere["sphere_dim"]:
            raise ValueError("`param` contains a sphere that is not compatible to `input_dim`.")

        # Try loading a pre-computed fundamental set.
        try:
            fund_set = fundamental_set_loader(dim)
        except ValueError:
            fund_set = None

        # initialise the parameters Vs. Set them to non-trainable if we do not truncate the phase.
        Vs = {}
        trainables = {}
        for n in self.levels:
            num_phase = _num_harmonics(dim, n)

            if (num_phase <= self.phase_truncation) and fund_set:
                V = fund_set(n)
                Vs[f"V_{n}"] = V
                trainables[f"V_{n}"] = False
            else:
                key, subkey = jax.random.split(key)
                V = jax.random.normal(subkey, (min(self.phase_truncation, num_phase), dim))
                Vs[f"V_{n}"] = V
                trainables[f"V_{n}"] = True

        # initialise the orthogonal basis with nans to get the structure
        orth_basis = tree_leaves(
            tree_map(lambda x: jnp.nan * jnp.zeros((x.shape[0], x.shape[0]), dtype=x.dtype), Vs)
        )

        # create the collections for the current object.
        bijectors = {k: identity() for k in Vs.keys()}
        collection = "variational"
        var_params = {"inducing_features": Vs}
        var_trainables = {"inducing_features": trainables}
        var_bijectors = {"inducing_features": bijectors}
        var_constants = {"inducing_features": {"orthogonal_basis": orth_basis}}

        # initialise the Gegenbauer lookup table and compute the relevant constatns on the sphere.
        if (
            not sphere
            or ("gegenbauer_lookup_table" not in sphere)
            or (sphere["gegenbauer_lookup_table"].max_level < self.num_frequencies)
        ):
            sphere = sphere or {}
            alpha = (dim - 2.0) / 2.0
            sphere["sphere_dim"] = sphere_dim
            sphere["alpha"] = alpha
            geg = GegenbauerLookupTable(self.num_frequencies, alpha)
            sphere["gegenbauer_lookup_table"] = geg

        all_params: VariableDict = {}
        all_trainables: TrainableDict = {}
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

        # append the initialised parameters for spherical harmonic features to the collections
        all_params[collection] = var_params
        all_bijectors[collection] = var_bijectors
        all_trainables[collection] = var_trainables
        all_constants[collection] = var_constants
        all_constants["sphere"] = sphere

        # create a new Param object.
        param = Param(
            params=all_params,
            _trainables=all_trainables,
            _bijectors=all_bijectors,
            constants=all_constants,
            _constrained=constrained,
        )

        # if we don't have any phase_truncation (so no trainable vars), pre-compute and save the
        # orthogonal basis, otherwise keep it with NaNs.
        if not any(tree_leaves(var_trainables)):
            orth_basis = self.orthogonalise_basis(param)

        all_constants[collection] = {"inducing_features": {"orthogonal_basis": orth_basis}}
        param = param.replace(constants=all_constants)
        return param

    def Vs(self, param: Param) -> List[Array]:
        """
        Get the fundamental set of points.

        NOTE: if we learn the fundamental set, then we normalise the points.

        Args:
            param: A `Param` initialised with the spherical harmonic features.

        Returns:
            A list containing the fundamental set of points at every frequency.
        """
        Vs = tree_leaves(param.params["variational"]["inducing_features"])

        # the orth_basis is nans if we learn the Vs
        orth_basis = param.constants["variational"]["inducing_features"]["orthogonal_basis"]

        def _normalise():
            return tree_map(
                lambda v: v / jnp.sqrt(jnp.sum(jnp.square(v), axis=1, keepdims=True)),
                Vs,
            )

        return jax.lax.cond(jnp.isnan(jnp.sum(orth_basis[0])), _normalise, lambda: Vs)

    def Ls(self, param: Param) -> List[Array]:
        """
        Compute the orthogonalised basis.

        NOTE: We compute the orthogonalisation only once, at the initialisation, if we don't
        truncate the phases/harmonics. Otherwise we need to orthogonalise at every iteration.

        Args:
            param: A `Param` initialised with the spherical harmonic features.

        Returns:
            A list containing the orthogonal basis at every frequency.
        """
        # the orth_basis is nans if we learn the Vs
        orth_basis = param.constants["variational"]["inducing_features"]["orthogonal_basis"]

        return jax.lax.cond(
            jnp.isnan(jnp.sum(orth_basis[0])),
            lambda: self.orthogonalise_basis(param),
            lambda: orth_basis,
        )

    def num_phase_in_frequency(self, param: Param) -> List[int]:
        """
        Get the total number of phases/harmonics at every frequency.

        Args:
            param: A `Param` initialised with the spherical harmonic features.

        Returns:
            A list with the number of phases per frequency.
        """
        return jax.tree_map(lambda x: x.shape[0], self.Vs(param))

    def num_inducing(self, param) -> int:
        """
        Computes the total number of inducing features, as the sum of all phases.

        Args:
            param: A `Param` initialised with the spherical harmonic features.

        Returns:
            The total number of inducing features.
        """
        return sum(self.num_phase_in_frequency(param))

    def orthogonalise_basis(self, param: Param) -> List[Array]:
        """
        Compute the basis from the fundamental set and orthogonalise it via Cholesky decomposition.

        Args:
            param: A `Param` initialised with the spherical harmonic features.

        Returns:
            A list containing the orthogonal basis at every frequency.
        """
        alpha = param.constants["sphere"]["alpha"]
        # gegenbauer = param.constants["sphere"]["gegenbauer_lookup_table"]
        levels = jnp.split(self.levels, self.num_frequencies)
        const = alpha / (alpha + self.levels.astype(jnp.float64))
        const = jnp.split(const, self.num_frequencies)

        def _func(v, n, c):
            x = jnp.matmul(v, v.T)
            B = c * gegenbauer(n[0], alpha, x)
            L = jnp.linalg.cholesky(B + 1e-16 * jnp.eye(B.shape[0], dtype=B.dtype))
            return L

        Ls = tree_map(_func, self.Vs(param), levels, const)
        return Ls

    def polynomial_expansion(self, param: Param, X: Float[Array, "N D"]) -> Float[Array, "M N"]:
        """
        Evaluate the polynomial expansion of an input on the sphere given the harmonic basis.

        Args:
            param: A `Param` initialised with the spherical harmonic features.
            X: Input Array.

        Returns:
            The harmonics evaluated at the input as a polynomial expansion of the basis.
        """
        alpha = param.constants["sphere"]["alpha"]
        # gegenbauer = param.constants["sphere"]["gegenbauer_lookup_table"]
        levels = jnp.split(self.levels, self.num_frequencies)
        const = alpha / (alpha + self.levels.astype(jnp.float64))
        const = jnp.split(const, self.num_frequencies)

        def _func(v, n, L):  # , c):
            vxT = jnp.dot(v, X.T)
            zonal = gegenbauer(n[0], alpha, vxT)

            # vvT = jnp.matmul(v, v.T)
            # B = c * gegenbauer(n[0], alpha, vvT)
            # # B = c * geg2(n[0], alpha, vvT)
            # L = jnp.linalg.cholesky(B + 1e-16 * jnp.eye(B.shape[0], dtype=B.dtype))

            harmonic = triangular_solve(L, zonal, left_side=True, lower=True)
            return harmonic

        harmonics = tree_map(_func, self.Vs(param), levels, self.Ls(param))  # , const)
        return jnp.concatenate(harmonics, axis=0)

    def Kuu(self, param: Param, kernel: Spherical) -> Float[Array, " M"]:
        """
        Compute the covariance between the harmonic features.

        It is as a diagonal matrix which holds the reciprocal of the eigenvalues.

        Args:
            param: A `Param` initialised with the spherical harmonic features.
            kernel: A `Spherical` kernel.

        Returns:
            The covariance of the inducing features.
        """
        eigs = kernel.eigenvalues(param, self.levels)

        # split the eigenvalue array to a list so we can pass it to tree_map.
        eigs = jnp.split(eigs, self.num_frequencies)

        # get the repetition of each eigenvalue, i.e., the harmonic phases in each frequency.
        reps = self.num_phase_in_frequency(param)
        return jnp.concatenate(jax.tree_util.tree_map(lambda e, r: jnp.ones(r) / e, eigs, reps))

    def Kuf(self, param: Param, kernel: Spherical, x: Float[Array, "N D"]) -> Float[Array, "M N"]:
        """
        Compute the covariance between the harmonic features and the provided input.

        Args:
            param: A `Param` initialised with the spherical harmonic features.
            kernel: A `Spherical` kernel.
            x: Input Array.

        Returns:
            The cross covariance between the inducing features and the input.
        """
        # project the input to the unit sphere.
        x_sphere, rx = kernel.to_sphere(param, x)

        # get the harmonics evaluated at the input as a polynomial expansion of the basis.
        sh = self.polynomial_expansion(param, x_sphere)
        variance = param.params[kernel.name]["variance"]
        return sh * rx[..., None, :] * jnp.sqrt(variance)  # apply back to radius before returning

    def conditional_fun(
        self,
        kernel: Spherical,
    ) -> Tuple[
        Callable[[Param, Float[Array, "N D"]], Float[Array, "M N"]],
        Callable[[Param, Float[Array, "N D"], Float[Array, "M N"]], Float[Array, " N"]],
        Callable[[Param, Float[Array, "N D"], Float[Array, "M N"]], Float[Array, "N N"]],
    ]:
        """
        Prepare the necessary functions for conditioning a GP on the spherial harmonic features.

        We do not explicitly evalutate the functions here as we only want to pass it to a GP so that
        we know how to perform the conditioning.

        Args:
            kernel: The `Spherical` kernel of the GP.

        Returns:
            A tuple of 3 functions for computing the projection of the input, the conditional
            diagonal variance and the conditional covariance.
        """
        project_fun = lambda param, x: jnp.sqrt(1 / self.Kuu(param, kernel))[..., None] * self.Kuf(
            param, kernel, x
        )
        conditional_var_fun = lambda param, x, proj: kernel.K_diag(param, x) - jnp.sum(
            jnp.square(proj), -2
        )
        conditional_cov_fun = lambda param, x, proj: kernel.K(param, x) - jnp.matmul(
            proj.swapaxes(-1, -2), proj
        )

        return (project_fun, conditional_cov_fun, conditional_var_fun)
