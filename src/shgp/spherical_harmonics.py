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

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array, random
from jax._src.lax.linalg import triangular_solve
from jax.tree_util import tree_leaves, tree_map
from jaxtyping import ArrayLike, Float, Int
from scipy.special import comb

from shgp.gegenbauer import GegenbauerLookupTable, gegenbauer, gegenbauer2
from shgp.harmonics.fund_set import fundamental_set_loader
from shgp.param import Param, identity
from shgp.spherical import Spherical
from shgp.utils import dataclass, field

PRNG = Union[random.PRNGKeyArray, jax.Array]


def _num_harmonics(dim: int, frequency: int) -> int:
    if frequency == 0:
        return 1
    elif dim == 3:
        return int(2 * frequency + 1)
    else:
        c = comb(frequency + dim - 3, frequency - 1)
        return int(jnp.round((2 * frequency + dim - 2) * c / frequency))


@dataclass
class SphericalHarmonics:
    num_frequencies: int = field(default_factory=10, pytree_node=False)
    phase_truncation: int = field(default=2**31 - 1, pytree_node=False)

    @property
    def levels(self):
        return jnp.arange(self.num_frequencies, dtype=jnp.int32)

    def init(self, key: PRNG, input_dim: int, param: Optional[Param] = None) -> Param:
        sphere_dim = input_dim + 1
        sphere = param.constants.get("sphere") if param else {}
        if sphere and sphere_dim != sphere["sphere_dim"]:
            raise ValueError(
                "`param` contains a sphere that is not compatible to `input_dim`."
            )
        try:
            fund_set = fundamental_set_loader(sphere_dim)
        except ValueError:
            fund_set = None
        Vs = {}
        trainables = {}
        for n in self.levels:
            num_phase = _num_harmonics(sphere_dim, n)

            if (num_phase <= self.phase_truncation) and fund_set:
                V = fund_set(n)
                Vs[f"V_{n}"] = V
                trainables[f"V_{n}"] = False
            else:
                key, subkey = jax.random.split(key)
                V = jax.random.normal(
                    subkey, (min(self.phase_truncation, num_phase), sphere_dim)
                )
                Vs[f"V_{n}"] = V
                trainables[f"V_{n}"] = True

        bijectors = {k: identity() for k in Vs.keys()}
        collection = "variational"
        var_params = {"inducing_features": Vs}
        var_trainables = {"inducing_features": trainables}
        var_bijectors = {"inducing_features": bijectors}
        # constants = {}

        if (
            not sphere
            or ("gegenbauer_lookup_table" not in sphere)
            or (sphere["gegenbauer_lookup_table"].max_level < self.num_frequencies)
        ):
            sphere = sphere or {}
            alpha = (sphere_dim - 2.0) / 2.0
            # geg = GegenbauerLookupTable(self.num_frequencies, alpha)
            sphere["sphere_dim"] = sphere_dim
            sphere["alpha"] = alpha
            # sphere["geg"] = jax.jit(
            #     lambda n, X: jax.vmap(
            #         lambda x1: jax.vmap(lambda x2: gegenbauer(n[0], alpha, x2))(x1)
            #     )(X)
            # )
            # sphere["geg"] = jax.jit(
            #     lambda n, X: jax.vmap(lambda x: gegenbauer(n[0], alpha, x))(X)
            # )
            # sphere["gegenbauer_lookup_table"] = geg

        all_params, all_trainables, all_bijectors, all_constants, constrained = (
            {},
            {},
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

        all_params[collection] = var_params
        all_bijectors[collection] = var_bijectors
        all_trainables[collection] = var_trainables
        all_constants["sphere"] = sphere
        param = Param(
            params=all_params,
            _trainables=all_trainables,
            _bijectors=all_bijectors,
            constants=all_constants,
            _constrained=constrained,
        )

        if not any(tree_leaves(var_trainables)):
            orth_basis = self.orthogonalise_basis(param)
            all_constants[collection] = {
                "inducing_features": {"orthogonal_basis": orth_basis}
            }
            param = param.replace(constants=all_constants)
        return param

    def Vs(self, param: Param) -> List[Array]:
        Vs = tree_leaves(param.params["variational"].get("inducing_features"))

        # if the basis is pre-computed then no need for normalisation
        orth_precomputed = (
            param.constants.get("variational", {})
            .get("inducing_features", {})
            .get("orthogonal_basis", [])
        )
        if not orth_precomputed:
            return tree_map(
                lambda v: v / jnp.sqrt(jnp.sum(jnp.square(v), axis=1, keepdims=True)),
                Vs,
            )
        return Vs

    def Ls(self, param: Param) -> List[Array]:
        orth_basis = (
            param.constants.get("variational", {})
            .get("inducing_features", {})
            .get("orthogonal_basis", [])
        )
        if not orth_basis:
            orth_basis = self.orthogonalise_basis(param)
        return orth_basis

    def num_phase_in_frequency(self, param: Param) -> List[int]:
        return jax.tree_map(lambda x: x.shape[0], self.Vs(param))

    def num_inducing(self, param) -> int:
        return sum(self.num_phase_in_frequency(param))

    @jax.jit
    def orthogonalise_basis(self, param: Param) -> List[Array]:
        alpha = param.constants["sphere"]["alpha"]
        # gegenbauer = param.constants["sphere"]["gegenbauer_lookup_table"]
        # geg = param.constants["sphere"]["geg"]
        levels = jnp.split(self.levels, self.num_frequencies)
        const = alpha / (alpha + self.levels.astype(jnp.float64))
        const = jnp.split(const, self.num_frequencies)

        def _func(v, n, c):
            x = jnp.matmul(v, v.T)
            # B = c * geg(n, x)
            B = c * gegenbauer(n[0], alpha, x)
            # B = c * geg2(n[0], alpha, x)
            L = jnp.linalg.cholesky(B + 1e-16 * jnp.eye(B.shape[0], dtype=B.dtype))
            return L

        Ls = tree_map(_func, self.Vs(param), levels, const)
        return Ls

    @jax.jit
    def polynomial_expansion(
        self, param: Param, X: Float[Array, "N D"]
    ) -> Float[Array, "M N"]:
        print("TRACING")
        alpha = param.constants["sphere"]["alpha"]
        # gegenbauer = param.constants["sphere"]["gegenbauer_lookup_table"]
        # geg = param.constants["sphere"]["geg"]
        levels = jnp.split(self.levels, self.num_frequencies)
        const = alpha / (alpha + self.levels.astype(jnp.float64))
        const = jnp.split(const, self.num_frequencies)

        def _func(v, n, L):  # , c):
            vxT = jnp.dot(v, X.T)
            # zonal = geg(n, vxT)
            zonal = gegenbauer(n[0], alpha, vxT)

            # vvT = jnp.matmul(v, v.T)
            # B = c * gegenbauer(n[0], alpha, vvT)
            # # B = c * geg2(n[0], alpha, vvT)
            # L = jnp.linalg.cholesky(B + 1e-16 * jnp.eye(B.shape[0], dtype=B.dtype))

            # zonal = geg2(n[0], alpha, vxT)
            harmonic = triangular_solve(L, zonal, left_side=True)
            return harmonic

        harmonics = tree_map(_func, self.Vs(param), levels, self.Ls(param))  # , const)
        return jnp.concatenate(harmonics, axis=0)

    def Kuu(self, param, kernel):
        eigs = kernel.eigenvalues(param, self.levels)
        eigs = jnp.split(eigs, self.num_frequencies)
        reps = self.num_phase_in_frequency(param)
        return jnp.concatenate(
            jax.tree_util.tree_map(lambda e, r: jnp.ones(r) / e, eigs, reps)
        )

    def Kuf(self, param, kernel, x):
        x_sphere, rx = kernel.to_sphere(param, x)
        sh = self.polynomial_expansion(param, x_sphere)
        variance = param.params[kernel.name]["variance"]
        return sh * rx[..., None, :] * jnp.sqrt(variance)

    def conditional_fun(
        self,
        kernel: Spherical,
    ) -> Tuple[Callable, Callable, Callable]:
        # ) -> Callable[[Float[Array, "N D"]], Float[Array, "M N"]]:

        project_fun = lambda param, x: jnp.sqrt(1 / self.Kuu(param, kernel))[
            ..., None
        ] * self.Kuf(param, kernel, x)
        conditional_var_fun = lambda param, x: kernel.K_diag(param, x) - jnp.sum(
            jnp.square(project_fun(param, x)), -2
        )
        conditional_cov_fun = lambda param, x: kernel.K(param, x) - jnp.matmul(
            project_fun(param, x).swapaxes(-1, -2), project_fun(param, x)
        )
        return (project_fun, conditional_cov_fun, conditional_var_fun)
