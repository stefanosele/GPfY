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

import jax
import jax.numpy as jnp

from shgp.gegenbauer import gegenbauer
from shgp.likelihoods import Gaussian
from shgp.model import GP, elbo
from shgp.spherical import NTK
from shgp.spherical_harmonics import SphericalHarmonics
from shgp.variational import VariationalDistributionTriL

key = jax.random.PRNGKey(42)
k = NTK(depth=5)
sh = SphericalHarmonics(10, phase_truncation=50)
lik = Gaussian()
q = VariationalDistributionTriL()
m = GP(k)
m_new = m.conditional(sh, q)
param = m_new.init(key, 2, 1, likelihood=lik, sh_features=sh, variational_dist=q)

levels = jnp.split(sh.levels, 10)

xs = [jnp.matmul(v, v.T) for v in sh.Vs(param)]
alpha = 0.5

X = jax.random.normal(key, (100, 2))
Y = jax.random.normal(key, (100, 1))

x_s, r = k.to_sphere(param, X)

jit_elbo = jax.jit(elbo)
jit_grad = jax.jit(jax.value_and_grad(elbo))


@jax.jit
def loss(param_free):
    param = param_free.constrained()
    return -elbo(param, m_new, q, lik, (X, Y))


param_free = param.unconstrained()
import optax

frozen = jax.tree_map(lambda x: not (x), param_free._trainables)
_, tree_def = jax.tree_util.tree_flatten(param_free)
frozen = jax.tree_util.tree_unflatten(tree_def, jax.tree_util.tree_leaves(frozen))
opt = optax.chain(optax.adam(1e-3), optax.masked(optax.set_to_zero(), frozen))
opt_state = opt.init(param_free)  # type: ignore


@jax.jit
def step(param_free, opt_state):
    loss_val, loss_grad = jax.value_and_grad(loss)(param_free)
    updates, opt_state = opt.update(loss_grad, opt_state, param_free)
    param_free = optax.apply_updates(param_free, updates)
    # updates, opt_state = opt.update(loss_grad.params, opt_state, param_free.params)
    # new_params = optax.apply_updates(param_free.params, updates)
    # param_free = param_free.replace(params=new_params)
    return loss_val, param_free, opt_state


def step2(param_free_and_opt_state, lala):
    param_free, opt_state = param_free_and_opt_state
    loss_val, loss_grad = jax.value_and_grad(loss)(param_free)
    updates, opt_state = opt.update(loss_grad, opt_state, param_free)
    param_free = optax.apply_updates(param_free, updates)
    # updates, opt_state = opt.update(loss_grad.params, opt_state, param_free.params)
    # new_params = optax.apply_updates(param_free.params, updates)
    # param_free = param_free.replace(params=new_params)
    return (param_free, opt_state), loss_val


def func(v, n, L):
    vxT = jnp.dot(v, x_s.T)
    zonal = gegenbauer(n[0], alpha, vxT)
    return jax.lax.linalg.triangular_solve(L, zonal, left_side=True)


def dummy(params, param, levels):
    # Vs = sh.Vs(param)
    Ls = sh.Ls(param)
    Vs = jax.tree_util.tree_leaves(params["variational"]["inducing_features"])
    poly = jax.tree_map(func, Vs, levels, Ls)
    return sum(jnp.sum(p) for p in poly)


elbo_vals = []
from tqdm import trange

# for i in trange(2000):
#     elbo_val, param_free, opt_state = step(param_free, opt_state)
#     # if i % 100 == 0:
#     #     print(elbo_val)
#     elbo_vals.append(elbo_val)
