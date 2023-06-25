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
import jax.random as jr
import numpy as np
import pytest
from jax.tree_util import tree_leaves, tree_map

from gpfy.spherical_harmonics import SphericalHarmonics, _num_harmonics

key = jr.PRNGKey(42)


freq_list = [2, 5]
phases_list = [5, 10]


def test_init_fails_when_passing_params_with_wrong_sphere(kernel_param_5d):
    sh = SphericalHarmonics(5, 10)
    kernel_param = kernel_param_5d[1]
    with pytest.raises(ValueError):
        _ = sh.init(key, input_dim=4, param=kernel_param)


@pytest.mark.parametrize("use_kernel_param", [True, False])
@pytest.mark.parametrize("freq", freq_list)
@pytest.mark.parametrize("phases", phases_list)
def test_init_create_correct_variables(use_kernel_param, freq, phases, kernel_param_5d):
    input_dim = 5
    kernel_param = kernel_param_5d[1]
    kernel_param = kernel_param if use_kernel_param else None
    sh = SphericalHarmonics(freq, phases)
    param = sh.init(key, input_dim=input_dim, param=kernel_param)

    assert all(f"V_{f}" in param.params["variational"]["inducing_features"] for f in range(freq))

    # check trainability
    for n, v in enumerate(param._trainables["variational"]["inducing_features"].values()):
        assert v is False if phases >= _num_harmonics(input_dim + 1, n) else True

    assert "sphere" in param.constants
    assert "gegenbauer_lookup_table" in param.constants["sphere"]

    orth_basis = param.constants["variational"]["inducing_features"]["orthogonal_basis"]
    if any(tree_leaves(param._trainables["variational"]["inducing_features"])):
        assert all(tree_map(lambda b: jnp.isnan(b).all(), orth_basis))
    else:
        assert not any(tree_map(lambda b: jnp.isnan(b).any(), orth_basis))


@pytest.mark.parametrize("freq", freq_list)
@pytest.mark.parametrize("phases", phases_list)
def test_correct_levels(freq, phases):
    sh = SphericalHarmonics(freq, phases)
    np.testing.assert_allclose(sh.levels, jnp.arange(freq))


@pytest.mark.parametrize("freq", freq_list)
@pytest.mark.parametrize("phases", phases_list)
def test_Vs_are_normalised(freq, phases):
    input_dim = 4
    sh = SphericalHarmonics(freq, phases)
    param = sh.init(key, input_dim=input_dim)
    Vs = sh.Vs(param)
    norms = tree_map(lambda v: jax.vmap(lambda vv: jnp.dot(vv, vv))(v), Vs)
    np.testing.assert_allclose(jnp.concatenate(norms), 1.0)


def test_num_phases_do_not_exceed_truncation_level():
    input_dim = 5
    phase_truncation = 3  # from the 2nd frequency we have to truncat since 2nd has num_dim phases.
    sh = SphericalHarmonics(3, phase_truncation=phase_truncation)
    param = sh.init(key, input_dim=input_dim)

    assert all(phases <= phase_truncation for phases in sh.num_phase_in_frequency(param))


@pytest.mark.parametrize("freq", freq_list)
@pytest.mark.parametrize("phases", phases_list)
def test_num_inducing_matches_shape_of_Vs(freq, phases):
    input_dim = 4
    sh = SphericalHarmonics(freq, phases)
    param = sh.init(key, input_dim=input_dim)
    assert jnp.concatenate(sh.Vs(param), 0).shape[0] == sh.num_inducing(param)


@pytest.mark.parametrize("freq", freq_list)
@pytest.mark.parametrize("phases", phases_list)
def test_Kuu_shape(freq, phases, kernel_param_5d):
    k, kernel_param = kernel_param_5d
    input_dim = 5
    sh = SphericalHarmonics(freq, phases)
    param = sh.init(key, input_dim=input_dim, param=kernel_param)
    Kuu = sh.Kuu(param, k)
    assert Kuu.shape[0] == sh.num_inducing(param)


@pytest.mark.parametrize("freq", freq_list)
@pytest.mark.parametrize("phases", phases_list)
def test_Kuf_shape(freq, phases, kernel_param_5d):
    k, kernel_param = kernel_param_5d
    input_dim = 5
    x = jr.normal(key, (10, 5), dtype=jnp.float64)
    sh = SphericalHarmonics(freq, phases)
    param = sh.init(key, input_dim=input_dim, param=kernel_param)
    Kuf = sh.Kuf(param, k, x)
    np.testing.assert_array_equal(Kuf.shape, (sh.num_inducing(param), x.shape[0]))
