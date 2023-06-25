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
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

from gpfy.param import Param, identity, positive


def test_param_raises_with_invalid_tree_structure():
    """Test we raise if the structure does not much."""
    with pytest.raises(ValueError):
        Param(params={"a": 0}, _trainables={"b": False})

    with pytest.raises(ValueError):
        Param(params={"a": {"b": 0}}, _trainables={"c": 3.0})

    with pytest.raises(ValueError):
        Param(params={"a": 0}, _bijectors={"b": identity()})

    with pytest.raises(ValueError):
        Param(params={"a": {"b": 0}}, _bijectors={"c": positive()})

    with pytest.raises(ValueError):
        Param(params={"a": 0}, constants={"b": 3.0})


def test_is_subtree():
    """Test we correctly detect sub-trees starting from same level."""
    dummy = Param()
    assert dummy._is_subtree({"a": {"b": False}}, {"a": {"b": 1.0, "c": 0.0}})
    assert dummy._is_subtree({"b": False}, {"b": 1.0, "c": 0.0})
    assert dummy._is_subtree({"b": False}, {"b": 1.0})
    assert not dummy._is_subtree(1.0, {"c": 1.0})
    assert not dummy._is_subtree({"b": False}, {"c": 1.0})
    assert not dummy._is_subtree({"c": False}, {"a": {"b": 1.0}})


def test_param_construction_can_correctly_fill_the_dictionaries():
    """Test we correctly initialise bijectors and trainable status for unspecified variables."""
    param = Param(params={"a": 0.0, "b": 1.0}, _trainables={"b": False})
    assert jax.tree_util.tree_structure(param.params) == jax.tree_util.tree_structure(
        param._trainables
    )
    assert param._trainables["a"] is True  # by default we have trainables=True
    assert param._trainables["b"] is False

    assert jax.tree_util.tree_structure(param.params) == jax.tree_util.tree_structure(
        param._bijectors, is_leaf=lambda x: isinstance(x, tfp.bijectors.Bijector)
    )

    # default bijector is positive
    assert isinstance(param._bijectors["a"], positive)
    assert isinstance(param._bijectors["b"], positive)


def test_replace_param():
    """Test we can replace the params."""
    param = Param(params={"collection_a": {"a": 0.0, "b": 1.0}, "collection_b": {"c": 2.0}})
    new_param = param.replace_param(collection="collection_a", b=42.0)
    assert new_param.params["collection_a"]["b"] == 42.0
    new_param = new_param.replace_param(collection="collection_b", c=42.0)
    assert new_param.params["collection_b"]["c"] == 42.0

    with pytest.raises(ValueError):
        _ = param.replace_param(collection="unknown_collection", b=3.0)


def test_set_trainable():
    """Test we can change trainable status."""
    param = Param(params={"collection_a": {"a": 0.0, "b": 1.0}, "collection_b": {"c": 2.0}})
    assert all(v is True for v in param._trainables["collection_a"].values())
    assert all(v is True for v in param._trainables["collection_b"].values())

    param = param.set_trainable("collection_a", a=False, b=False)
    assert all(v is False for v in param._trainables["collection_a"].values())
    param = param.set_trainable("collection_b", c=False)
    assert all(v is False for v in param._trainables["collection_b"].values())

    with pytest.raises(ValueError):
        _ = param.set_trainable(collection="unknown_collection", b=False)


def test_set_bijector():
    """Test we can set bijector."""
    param = Param(params={"collection_a": {"a": 0.0, "b": 1.0}, "collection_b": {"c": 2.0}})
    assert all(isinstance(v, positive) for v in param._bijectors["collection_a"].values())
    assert all(isinstance(v, positive) for v in param._bijectors["collection_b"].values())

    param = param.set_bijector("collection_a", a=identity(), b=identity())
    assert all(isinstance(v, identity) for v in param._bijectors["collection_a"].values())
    param = param.set_bijector("collection_b", c=identity())
    assert all(isinstance(v, identity) for v in param._bijectors["collection_b"].values())

    with pytest.raises(ValueError):
        _ = param.set_bijector(collection="unknown_collection", b=identity())


def test_constrained_and_unconstrained():
    """Test we can apply fwd-inverse bijector."""
    param = Param(
        params={"collection_a": {"a": 0.1, "b": 1.0}, "collection_b": {"c": 2.0}},
        _bijectors={
            "collection_a": {"a": positive(), "b": identity()},
            "collection_b": {"c": positive()},
        },
    )

    unconstrained = param.unconstrained()

    expected_params = {
        "collection_a": {"a": jnp.log(0.1).astype(jnp.float64), "b": 1.0},
        "collection_b": {"c": jnp.log(2.0).astype(jnp.float64)},
    }

    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose, unconstrained.params, expected_params
    )

    constrained = unconstrained.constrained()
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose, constrained.params, param.params
    )
