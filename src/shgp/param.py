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

from typing import Any, Callable, Dict, Tuple, TypeVar, Union

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax.tree_util import tree_flatten, tree_map

from shgp.utils import PyTreeNode, field

positive = tfp.bijectors.Exp
identity = tfp.bijectors.Identity


class Param(PyTreeNode):
    params: Dict = field(default_factory=dict, pytree_node=True)
    _trainables: Dict = field(default_factory=dict, pytree_node=False)
    _bijectors: Dict = field(default_factory=dict, pytree_node=False)
    constants: Dict = field(default_factory=dict, pytree_node=False)
    _constrained: bool = field(default=True, pytree_node=False)

    def _has_valid_keys(self):
        # valid_keys = set(self.params.keys())
        # param_tree = jax.tree_util.tree_flatten_with_path(self.params)[1]
        # trainables_tree = jax.tree_util.tree_flatten_with_path(self._trainables)[1]
        # bijectors_tree = jax.tree_util.tree_flatten_with_path(self._bijectors)[1]
        if not self._is_subtree(self._trainables, self.params):
            raise ValueError(f"Invalid key in `_trainables`")
        if not self._is_subtree(self._bijectors, self.params):
            raise ValueError(f"Invalid key in `_bijectors`")
        if not all(
            collection in self.params or collection == "sphere"
            for collection in self.constants.keys()
        ):
            raise ValueError(f"Invalid key in `_constants`")

        # trainables_valid_keys = [k in self.params for k in self._trainables.keys()]
        # bijectors_valid_keys = [k in self.params for k in self._bijectors.keys()]

        # if not all(trainables_valid_keys):
        #     raise ValueError(f"Invalid key in `_trainables`")
        # if not all(bijectors_valid_keys):
        #     raise ValueError(f"Invalid key in `_bijectors`")

    def _is_subtree(self, t1: Union[dict, Any], t2: Union[dict, Any]):
        """t1 is subtree of t2."""
        if isinstance(t1, dict) and isinstance(t2, dict):
            ret = []
            for k1 in t1.keys():
                if k1 in t2:
                    ret.append(self._is_subtree(t1[k1], t2[k1]))
                else:
                    return False
            return all(ret)
        elif isinstance(t2, dict):
            return False
        else:  # we arrived at a leaf
            return True

    def _tree_update_from_subtree(self, t1: dict, t2: dict):
        """Update t1 from subtree t2."""
        ret = {}
        for k1, v1 in t1.items():
            if k1 in t2:
                if isinstance(v1, dict):
                    ret[k1] = self._tree_update_from_subtree(t1[k1], t2[k1])
                else:
                    ret[k1] = t2[k1]
            else:
                if isinstance(v1, dict):
                    ret[k1] = self._tree_update_from_subtree(t1[k1], t2)
                else:
                    ret[k1] = v1
        return ret

    def __post_init__(self):
        # check we have valid keys in all dicts
        self._has_valid_keys()

        # initialise the trainable status
        trainables = self._trainables
        if not trainables or (
            tree_flatten(trainables)[1] != tree_flatten(self.params)[1]
        ):
            trainables = tree_map(lambda _: True, self.params)
            trainables = self._tree_update_from_subtree(trainables, self._trainables)

        # same for initialising the bijectors
        bijectors = self._bijectors
        if not bijectors or (
            tree_flatten(bijectors)[1] != tree_flatten(self.params)[1]
        ):
            bijectors = tree_map(lambda _: positive(), self.params)
            bijectors = self._tree_update_from_subtree(bijectors, self._bijectors)

        # make params Arrays
        params = tree_map(lambda x: jnp.array(x, dtype=jnp.float64), self.params)
        # params = tree_map(lambda x: jnp.array(x), self.params)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "_trainables", trainables)
        object.__setattr__(self, "_bijectors", bijectors)

    def replace_param(self, collection: str, **kwargs):
        if collection not in self.params:
            raise ValueError(f"there is no {collection} collection")

        updates = self._tree_update_from_subtree(self.params[collection], kwargs)
        updates = self._tree_update_from_subtree(self.params, updates)
        return self.replace(params=updates)

    def set_trainable(self, collection: str, **kwargs):
        if collection not in self._trainables:
            raise ValueError(f"there is no {collection} collection")

        updates = self._tree_update_from_subtree(self._trainables[collection], kwargs)
        updates = self._tree_update_from_subtree(self._trainables, updates)
        return self.replace(_trainables=updates)

    def set_bijector(self, collection: str, **kwargs):
        if collection not in self._trainables:
            raise ValueError(f"there is no {collection} collection")

        updates = self._tree_update_from_subtree(self._bijectors[collection], kwargs)
        updates = self._tree_update_from_subtree(self._bijectors, updates)
        return self.replace(_bijectors=updates)

    def unconstrained(self):
        # if self._constrained:
        unconstrained_params = tree_map(
            lambda p, t: t.inverse(p), self.params, self._bijectors
        )
        return self.replace(_constrained=False, params=unconstrained_params)
        # else:
        #     return self

    def constrained(self):
        # if not self._constrained:
        constrained_params = tree_map(
            lambda p, t: t.forward(p), self.params, self._bijectors
        )
        return self.replace(_constrained=True, params=constrained_params)
        # else:
        #     return self
