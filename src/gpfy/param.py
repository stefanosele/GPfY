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

from typing import Any, Dict, Union

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax.tree_util import tree_map, tree_structure

from gpfy.typing import BijectorDict, ConstantDict, TrainableDict, VariableDict
from gpfy.utils import PyTreeNode, field

positive = tfp.bijectors.Exp
identity = tfp.bijectors.Identity


class Param(PyTreeNode):
    """
    Basic `PyTreeNode` that holds information regarding all the parameters of an initialised object.

    Attributes:
        params: A dictionary holding a collection of parameterised objects with a mapping to their
            parameters.
        _trainables: A dictionary with the same structure as `params` that specifies if a parameter
            is trainable or not. It defaults to `True` for all unspecified parameters.
        _bijectors: A dictionary with the same structure as `params` that specifies the required
            bijector to transform to the unconstrained space. It defaults to a positive
            `tfp.bijectors.Exp` for all unspecified parameters.
        constants: A dictionary that holds information for additional variables that are considered
            constant during optimisation.
        _constrained: A flag specifying if the parameters are constrained or not.

    Note that only the `params` attribute acts as a pytree_node.
    """

    params: VariableDict = field(default_factory=dict, pytree_node=True)
    _trainables: TrainableDict = field(default_factory=dict, pytree_node=False)
    _bijectors: BijectorDict = field(default_factory=dict, pytree_node=False)
    constants: ConstantDict = field(default_factory=dict, pytree_node=False)
    _constrained: bool = field(default=True, pytree_node=False)

    def _has_valid_keys(self) -> None:
        """
        Checks if the provided collections have the same structure as the `self.params`.

        Raises:
            ValueError: if the user specified `self._trainables is not a subtree of `self.params`.
            ValueError: if the user specified `self._bijectors is not a subtree of `self.params`.
            ValueError: if the user specified `self._constants has collections other than
                the ones in `self.params` or `"sphere"`.
        """
        # valid_keys = set(self.params.keys())
        # param_tree = jax.tree_util.tree_flatten_with_path(self.params)[1]
        # trainables_tree = jax.tree_util.tree_flatten_with_path(self._trainables)[1]
        # bijectors_tree = jax.tree_util.tree_flatten_with_path(self._bijectors)[1]
        if not self._is_subtree(self._trainables, self.params):
            raise ValueError("Invalid key in `_trainables`")
        if not self._is_subtree(self._bijectors, self.params):
            raise ValueError("Invalid key in `_bijectors`")
        if not all(
            collection in self.params or collection == "sphere"
            for collection in self.constants.keys()
        ):
            raise ValueError("Invalid key in `_constants`")

    def _is_subtree(self, t1: Union[VariableDict, Any], t2: Union[VariableDict, Any]) -> bool:
        """
        Check if `t1` is subtree of `t2`, strating from the same level.

        Args:
            t1: a `VariableDict` pytree or a leaf node
            t2: a `VariableDict` pytree or a leaf node

        Returns:
            If `t1` is a subtree of `t2`.
        """
        if isinstance(t1, Dict) and isinstance(t2, Dict):
            ret = []
            for k1 in t1.keys():
                if k1 in t2:  # Check if a subtree of t1 has same structure as in t2
                    ret.append(self._is_subtree(t1[k1], t2[k1]))
                else:
                    return False
            return all(ret)
        elif isinstance(t2, Dict):  # t1 is a leaf but t2 is a tree
            return False
        else:  # both t1 and t2 are leaves
            return True

    def _tree_update_from_subtree(self, t1: VariableDict, t2: VariableDict) -> VariableDict:
        """
        Update tree `t1` from subtree `t2`.

        Args:
            t1: a `VariableDict` pytree.
            t2: a `VariableDict` pytree.

        Returns:
            A `VariableDict` with the updated values.
        """
        ret = {}
        for k1, v1 in t1.items():
            if k1 in t2:  # k1 needs to be updated
                if isinstance(v1, Dict):  # v1 is a tree so recurse
                    ret[k1] = self._tree_update_from_subtree(t1[k1], t2[k1])
                else:
                    ret[k1] = t2[k1]  # we have a leaf so update the value
            else:
                if isinstance(v1, Dict):  # check if t2 is a subtree of t1 so we need to recurse
                    ret[k1] = self._tree_update_from_subtree(t1[k1], t2)
                else:
                    ret[k1] = v1  # no update needed
        return ret

    def __post_init__(self) -> None:
        """
        Runs automatically after the `__init__` of the dataclass to do further checks.
        """
        # check we have valid keys in all dicts
        self._has_valid_keys()

        # initialise the trainable status to `True` for all unpsecified variables
        trainables = self._trainables
        if not trainables or (tree_structure(trainables) != tree_structure(self.params)):
            trainables = tree_map(lambda _: True, self.params)
            trainables = self._tree_update_from_subtree(trainables, self._trainables)

        # initialising the bijectors to `positive` for all unpsecified variables
        bijectors = self._bijectors
        if not bijectors or (
            tree_structure(bijectors, is_leaf=lambda x: isinstance(x, tfp.bijectors.Bijector))
            != tree_structure(self.params)
        ):
            bijectors = tree_map(lambda _: positive(), self.params)
            bijectors = self._tree_update_from_subtree(bijectors, self._bijectors)

        # make sure all params are Arrays with float64 dtype
        params = tree_map(lambda x: jnp.array(x, dtype=jnp.float64), self.params)

        # write back the modified `VariableDict`s
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "_trainables", trainables)
        object.__setattr__(self, "_bijectors", bijectors)

    def replace_param(self, collection: str, **kwargs) -> "Param":
        """
        Replace the value of parameters in the `VariableDict` from a specified `collection`.

        Args:
            collection: the name of the collection that holds the target variable.
            kwargs: The name and the new value of the target variables within the `collection`.

        Raises:
            ValueError: if the specified `collection` is not present in the `param` `VariableDict`.

        Returns:
            A new `Param` with the updated variables.
        """
        if collection not in self.params:
            raise ValueError(f"there is no {collection} collection")

        # first update the subtree within the specified collection
        updates = self._tree_update_from_subtree(self.params[collection], kwargs)
        # then update the params with the newly updated collection
        updates = self._tree_update_from_subtree(self.params, updates)
        return self.replace(params=updates)

    def set_trainable(self, collection: str, **kwargs) -> "Param":
        """
        Replace the trainable status of parameters in a specified `collection`.

        Args:
            collection: the name of the collection that holds the target variable.
            kwargs: The name and the new trainable status of the target variables within
                the `collection`.

        Raises:
            ValueError: if the specified `collection` is not present in the `param` `VariableDict`.

        Returns:
            A new `Param` with the updated trainable status of the variables.
        """
        if collection not in self._trainables:
            raise ValueError(f"there is no {collection} collection")

        # first update the subtree within the specified collection
        updates = self._tree_update_from_subtree(self._trainables[collection], kwargs)
        # then update the trainables with the newly updated collection
        updates = self._tree_update_from_subtree(self._trainables, updates)
        return self.replace(_trainables=updates)

    def set_bijector(self, collection: str, **kwargs):
        """
        Replace the bijector of parameters in a specified `collection`.

        Args:
            collection: the name of the collection that holds the target variable.
            kwargs: The name and the new bijector of the target variables within the `collection`.

        Raises:
            ValueError: if the specified `collection` is not present in the `param` `VariableDict`.

        Returns:
            A new `Param` with the updated bijectors of the variables.
        """
        if collection not in self._trainables:
            raise ValueError(f"there is no {collection} collection")

        # first update the subtree within the specified collection
        updates = self._tree_update_from_subtree(self._bijectors[collection], kwargs)
        # then update the bijectors with the newly updated collection
        updates = self._tree_update_from_subtree(self._bijectors, updates)
        return self.replace(_bijectors=updates)

    def unconstrained(self) -> "Param":
        """
        Move the `params` in the unconstrained (optimisation) space to optimise over them.

        NOTE: There is the logic to check if it is already unconstrained and return the same object,
        I just need to test it.

        Returns:
            The `Param` with the variables at the unconstrained space (the optimisation space).
        """
        # if self._constrained:
        unconstrained_params = tree_map(lambda p, t: t.inverse(p), self.params, self._bijectors)
        return self.replace(_constrained=False, params=unconstrained_params)
        # else:
        #     return self

    def constrained(self) -> "Param":
        """
        Move the `params` in the (original) constrained space.

        NOTE: There is the logic to check if it is already unconstrained and return the same object,
        I just need to test it.

        Returns:
            The `Param` with the variables at the constrained space (the original space).
        """
        # if not self._constrained:
        constrained_params = tree_map(lambda p, t: t.forward(p), self.params, self._bijectors)
        return self.replace(_constrained=True, params=constrained_params)
        # else:
        #     return self
