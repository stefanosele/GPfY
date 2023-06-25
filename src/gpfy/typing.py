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

from typing import Any, Callable, Dict, MutableMapping, Optional, Sequence, Tuple, Union

import jax
from jaxtyping import Array, Float

from gpfy.training import TrainState

PRNG = Union[jax.random.PRNGKeyArray, Array]
ActiveDims = Union[slice, Sequence[int]]
Collection = MutableMapping[str, Any]
VariableDict = MutableMapping[str, Collection]
BijectorDict = MutableMapping[str, Collection]
ConstantDict = MutableMapping[str, Collection]
TrainableDict = MutableMapping[str, Collection]

TrainingData = Tuple[Float[Array, "N Din"], Float[Array, "N Dout"]]
TrainingDataDict = Dict[str, Float[Array, "N D"]]
TrainStepFn = Callable[[TrainState, Optional[Array]], Tuple[TrainState, float]]
