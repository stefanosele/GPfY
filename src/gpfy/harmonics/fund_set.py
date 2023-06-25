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

from pathlib import Path
from typing import Callable

import numpy as np
from jax import Array


def fundamental_set_loader(dimension: int, load_dir="fundamental_system") -> Callable[[int], Array]:
    load_dir = Path(__file__).parents[0] / ".." / load_dir
    file_name = load_dir / f"fs_{dimension}D.npz"

    cache = {}
    if file_name.exists():
        with np.load(file_name) as f:
            cache = {k: v for (k, v) in f.items()}
    else:
        raise ValueError(f"Fundamental system for dimension {dimension} has not been precomputed.")

    def load(degree: int) -> Array:
        key = f"degree_{degree}"
        if key not in cache:
            raise ValueError()
        return cache[key]

    return load
