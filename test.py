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
from datasets import Dataset
from uci_datasets import Dataset as DatasetUCI

from gpfy.likelihoods import Gaussian
from gpfy.model import GP
from gpfy.optimization import create_training_step
from gpfy.spherical import NTK
from gpfy.spherical_harmonics import SphericalHarmonics
from gpfy.variational import VariationalDistributionTriL

key = jax.random.PRNGKey(42)
k = NTK(depth=5)
sh = SphericalHarmonics(10, phase_truncation=30)
lik = Gaussian()
q = VariationalDistributionTriL()
m = GP(k)
m_new = m.conditional(sh, q)

levels = jnp.split(sh.levels, sh.num_frequencies)

data = DatasetUCI("yacht")
data_dict = {"x": data.x, "y": data.y}
dataset = Dataset.from_dict(data_dict).with_format("jax", dtype=jnp.float64)
dataset = dataset.train_test_split(test_size=0.2)

param = m_new.init(
    key,
    input_dim=data.x.shape[-1],
    num_independent_processes=data.y.shape[-1],
    likelihood=lik,
    sh_features=sh,
    variational_dist=q,
)

train_step = create_training_step(m_new, dataset["train"], ("x", "y"), q, lik)
