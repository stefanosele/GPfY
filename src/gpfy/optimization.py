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

from collections.abc import Iterator
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from datasets import Dataset
from jaxtyping import Array

from gpfy.likelihoods import Likelihood
from gpfy.model import GP
from gpfy.param import Param
from gpfy.training import TrainState
from gpfy.typing import TrainingData, TrainingDataDict, TrainStepFn
from gpfy.variational import VariationalDistribution


def batched_data_generator(dataset: Dataset, batch_size: int) -> Iterator[TrainingDataDict]:
    """
    Create a data-generator for minibatching using a HuggingFace Dataset.

    NOTE: Currently there is no way to control the seed for the random shuffling via passing a
    `PRNG` to `Dataset`.

    Args:
        dataset: The dataset to wrap in the generator.
        batch_size: The size of the minibatch.

    Yields:
        A `TrainingDataDict` with the arrays the size of `batch_size`.
    """

    def _reset() -> Dataset:
        """Reset the iterator once it is exhausted."""
        # currently shuffling with a PRNG in a jit function is not working.
        # key, subkey = jr.split(key)
        # return dataset.shuffle(subkey.tolist()).iter(batch_size=batch_size), key
        return dataset.shuffle().iter(batch_size=batch_size, drop_last_batch=True)

    gen = _reset()
    while True:
        try:
            yield next(gen)
        except StopIteration:
            gen = _reset()


def create_training_step(
    model: GP,
    dataset: Dataset,
    dataset_xy_keys: Tuple[str, str],
    q: VariationalDistribution,
    lik: Likelihood,
    batch_size=None,
) -> TrainStepFn:
    """
    Create a training step callable that we can use in a `jax.lax.scan` to optimise the model.

    Args:
        model: A GP model.
        dataset: A HuggingFace dataset to fit with the GP model.
        dataset_xy_keys: The keys to the dataset dictionary that correspond to the `X` and `Y` data.
        q: The variational distribution.
        lik: The likelihood.
        batch_size: The size of the minibatch. Default to None, which means we fit the entire data.

    Returns:
        A training step callable that expects a `TrainState` input to call during optimisation.
    """
    x_key, y_key = dataset_xy_keys
    if batch_size:
        N = dataset.num_rows
        data_gen = batched_data_generator(dataset, batch_size=batch_size)

        @jax.jit
        def train_step(
            state: TrainState, dummy_input_for_scan: Optional[Array] = None
        ) -> Tuple[TrainState, float]:
            data = next(data_gen)
            X, Y = data[x_key], data[y_key]

            def loss_fn(params):
                free_param = state.apply_fn(params)
                return -elbo(free_param.constrained(), model, q, lik, (X, Y), dataset_size=N)

            loss_val, grads = jax.value_and_grad(loss_fn)(state.params)
            new_state = state.apply_gradients(grads=grads)

            return new_state, loss_val

        return train_step
    else:

        @jax.jit
        def train_step(
            state: TrainState, dummy_input_for_scan: Optional[Array] = None
        ) -> Tuple[TrainState, float]:
            X, Y = dataset[x_key], dataset[y_key]

            def loss_fn(params):
                free_param = state.apply_fn(params)
                return -elbo(free_param.constrained(), model, q, lik, (X, Y))

            loss_val, grads = jax.value_and_grad(loss_fn)(state.params)
            new_state = state.apply_gradients(grads=grads)

            return new_state, loss_val

        return train_step


@jax.jit
def elbo(
    param: Param,
    m: GP,
    q: VariationalDistribution,
    lik: Likelihood,
    train_data: TrainingData,
    dataset_size: int = -1,
) -> float:
    """
    The variational lower bound for inference in sparse Gaussian process.

    ELBO = Σ𝔼_q[log𝒩(p(y|f)] - KL[q(u)||p(u)]

    Args:
        param: A `Param` initialised with the model.
        m: The GP model.
        q: The variational distribution.
        lik: The likelihood.
        train_data: A tuple containing the training data.
        dataset_size: The full dataset size, in case we do minibatchgin. Defaults to -1, for no
        minibatching inference.

    Returns:
        The evidence lower bound.
    """
    X, Y = train_data
    fmu, fvar = m.predict_diag(param, X)
    var_exp = lik.variational_expectations(param, fmu, fvar)(Y)
    scale = jax.lax.cond(dataset_size > 0, lambda: dataset_size, lambda: jnp.shape(X)[0])
    KL = q.prior_KL(param)
    return scale * jnp.mean(var_exp, -1) - KL
