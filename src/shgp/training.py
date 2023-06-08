from typing import Callable, Dict, Union

import jax
import jax.random as jr
import optax
from datasets import Dataset

from shgp.utils import PyTreeNode, field


class TrainState(PyTreeNode):
    step: int
    apply_fn: Callable = field(pytree_node=False)
    params: Dict = field(pytree_node=True)
    tx: optax.GradientTransformation = field(pytree_node=False)
    opt_state: optax.OptState = field(pytree_node=True)

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


PRNG = Union[jr.PRNGKeyArray, jax.Array]


def batched_data_generator(key: PRNG, dataset: Dataset, batch_size: int):
    def _reset(key: PRNG):
        key, subkey = jr.split(key)
        # return dataset.shuffle(subkey.tolist()).iter(batch_size=batch_size), key
        return dataset.shuffle().iter(batch_size=batch_size), key

    gen, key = _reset(key)
    while True:
        try:
            yield next(gen), key
        except StopIteration:
            gen, key = _reset(key)
