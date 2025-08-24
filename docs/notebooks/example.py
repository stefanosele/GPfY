# %%
import jax
import jax.numpy as jnp
import optax
from datasets import Dataset
from uci_datasets import Dataset as DatasetUCI

from gpfy.likelihoods import Gaussian
from gpfy.model import GP
from gpfy.optimization import create_training_step
from gpfy.spherical import NTK
from gpfy.spherical_harmonics import SphericalHarmonics
from gpfy.variational import VariationalDistributionTriL

key = jax.random.PRNGKey(42)

# %% [markdown]
# Let's define some global variables for controlling the behaviour of our spherical harmonic features.
# We constrain our features to use only 5 frequencies (polynomial order) and maximum up to 30 phases/harmonics within each frequency.

# %%
NUM_FREQUENCIES = 7
PHASE_TRUNCATION = 30

# %% [markdown]
# Next, let's create the relevant objects:

# %%
k = NTK(depth=5)
sh = SphericalHarmonics(num_frequencies=NUM_FREQUENCIES, phase_truncation=PHASE_TRUNCATION)
lik = Gaussian()
q = VariationalDistributionTriL()

# %% [markdown]
# Note that these object are simple dataclasses and do not hold any (hyper-)parameters in them.
# We can always initialise a parameter on every object by calling the corresponding `object.init(...)` method.

# %% [markdown]
# Our key object is a GP model, which only knows of a kernel (mean function is not supported yet).

# %%
m = GP(k)
print(m)

# %% [markdown]
# Notice that in the above GP model the `conditional_fn` tuple is empty. So it is not conditioned on any data yet.
# If we were to predict, we would return the prior mean and covariance function evaluated at some point.
# Let's make it interesting and condition our model on the spherical harmonic features.

# %%
m_new = m.conditional(sh, q)
print(m_new)

# %% [markdown]
# Now the fun part. We load some data from the UCI repository and initialise our model.

# %%
data = DatasetUCI("yacht")
data_dict = {"x": data.x, "y": data.y}
dataset = Dataset.from_dict(data_dict).with_format("np", dtype=jnp.float64)
dataset = dataset.train_test_split(test_size=0.2)

# %%
param = m_new.init(
    key,
    input_dim=data.x.shape[-1],
    num_independent_processes=data.y.shape[-1],
    likelihood=lik,
    sh_features=sh,
    variational_dist=q,
)
param.__dataclass_fields__

# %% [markdown]
# Each field in the param object holds a mapping/dictionary with the same structure.
# For instance `param.parmas` holds the following dictionary.

# %%
param.params

# %% [markdown]
# We can change the trainable status of each parameter by doing the following:

# %%
print(param._trainables["NTK"]["variance"])
param = param.set_trainable(collection=k.name, variance=False)
print(param._trainables["NTK"]["variance"])
param = param.set_trainable(collection=k.name, variance=True)
print(param._trainables["NTK"]["variance"])

# %% [markdown]
# Now let's create a training step function to call in a loop so we can optimise the model.

# %%
train_step = create_training_step(m_new, dataset["train"], ("x", "y"), q, lik)

# %% [markdown]
# We then optimise the model using the training step.

# %%
param_new, state, elbos = m_new.fit(param, train_step, optax.adam(5e-2), 2000)

# %% [markdown]
# Finally, we can predict on some new inputs.

# %%
pred_mu, pred_var = m_new.predict_diag(param_new, dataset["test"]["x"][:])

# %%
test_y = jnp.array(dataset["test"]["y"])
jnp.sqrt(jnp.mean((test_y - pred_mu) ** 2))
