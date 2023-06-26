# %% [markdown]
# # Continuous depth with spherical kernels via parametrized eigenvalue decay
#

# %% [markdown]
# Spherical kernels, as every spherical function, are fully defined by an angular component acting on
# the inner product between the inputs, i.e., $\boldsymbol{x}^\top \boldsymbol{z}$.
#
# We refer to this angular component as the _shape function_. The spectrum of the shape function can be
# studied to analyse the properties and the behaviour of the spherical kernels and it has been shown
# in the literature that the eigenvalues of spherical kernels decay polynomially as we move to higher
# frequencies in the spectrum. More importantly, the effective depth of the kernel can affect the
# decay rate for certain shape functions.

# %% [markdown]
# In this notebook we look into our `PolynomialDecay` kernel and compare the eigenvalue deacy to the
# classic `NTK` kernel.

import jax
import jax.numpy as jnp

# %%
import matplotlib.pyplot as plt

from gpfy.spherical import NTK, PolynomialDecay

# %%
key = jax.random.PRNGKey(0)
eigvals_ntk = {}
eigvals_poly = {}
dims = [3, 10]
depth = [5, 10, 50, 100]
beta = [2.0, 1.0, 0.1, 0.01]

k_ntk = NTK(depth=1)
k_poly = PolynomialDecay(truncation_level=50)

# %%
for dim in dims:
    eigvals_ntk[dim] = []
    eigvals_poly[dim] = []

    for d, b in zip(depth, beta):
        k_ntk = k_ntk.replace(depth=d)
        param_ntk = k_ntk.init(key, input_dim=dim, max_num_eigvals=50)
        # ignore the eigenvalue of the constant frequency
        eig = k_ntk.eigenvalues(param_ntk, jnp.arange(50))[1:]
        eigvals_ntk[dim].append(eig / eig[0])

        param_poly = k_poly.init(key, input_dim=dim)
        param_poly = param_poly.replace_param(collection=k_poly.name, beta=b)
        # ignore the eigenvalue of the constant frequency
        eig = k_poly.eigenvalues(param_poly, jnp.arange(50))[1:]
        eigvals_poly[dim].append(eig / eig[0])

# %%
marker = ["x", "o", "d", "s"]
label_poly = [r"$\beta $ = " + str(b) for b in beta]
label_ntk = [f"NTK (L = {d})" for d in depth]

fig, ax = plt.subplots(1, len(dims), figsize=(12, 4))

for a, dim in zip(ax.ravel(), dims):
    _ = [
        a.semilogy(e, "C0", marker=m, ms=5, markevery=5, label=l)
        for e, m, l in zip(eigvals_poly[dim], marker, label_poly)
    ]
    _ = [
        a.semilogy(e, "C1--", marker=m, ms=5, markevery=5, label=l)
        for e, m, l in zip(eigvals_ntk[dim], marker, label_ntk)
    ]

ax[1].legend()
_ = ax[0].set_xlabel("Ordinal eigenvalue", fontsize=12)
_ = ax[0].set_ylabel("Eigenvalue magnitude", fontsize=12)
_ = ax[0].set_title("Eigenvalue decay in 3D", fontsize=12)

_ = ax[1].set_xlabel("Ordinal eigenvalue", fontsize=12)
_ = ax[1].set_title("Eigenvalue decay in 10D", fontsize=12)

# %% [markdown]
# In the above plot we see that we can alter the decay rate of the eigenvalues, by selecting different
# $\beta$ parameter for the `PolynomiaDecay` kernel.
#
# More importantly, we can simulate an effect of "continuous" depth with our spherical kernel, which
# we can directly learn from the data, by optimising the hyper-parameter $\beta$.
