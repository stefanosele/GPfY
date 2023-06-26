# %% [markdown]
# # Sparse spherical harmonics features

# %% [markdown]
# In the standard "Sparse Gaussian Processes with Spherical Harmonic Features" from Dutordoir et al.,
# we only need to select a truncation level for the frequency of the features and then enjoy the super
# fast inference due to the diagonalisation of the kernel and the precomputations.
#
# Although very efficient, it can be problematic when the dimensionality of the input is not small
# enough. This is because the number of harmonic phases in each frequency increases in a combinatorial
# way, so we practically need to restrain our features only to low-frequency components.

# %%
import jax.random as jr

from gpfy.spherical import NTK
from gpfy.spherical_harmonics import SphericalHarmonics

# %%
key = jr.PRNGKey(0)

k = NTK(depth=5)
sh = SphericalHarmonics(num_frequencies=5)

# %%
key, subkey = jr.split(key)
param_kernel = k.init(subkey, input_dim=10)
param = sh.init(key, input_dim=10, param=param_kernel)

# %%
print(
    (
        f"The 10D hyper-sphere, has {sh.num_phase_in_frequency(param)} phases at each frequency"
        f" and {sh.num_inducing(param)} inducing points, in total."
    )
)

# %% [markdown]
# Going to higher than 5 frequencies would result in a few thousands of harmonic phases, which would
# make infeasible the orthogonalisation of the basis (via a Cholesky decomposition).

# %% [markdown]
# ## Projection to a lower-dimensional hyper-sphere

# %% [markdown]
# Instead of constraining ourselves to low frequencies, alternatively, we can learn via the spherical
# kernel a linear projection to a lower-dimensional hyper-sphere.

# %%
param_kernel_proj = k.init(subkey, input_dim=10, projection_dim=4)
sh_10freq = SphericalHarmonics(num_frequencies=10)
param = sh_10freq.init(key, input_dim=4, param=param_kernel_proj)

# %%
print(
    (
        "The 10D input is now projected to the 5D hyper-sphere, which has "
        f"{sh_10freq.num_phase_in_frequency(param)} phases at each frequency"
        f" and {sh_10freq.num_inducing(param)} inducing points, in total."
    )
)

# %% [markdown]
# With this trick, we implicitly make the assumption that we can effectively learn a mapping from the
# 10D sphere down to the 4D sphere. Of course, this involves learning a projection matrix with
# `input_dim * projection_dim` hyper-parameters. On the positive side, we can now expand our inducing
# features to higher frequencies.

# %% [markdown]
# ## Phase truncation to the rescue!

# %% [markdown]
# Perhaps a better way to allow for high frequency features is to select a truncation level for the
# harmonic phases in each frequency. By doing so, we end up with a sparse basis, which is not complete
# anymore, as we potentially omit several harmonics. But, it allows us to go deeper in the frequency
# spectrum!

# %% [markdown]
# This, of course, comes at a cost. Since we do not have a complete basis anymore, we would need to
# learn the fundamental set of points and, consequently, re-orthogonalise the basis. So, effectively,
# we give away speed (no precompute anymore) for more expressive inducing features. As long as we keep
# the phase truncation to a relatively small number, we would only need to perform Cholesky
# decompositions of small `[phase_truncation, phase_truncation]` matrices.

# %%
sparse_sh = SphericalHarmonics(num_frequencies=15, phase_truncation=100)
param2 = sparse_sh.init(key, input_dim=10, param=param_kernel)

print(
    (
        f"The 10D hyper-sphere, has {sparse_sh.num_phase_in_frequency(param2)} harmonics at each"
        f" frequency and {sh.num_inducing(param2)} inducing points, in total."
    )
)

# %% [markdown]
# So, we see that we can go to much higher frequency components with roughly the same number of inducing points.
