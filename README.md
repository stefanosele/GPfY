# $GP \mathcal{f} Y_\ell^m$

A lightweight library in JAX for Gaussian process with spherical kernels and sparse spherical harmonic inducing features.

$GP \mathcal{f} Y_\ell^m$ is based on the simple [flax.struct](https://github.com/google/flax/blob/main/flax/struct.py) dataclass. It implements [(Eleftheriadis et al. 2023)](https://arxiv.org/abs/2303.15948), which revisits the Sparse Gaussian Process with Spherical Harmonic features from [Dutordoir et al.](http://proceedings.mlr.press/v119/dutordoir20a.html), and introduces:

1. `PolynomialDecay` kernel with "continuous" depth.
2. Sparse orthogonal basis derived from `SphericalHarmonics` features with phase truncation.

## Installation

### Latest (stable) release from PyPI

```bash
pip install gpfy
```

### Development version
Alternatively, you can install the latest GitHub `develop` version.
First create a virtual enviroment via conda:
```bash
conda create -n gpfy_env python=3.10.0
conda activate gpjax_env
```

Then clone a copy of the repository to your local machine and run the setup configuration in development mode:
```bash
git clone https://github.com/stefanosele/GPfY.git
cd GPfY
make intall
```
This will automatically install all required dependencies.

Finally you can check the installation via running the tests:
```bash
make test
```
