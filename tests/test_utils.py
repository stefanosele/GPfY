import numpy as np
import pytest
from scipy.integrate import quad

from shgp.gegenbauer import gegenbauer
from shgp.harmonics.utils import weight_func

alpha_list = [0.5, 1.5, 3.0, 10.0]
freq_list = [0, 1, 2, 5, 10]


@pytest.mark.parametrize("n", freq_list)
@pytest.mark.parametrize("alpha", alpha_list)
def test_orthogonality_of_the_gegenbauer_under_the_weight_function(n, alpha):
    """Check the Gegenbauer orthogonality wrt to the weight function."""
    dim = int(2 * alpha + 2)

    def integrand(x):
        return gegenbauer(n, alpha, x) * gegenbauer(n + 1, alpha, x) * weight_func(x, dim)

    res, tol = quad(integrand, -1, 1)

    np.testing.assert_allclose(res, 0, atol=tol)
