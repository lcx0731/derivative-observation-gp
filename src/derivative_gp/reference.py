"""Independent derivative kernel check using high-order Gauss-Hermite quadrature.

For Z~N(0,1), k(x,z)=amplitude^2 E[cos(Z*(x-z)/ell)]. Differentiate
the cosine features in their respective arguments, without using core.covariance.
This is a numerical small-case reference, not an exact-arithmetic oracle.
"""

import math
import numpy as np
from numpy.polynomial.hermite import hermgauss


def spectral_covariance(left, right, kernel, nodes=100):
    roots, weights = hermgauss(nodes)
    frequency = math.sqrt(2) * roots / kernel.length_scale
    weights = weights / math.sqrt(math.pi)

    def features(query):
        angle = frequency * query.x
        if query.order == 0:
            return np.cos(angle), np.sin(angle)
        return -frequency * np.sin(angle), frequency * np.cos(angle)

    result = np.empty((len(left), len(right)))
    for i, a in enumerate(left):
        ca, sa = features(a)
        for j, b in enumerate(right):
            cb, sb = features(b)
            result[i, j] = kernel.amplitude**2 * np.sum(weights * (ca * cb + sa * sb))
    return result
