"""Analytic prior-predictive coverage under a specified true GP kernel."""

from numbers import Real
import numpy as np
from scipy.special import ndtr, ndtri

from .core import Kernel, Prediction, NumericalError, covariance


def prior_predictive_coverage(prediction: Prediction, true_kernel=Kernel(), level=0.95):
    """Integrate over the zero-mean true GP AND declared independent noise.

    The fitted predictor weights are fixed. This is not coverage for one fixed
    unknown function, nor conditional coverage given the observed measurements.
    Only kernel misspecification is varied; noise and observation types stay fixed.
    """
    if isinstance(level, bool) or not isinstance(level, Real) or not 0.5 <= level <= 0.9999:
        raise ValueError("level must be in [0.5, 0.9999]")
    train = tuple(o.query for o in prediction.observations)
    noise = np.array([o.noise_std for o in prediction.observations])
    true_a = covariance(train, train, true_kernel) + np.diag(noise**2)
    true_cross = covariance(train, prediction.queries, true_kernel)
    true_variance = np.diag(covariance(prediction.queries, prediction.queries, true_kernel))
    w = prediction.weights
    error_variance = true_variance - 2 * np.sum(w * true_cross.T, axis=1) + np.sum((w @ true_a) * w, axis=1)
    if np.any(error_variance <= 1e-12 * true_variance):
        raise NumericalError("coverage audit requires nondegenerate error variance")
    z = float(ndtri((1 + level) / 2))
    coverage = 2 * ndtr(z * np.sqrt(prediction.variance / error_variance)) - 1
    return {"level": float(level), "pointwise_coverage": coverage,
            "true_error_variance": error_variance,
            "minimum_coverage": float(np.min(coverage)), "grid_mean_coverage": float(np.mean(coverage))}
