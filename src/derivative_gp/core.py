"""Fixed-hyperparameter GP conditioning; no implicit jitter or fitting."""

from dataclasses import dataclass
from numbers import Integral, Real
import math

import numpy as np
from scipy.linalg import solve_triangular

MAX_OBSERVATIONS = 200
MAX_QUERIES = 600
MAX_CONDITION = 1e12
VARIANCE_TOL = 1e-9


class NumericalError(ValueError):
    """The requested covariance system cannot be trusted in float64."""


def _real(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is too large") from exc
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return result


@dataclass(frozen=True)
class Query:
    x: float
    order: int = 0

    def __post_init__(self):
        object.__setattr__(self, "x", _real(self.x, "x", -1e6, 1e6))
        if isinstance(self.order, (bool, np.bool_)) or not isinstance(self.order, Integral) or self.order not in (0, 1):
            raise ValueError("order must be integer 0 (value) or 1 (derivative)")
        object.__setattr__(self, "order", int(self.order))


@dataclass(frozen=True)
class Observation:
    x: float
    order: int
    value: float
    noise_std: float

    def __post_init__(self):
        query = Query(self.x, self.order)
        object.__setattr__(self, "x", query.x)
        object.__setattr__(self, "order", query.order)
        object.__setattr__(self, "value", _real(self.value, "value", -1e12, 1e12))
        object.__setattr__(self, "noise_std", _real(self.noise_std, "noise_std", 0, 1e6))
        if 0 < self.noise_std < 1e-150:
            raise ValueError("positive noise_std must be at least 1e-150")

    @property
    def query(self):
        return Query(self.x, self.order)


@dataclass(frozen=True)
class Kernel:
    amplitude: float = 1.0
    length_scale: float = 1.0

    def __post_init__(self):
        for name in ("amplitude", "length_scale"):
            object.__setattr__(self, name, _real(getattr(self, name), name, 1e-6, 1e6))


def _queries(items):
    items = tuple(items)
    if not 1 <= len(items) <= MAX_QUERIES or not all(isinstance(x, Query) for x in items):
        raise ValueError("provide 1 to 600 Query objects")
    return items


def covariance(left, right, kernel=Kernel()):
    """Covariance of f or df/dx; derivative signs follow argument order."""
    left, right = _queries(left), _queries(right)
    if not isinstance(kernel, Kernel):
        raise ValueError("kernel must be a Kernel")
    x = np.array([q.x for q in left])[:, None]
    z = np.array([q.x for q in right])[None, :]
    a = np.array([q.order for q in left])[:, None]
    b = np.array([q.order for q in right])[None, :]
    r = (x - z) / kernel.length_scale
    base = kernel.amplitude**2 * np.exp(-0.5 * r**2)
    factor = np.where((a == 0) & (b == 1), r,
                      np.where((a == 1) & (b == 0), -r,
                               np.where((a == 1) & (b == 1), 1 - r**2, 1)))
    return base * factor / kernel.length_scale**(a + b)


def fuse_replicates(observations):
    """Fuse exact (x, order) matches, assuming known independent Gaussian errors.

    This preserves conditional predictions, not the original-data likelihood.
    Near duplicates remain distinct. Contradictory exact observations are errors.
    """
    observations = tuple(observations)
    if not 1 <= len(observations) <= MAX_OBSERVATIONS or not all(isinstance(o, Observation) for o in observations):
        raise ValueError("provide 1 to 200 Observation objects")
    groups = {}
    for observation in observations:
        groups.setdefault(observation.query, []).append(observation)
    fused = []
    for query, group in groups.items():
        exact = [o for o in group if o.noise_std == 0]
        if exact:
            if any(o.value != exact[0].value for o in exact):
                raise ValueError("contradictory noiseless observations at identical (x, order)")
            fused.append(exact[0])
        else:
            minimum = min(o.noise_std for o in group)
            weights = [(minimum / o.noise_std)**2 for o in group]
            total = math.fsum(weights)
            value = math.fsum(w / total * o.value for w, o in zip(weights, group))
            noise = minimum / math.sqrt(total)
            if noise < 1e-150:
                raise NumericalError("fused noise_std is below the supported numerical range")
            fused.append(Observation(query.x, query.order, value, noise))
    return tuple(fused)


@dataclass
class Prediction:
    mean: np.ndarray
    covariance: np.ndarray
    weights: np.ndarray
    observations: tuple
    queries: tuple
    diagnostics: dict

    @property
    def variance(self):
        # Only the separate variance summary clips tiny roundoff negatives.
        return np.maximum(np.diag(self.covariance), 0.0)


def predict(observations, queries, kernel=Kernel()):
    """Return latent predictions and weights mapping the fused measurements to mean."""
    observations = tuple(observations)
    fused = fuse_replicates(observations)
    queries = _queries(queries)
    train = tuple(o.query for o in fused)
    y = np.array([o.value for o in fused])
    noise = np.array([o.noise_std for o in fused])
    a = covariance(train, train, kernel) + np.diag(noise**2)
    scale = np.sqrt(np.diag(a))
    correlation = a / scale[:, None] / scale[None, :]
    correlation = (correlation + correlation.T) / 2
    try:
        eigenvalues = np.linalg.eigvalsh(correlation)
    except np.linalg.LinAlgError as exc:
        raise NumericalError("scaled covariance eigensolver failed") from exc
    if eigenvalues[0] <= 0 or eigenvalues[-1] / eigenvalues[0] > MAX_CONDITION:
        raise NumericalError("scaled observation covariance is singular or ill-conditioned; no jitter was added")
    try:
        factor = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as exc:
        raise NumericalError("scaled covariance Cholesky failed; no jitter was added") from exc
    cross = covariance(train, queries, kernel) / scale[:, None]
    whitened_cross = solve_triangular(factor, cross, lower=True)
    whitened_y = solve_triangular(factor, y / scale, lower=True)
    mean = whitened_cross.T @ whitened_y
    weights = solve_triangular(factor.T, whitened_cross, lower=False).T / scale[None, :]
    prior = covariance(queries, queries, kernel)
    posterior = prior - whitened_cross.T @ whitened_cross
    posterior = (posterior + posterior.T) / 2
    relative_diagonal = np.diag(posterior) / np.diag(prior)
    if not (np.isfinite(mean).all() and np.isfinite(posterior).all() and np.isfinite(weights).all()):
        raise NumericalError("nonfinite conditioning result")
    if np.min(relative_diagonal) < -VARIANCE_TOL:
        raise NumericalError("posterior variance is substantially negative")
    diagnostics = {
        "input_observations": len(observations), "fused_observations": len(fused),
        "scaled_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "min_scaled_eigenvalue": float(eigenvalues[0]),
        "noise_added_beyond_declared": 0.0,
        "roundoff_negative_variances": int(np.sum(np.diag(posterior) < 0)),
        "minimum_variance_to_prior_ratio": float(np.min(relative_diagonal)),
    }
    return Prediction(mean, posterior, weights, fused, queries, diagnostics)
