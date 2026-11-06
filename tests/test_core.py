import math

import numpy as np
import pytest
from scipy.linalg import solve

from derivative_gp import Kernel, NumericalError, Observation, Query, covariance, predict
from derivative_gp.audit import prior_predictive_coverage
from derivative_gp.core import fuse_replicates
from derivative_gp.reference import spectral_covariance


@pytest.mark.parametrize("seed", range(8))
def test_mixed_kernel_matches_independent_fourier_quadrature(seed):
    rng = np.random.default_rng(seed)
    left = [Query(x, i % 2) for i, x in enumerate(rng.uniform(-1.5, 1.5, 6))]
    right = [Query(x, i % 2) for i, x in enumerate(rng.uniform(-1.5, 1.5, 5))]
    kernel = Kernel(1.3, 0.8)
    assert covariance(left, right, kernel) == pytest.approx(spectral_covariance(left, right, kernel), abs=2e-12)


def test_derivative_signs_and_second_mixed_derivative_by_finite_differences():
    x, z, h = 0.4, -0.2, 1e-4
    kernel = Kernel(1.7, 0.9)
    def scalar(u, v):
        return kernel.amplitude**2 * math.exp(-(u-v)**2 / (2*kernel.length_scale**2))
    first_x = (scalar(x+h, z) - scalar(x-h, z)) / (2*h)
    first_z = (scalar(x, z+h) - scalar(x, z-h)) / (2*h)
    mixed = (scalar(x+h, z+h) - scalar(x+h, z-h) - scalar(x-h, z+h) + scalar(x-h, z-h)) / (4*h*h)
    assert covariance([Query(x, 1)], [Query(z, 0)], kernel)[0, 0] == pytest.approx(first_x, abs=2e-8)
    assert covariance([Query(x, 0)], [Query(z, 1)], kernel)[0, 0] == pytest.approx(first_z, abs=2e-8)
    assert covariance([Query(x, 1)], [Query(z, 1)], kernel)[0, 0] == pytest.approx(mixed, abs=4e-8)


def test_symmetry_psd_and_at_same_point_value_slope_independence():
    queries = [Query(x, order) for x in [-1, 0.1, 1.2] for order in [0, 1]]
    kernel = Kernel(2, 0.7)
    matrix = covariance(queries, queries, kernel)
    assert matrix == pytest.approx(matrix.T, abs=1e-14)
    assert np.linalg.eigvalsh(matrix)[0] > 0
    assert matrix[0, 1] == 0
    assert matrix[0, 0] == 4
    assert matrix[1, 1] == pytest.approx(4 / 0.7**2)


@pytest.mark.parametrize("seed", range(6))
def test_posterior_matches_spectral_kernel_and_direct_unscaled_solve(seed):
    rng = np.random.default_rng(seed)
    obs = [Observation(x, i % 2, rng.normal(), 0.15) for i, x in enumerate([-1.2, -0.5, 0.2, 1.1])]
    queries = [Query(-0.8, 0), Query(0.4, 1), Query(0.9, 0)]
    kernel = Kernel(1.2, 0.8)
    actual = predict(obs, queries, kernel)
    train = [o.query for o in obs]
    a = spectral_covariance(train, train, kernel) + np.eye(4) * 0.15**2
    b = spectral_covariance(train, queries, kernel)
    c = spectral_covariance(queries, queries, kernel)
    expected_mean = b.T @ solve(a, np.array([o.value for o in obs]), assume_a="pos")
    expected_cov = c - b.T @ solve(a, b, assume_a="pos")
    assert actual.mean == pytest.approx(expected_mean, abs=5e-12)
    assert actual.covariance == pytest.approx(expected_cov, abs=5e-12)
    assert actual.weights @ np.array([o.value for o in actual.observations]) == pytest.approx(actual.mean, abs=1e-12)


def test_noiseless_values_and_slopes_interpolate():
    obs = [Observation(-0.8, 0, 1, 0), Observation(0.2, 1, -0.4, 0), Observation(1, 0, -1, 0)]
    prediction = predict(obs, [o.query for o in obs])
    assert prediction.mean == pytest.approx([o.value for o in obs], abs=1e-12)
    assert prediction.covariance == pytest.approx(np.zeros((3, 3)), abs=1e-12)
    assert prediction.diagnostics["noise_added_beyond_declared"] == 0


def test_adding_independent_derivatives_reduces_covariance_in_psd_order():
    values = [Observation(-1, 0, 0.2, 0.1), Observation(1, 0, 0.5, 0.1)]
    mixed = values + [Observation(0.3, 1, -0.2, 0.1)]
    queries = [Query(x, order) for x in [-0.8, 0.1, 0.7] for order in [0, 1]]
    original = predict(values, queries)
    augmented = predict(mixed, queries)
    assert np.linalg.eigvalsh(original.covariance - augmented.covariance)[0] >= -1e-12
    assert np.all(augmented.variance <= original.variance + 1e-12)
    assert np.linalg.eigvalsh(augmented.covariance)[0] >= -1e-12


@pytest.mark.parametrize("unit_factor", [0.01, 10, 1000])
def test_coordinate_unit_change_preserves_means_and_covariances(unit_factor):
    obs = [Observation(-1, 0, 0.2, 0.1), Observation(0.3, 1, -0.2, 0.04), Observation(1, 0, 0.5, 0.1)]
    queries = [Query(x, order) for x in [-0.8, 0.1, 0.7] for order in [0, 1]]
    kernel = Kernel(1.2, 0.7)
    original = predict(obs, queries, kernel)
    changed = predict([Observation(o.x * unit_factor, o.order,
                                   o.value / unit_factor**o.order, o.noise_std / unit_factor**o.order) for o in obs],
                      [Query(q.x * unit_factor, q.order) for q in queries],
                      Kernel(kernel.amplitude, kernel.length_scale * unit_factor))
    scale = np.array([unit_factor**q.order for q in queries])
    assert changed.mean * scale == pytest.approx(original.mean, abs=1e-12)
    assert changed.covariance * scale[:, None] * scale[None, :] == pytest.approx(original.covariance, abs=1e-12)
    assert changed.diagnostics["scaled_condition_number"] == pytest.approx(original.diagnostics["scaled_condition_number"], rel=1e-12)


def test_precision_fusion_preserves_original_noisy_joint_posterior():
    obs = [Observation(0.2, 1, y, s) for y, s in [(0.4, 0.1), (-0.3, 0.2), (0.7, 0.3)]]
    obs += [Observation(-0.7, 0, 0.3, 0.15)]
    queries = [Query(x, order) for x in [-0.8, 0, 0.9] for order in [0, 1]]
    actual = predict(obs, queries)
    train = [o.query for o in obs]
    a = covariance(train, train) + np.diag([o.noise_std**2 for o in obs])
    b = covariance(train, queries)
    mean = b.T @ solve(a, np.array([o.value for o in obs]), assume_a="pos")
    post = covariance(queries, queries) - b.T @ solve(a, b, assume_a="pos")
    assert actual.mean == pytest.approx(mean, abs=2e-12)
    assert actual.covariance == pytest.approx(post, abs=2e-12)
    assert actual.diagnostics["fused_observations"] == 2


@pytest.mark.parametrize("repeats", [1, 2, 10, 100])
def test_repeated_noisy_value_matches_scalar_normal_update(repeats):
    obs = [Observation(0, 0, 1, 0.2)] * repeats
    actual = predict(obs, [Query(0)])
    assert actual.mean[0] == pytest.approx(repeats / (repeats + 0.04), abs=1e-13)
    assert actual.variance[0] == pytest.approx(0.04 / (repeats + 0.04), abs=1e-13)


def test_equal_exact_replicates_collapse_and_dominate_noisy_ones():
    fused = fuse_replicates([Observation(0, 0, 1, 0), Observation(0, 0, 1, 0), Observation(0, 0, -100, 1)])
    assert fused == (Observation(0, 0, 1, 0),)
    with pytest.raises(ValueError, match="contradictory"):
        fuse_replicates([Observation(0, 0, 1, 0), Observation(0, 0, 2, 0)])


def test_near_duplicate_singular_system_is_rejected_without_jitter():
    with pytest.raises(NumericalError, match="no jitter"):
        predict([Observation(0, 0, 0, 0), Observation(1e-10, 0, 0, 0)], [Query(0.5)])


def test_correct_prior_predictive_coverage_and_covariance_agree():
    obs = [Observation(-1, 0, 0, 0.1), Observation(0, 1, 0, 0.1), Observation(1, 0, 0, 0.1)]
    kernel = Kernel(1.1, 0.8)
    prediction = predict(obs, [Query(x, order) for x in [-0.8, 0.1, 0.7] for order in [0, 1]], kernel)
    report = prior_predictive_coverage(prediction, kernel)
    assert report["true_error_variance"] == pytest.approx(prediction.variance, abs=1e-12)
    assert report["pointwise_coverage"] == pytest.approx(np.full(6, 0.95), abs=1e-12)


def test_misspecified_coverage_error_variance_matches_joint_linear_transform():
    obs = [Observation(-1, 0, 0.3, 0.1), Observation(0.1, 1, -0.2, 0.1), Observation(1, 0, 0.1, 0.1)]
    queries = [Query(-0.5), Query(0.5, 1)]
    prediction = predict(obs, queries, Kernel(1, 1.2))
    true = Kernel(1, 0.7)
    joint = spectral_covariance([o.query for o in obs] + queries, [o.query for o in obs] + queries, true)
    joint[:3, :3] += np.eye(3) * 0.01
    transform = np.column_stack([-prediction.weights, np.eye(2)])
    expected = np.diag(transform @ joint @ transform.T)
    assert prior_predictive_coverage(prediction, true)["true_error_variance"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("x,order", [(True, 0), (math.nan, 0), (1e7, 0), (0, True), (0, 2), (0, 1.0)])
def test_bad_queries(x, order):
    with pytest.raises(ValueError):
        Query(x, order)


@pytest.mark.parametrize("value,noise", [(math.inf, 0.1), (True, 0.1), (0, -1), (0, math.nan), (0, 1e-200)])
def test_bad_observations(value, noise):
    with pytest.raises(ValueError):
        Observation(0, 0, value, noise)


@pytest.mark.parametrize("amplitude,length", [(0, 1), (1, -1), (True, 1), (1, math.inf), (1, 1e7)])
def test_bad_kernels(amplitude, length):
    with pytest.raises(ValueError):
        Kernel(amplitude, length)


def test_size_limits_and_noninformative_coverage_cases():
    for obs in [[], [Observation(0, 0, 1, 0.1)] * 201, [1]]:
        with pytest.raises(ValueError):
            predict(obs, [Query(0)])
    for queries in [[], [Query(0)] * 601, [1]]:
        with pytest.raises(ValueError):
            predict([Observation(0, 0, 1, 0.1)], queries)
    p = predict([Observation(0, 0, 1, 0)], [Query(0)])
    with pytest.raises(NumericalError, match="nondegenerate"):
        prior_predictive_coverage(p)
    for level in [True, math.nan, 1, 0.1]:
        with pytest.raises(ValueError):
            prior_predictive_coverage(p, level=level)
