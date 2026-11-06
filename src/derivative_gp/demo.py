"""Deterministic analytic function and Gaussian-law experiments; no external data."""

from dataclasses import asdict
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri

from .core import Kernel, Observation, Query, NumericalError, predict
from .audit import prior_predictive_coverage


def signal(x):
    return np.sin(1.3 * np.asarray(x)) + 0.15 * np.asarray(x)**2


def slope(x):
    return 1.3 * np.cos(1.3 * np.asarray(x)) + 0.3 * np.asarray(x)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    kernel = Kernel(1.2, 0.9)
    value_x = [-1.8, -0.4, 1.3]
    slope_x = [-1.1, 0.6]
    values = [Observation(x, 0, float(signal(x) + delta), 0.08) for x, delta in zip(value_x, [0.04, -0.03, 0.02])]
    derivatives = [Observation(x, 1, float(slope(x) + delta), 0.12) for x, delta in zip(slope_x, [-0.06, 0.05])]
    observations = values + derivatives
    grid = np.linspace(-2.2, 2.2, 181)
    queries = [Query(x) for x in grid]
    derivative_queries = [Query(x, 1) for x in grid]
    values_only = predict(values, queries, kernel)
    combined = predict(observations, queries, kernel)
    combined_slopes = predict(observations, derivative_queries, kernel)

    unit_factor = 100
    scaled_kernel = Kernel(kernel.amplitude, kernel.length_scale * unit_factor)
    scaled_queries = [Query(q.x * unit_factor, q.order) for q in queries]
    correct_units = [Observation(o.x * unit_factor, o.order, o.value / unit_factor**o.order,
                                 o.noise_std / unit_factor**o.order) for o in observations]
    wrong_units = [Observation(o.x * unit_factor, o.order, o.value,
                               o.noise_std / unit_factor**o.order) for o in observations]
    correct = predict(correct_units, scaled_queries, scaled_kernel)
    wrong = predict(wrong_units, scaled_queries, scaled_kernel)

    matched = prior_predictive_coverage(combined, kernel)
    misspecified = prior_predictive_coverage(combined, Kernel(kernel.amplitude, 0.4))
    replicated = observations + [Observation(-1.1, 1, float(slope(-1.1) + 0.03), 0.2)]
    duplicate_report = predict(replicated, queries, kernel)
    try:
        predict([Observation(0, 0, 0, 0), Observation(1e-10, 0, 0, 0)], [Query(0.5)])
    except NumericalError as exc:
        singular_message = str(exc)
    else:
        raise AssertionError("near-duplicate noiseless covariance should be rejected")

    def rmse(mean, truth):
        return float(np.sqrt(np.mean((mean - truth)**2)))

    metrics = {
        "value_only_rmse_on_analytic_function": rmse(values_only.mean, signal(grid)),
        "values_plus_derivatives_rmse_on_analytic_function": rmse(combined.mean, signal(grid)),
        "derivative_rmse_on_analytic_function": rmse(combined_slopes.mean, slope(grid)),
        "mean_latent_variance_values_only": float(np.mean(values_only.variance)),
        "mean_latent_variance_combined": float(np.mean(combined.variance)),
        "correct_unit_change_max_mean_difference": float(np.max(np.abs(correct.mean - combined.mean))),
        "correct_unit_change_max_variance_difference": float(np.max(np.abs(correct.variance - combined.variance))),
        "wrong_derivative_units_rmse": rmse(wrong.mean, signal(grid)),
        "matched_prior_predictive_minimum_coverage": matched["minimum_coverage"],
        "misspecified_prior_predictive_minimum_coverage": misspecified["minimum_coverage"],
        "misspecified_prior_predictive_grid_mean_coverage": misspecified["grid_mean_coverage"],
    }
    summary = {
        "data": "deterministic analytic function plus fixed offsets; separate analytic GP coverage experiment",
        "signal": "sin(1.3*x) + 0.15*x^2", "kernel": asdict(kernel),
        "observations": [asdict(o) for o in observations], "metrics": metrics,
        "diagnostics": combined.diagnostics, "replicate_diagnostics": duplicate_report.diagnostics,
        "near_duplicate_error": singular_message,
        "coverage_experiment": {"fitted_length_scale": 0.9, "true_length_scale_matched": 0.9,
            "true_length_scale_misspecified": 0.4, "amplitude": 1.2, "level": 0.95,
            "scope": "pointwise, integrated over a zero-mean GP and independent observation noise; fixed design and hyperparameters"},
        "grid": grid.tolist(), "truth": signal(grid).tolist(),
        "values_only_mean": values_only.mean.tolist(), "combined_mean": combined.mean.tolist(),
        "values_only_variance": values_only.variance.tolist(), "combined_variance": combined.variance.tolist(),
        "slope_mean": combined_slopes.mean.tolist(), "slope_variance": combined_slopes.variance.tolist(),
        "misspecified_pointwise_coverage": misspecified["pointwise_coverage"].tolist(),
        "matched_pointwise_coverage": matched["pointwise_coverage"].tolist(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    example = {"kernel": asdict(kernel), "observations": [asdict(o) for o in observations],
               "queries": [asdict(Query(0, 0)), asdict(Query(0, 1)), asdict(Query(1, 0))]}
    (args.output / "example.json").write_text(json.dumps(example, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)
    z = float(ndtri(0.975))
    ax = axes[0, 0]
    ax.plot(grid, signal(grid), color="0.2", ls="--", label="Analytic function")
    ax.plot(grid, values_only.mean, color="#7b8da6", label="Values only")
    ax.plot(grid, combined.mean, color="#16776b", label="Values + slopes")
    ax.fill_between(grid, combined.mean-z*np.sqrt(combined.variance), combined.mean+z*np.sqrt(combined.variance),
                    color="#16776b", alpha=0.15, label="95% pointwise model band")
    ax.scatter(value_x, [o.value for o in values], color="#b54832", s=25, zorder=3, label="Observed values")
    ax.set(xlabel="x", ylabel="f(x)", title="Independent slope measurements change the fit")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[0, 1]
    ax.plot(grid, slope(grid), color="0.2", ls="--", label="Analytic derivative")
    ax.plot(grid, combined_slopes.mean, color="#16776b", label="Posterior slope mean")
    ax.fill_between(grid, combined_slopes.mean-z*np.sqrt(combined_slopes.variance),
                    combined_slopes.mean+z*np.sqrt(combined_slopes.variance), color="#16776b", alpha=0.15)
    ax.errorbar(slope_x, [o.value for o in derivatives], yerr=z*0.12, fmt="o", color="#b54832", capsize=3, label="Slope data ± 1.96 noise SD")
    ax.set(xlabel="x", ylabel="df/dx", title="Derivatives have their own units and noise")
    ax.legend(frameon=False, fontsize=8, loc="lower center")

    ax = axes[1, 0]
    ax.plot(grid, matched["pointwise_coverage"], color="#16776b", label="True length scale = fitted 0.9")
    ax.plot(grid, misspecified["pointwise_coverage"], color="#b54832", label="True length scale = 0.4")
    ax.axhline(0.95, color="0.4", ls=":")
    ax.set(xlabel="Prediction location x", ylabel="Analytic pointwise coverage", ylim=(0, 1.02),
           title="Prior-predictive coverage depends on the kernel")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[1, 1]
    errors = [metrics["values_plus_derivatives_rmse_on_analytic_function"], rmse(correct.mean, signal(grid)),
              metrics["wrong_derivative_units_rmse"]]
    bars = ax.bar(["Original x", "x'=100x\nslopes / 100", "x'=100x\nslopes unchanged"], errors,
                  color=["#16776b", "#7b8da6", "#b54832"], width=0.6)
    ax.set_yscale("log")
    for bar, value in zip(bars, errors):
        ax.text(bar.get_x()+bar.get_width()/2, value*1.12, f"{value:.3f}", ha="center", fontsize=9)
    ax.set(ylabel="RMSE on analytic function (log axis)", title="Changing input units requires changing slopes")
    ax.set_ylim(min(errors)*0.5, max(errors)*2)
    fig.suptitle("Derivative observations: conditioning, units, and honest uncertainty", fontsize=15)
    fig.savefig(args.output / "derivative-gp.png", dpi=170)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
