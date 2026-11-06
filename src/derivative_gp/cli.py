"""Read a strict JSON prediction request and write a latent GP report."""

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

from .core import Kernel, Observation, Query, predict


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _object(value, allowed, required):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"invalid object fields; allowed: {', '.join(allowed)}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        with args.input.open("rb") as handle:
            data = handle.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("input exceeds 1 MiB")
        request = _object(json.loads(data, object_pairs_hook=_pairs),
                          ["observations", "queries", "kernel"], ["observations", "queries"])
        if not isinstance(request["observations"], list) or not isinstance(request["queries"], list):
            raise ValueError("observations and queries must be arrays")
        observations = [Observation(**_object(o, ["x", "order", "value", "noise_std"],
                                                ["x", "order", "value", "noise_std"])) for o in request["observations"]]
        queries = [Query(**_object(q, ["x", "order"], ["x", "order"])) for q in request["queries"]]
        kernel = Kernel(**_object(request.get("kernel", {}), ["amplitude", "length_scale"], []))
        prediction = predict(observations, queries, kernel)
        result = {"kernel": asdict(kernel), "queries": [asdict(q) for q in queries],
                  "mean": prediction.mean.tolist(), "latent_variance": prediction.variance.tolist(),
                  "diagnostics": prediction.diagnostics}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, ArithmeticError) as exc:
        print(f"derivative-gp: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
