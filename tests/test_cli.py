import json
import subprocess
import sys

import pytest


def run(path):
    return subprocess.run([sys.executable, "-m", "derivative_gp.cli", str(path)],
                          capture_output=True, text=True, timeout=15)


def test_cli_outputs_latent_prediction_without_modifying_input(tmp_path):
    request = {"observations": [{"x": 0, "order": 1, "value": 2, "noise_std": 0.1}],
               "queries": [{"x": 0, "order": 0}, {"x": 0, "order": 1}]}
    path = tmp_path / "example.json"
    raw = json.dumps(request)
    path.write_text(raw)
    result = run(path)
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["mean"] == pytest.approx([0, 2/1.01])
    assert data["latent_variance"] == pytest.approx([1, 0.01/1.01])
    assert path.read_text() == raw


@pytest.mark.parametrize("input_text", [
    '{}', '[]', '{"observations":[],"queries":[],"other":1}',
    '{"observations":[],"queries":[],"queries":[]}',
    '{"observations":{},"queries":[]}',
    '{"observations":[{"x":NaN,"order":0,"value":1,"noise_std":0.1}],"queries":[]}',
    '{"observations":[{"x":0,"order":0,"value":1,"noise_std":0.1,"extra":1}],"queries":[]}',
    '{broken', ' ' * (1024*1024 + 1),
], ids=["missing", "array", "unknown", "duplicate", "not-array", "nan", "unknown-nested", "broken", "large"])
def test_cli_rejects_invalid_json_requests(tmp_path, input_text):
    path = tmp_path / "bad.json"
    path.write_text(input_text)
    result = run(path)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "derivative-gp:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_file_and_usage_errors(tmp_path):
    assert run(tmp_path / "missing.json").returncode == 1
    assert subprocess.run([sys.executable, "-m", "derivative_gp.cli"], capture_output=True).returncode == 2
