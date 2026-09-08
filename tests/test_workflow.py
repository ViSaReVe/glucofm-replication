"""The workflow's output-directory guard, exercised without running the pipeline.

Every case here fails at the guard, before conversion or training, so the data
roots can be empty directories and nothing expensive is launched.
"""

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "real_data.sh"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="workflow script absent")


def run(output, tmp_path, environment=None, expect_failure=True):
    roots = tmp_path / "roots"
    (roots / "shanghai").mkdir(parents=True, exist_ok=True)
    (roots / "cgmacros").mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["bash", str(SCRIPT), str(roots / "shanghai"), str(roots / "cgmacros"), str(output)],
        capture_output=True, text=True,
        env={**os.environ, **{k: str(v) for k, v in (environment or {}).items()}})
    if expect_failure:
        assert result.returncode != 0, f"expected refusal, got:\n{result.stdout}"
    return result


def fingerprint(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def seeded_run_directory(tmp_path, environment=None):
    """An output directory holding a completed-looking run and its manifest."""
    output = tmp_path / "run"
    run(output, tmp_path, environment, expect_failure=False)  # writes the manifest
    (output / "probes").mkdir(parents=True, exist_ok=True)
    return output


def test_refuses_a_nonempty_output_directory_and_leaves_it_untouched(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "aggregate.json").write_text('{"precious": true}')
    (output / "probes").mkdir()
    (output / "probes" / "dexcom_diabetes.seed42.json").write_text("{}")
    before = fingerprint(output)
    result = run(output, tmp_path)
    assert "not empty" in result.stderr
    # The whole point: an existing run survives byte for byte.
    assert fingerprint(output) == before


def test_resume_without_a_manifest_is_refused(tmp_path):
    output = tmp_path / "run"
    (output / "probes").mkdir(parents=True)
    (output / "probes" / "dexcom_diabetes.seed42.json").write_text("{}")
    before = fingerprint(output)
    result = run(output, tmp_path, {"RESUME": 1})
    assert "does not exist" in result.stderr
    assert fingerprint(output) == before


def test_resume_with_a_different_configuration_is_refused(tmp_path):
    output = seeded_run_directory(tmp_path, {"SEEDS": "42 43 44"})
    before = fingerprint(output)
    # Same directory, fewer seeds: the old seeds' probes would join the aggregate.
    result = run(output, tmp_path, {"RESUME": 1, "SEEDS": "42"})
    assert "differs from the manifest" in result.stderr
    assert "seeds=" in result.stderr
    assert fingerprint(output) == before


def test_resume_refuses_probe_reports_outside_the_current_grid(tmp_path):
    # The stale-file case: manifest matches, but a report from an earlier grid is
    # still sitting in probes/ and would be pooled.
    output = seeded_run_directory(tmp_path, {"SEEDS": "42", "SENSORS": "dexcom",
                                             "LABELS": "diabetes"})
    (output / "probes" / "dexcom_diabetes.seed42.json").write_text("{}")
    (output / "probes" / "libre_obesity.seed99.json").write_text("{}")
    before = fingerprint(output)
    result = run(output, tmp_path, {"RESUME": 1, "SEEDS": "42", "SENSORS": "dexcom",
                                    "LABELS": "diabetes"})
    assert "outside this run's grid" in result.stderr
    assert "libre_obesity.seed99.json" in result.stderr
    assert fingerprint(output) == before


def test_a_changed_mask_policy_cannot_resume_into_an_existing_run(tmp_path):
    # Different prepared datasets must not land in one aggregate.
    output = seeded_run_directory(tmp_path, {"RECOVERY": "slope-change"})
    result = run(output, tmp_path, {"RESUME": 1, "RECOVERY": "lattice"})
    assert "differs from the manifest" in result.stderr
    assert "recovery=" in result.stderr


def test_a_fresh_directory_is_accepted_and_records_its_manifest(tmp_path):
    output = tmp_path / "fresh"
    run(output, tmp_path, {"SEEDS": "42"}, expect_failure=False)
    manifest = (output / "run-manifest.txt").read_text()
    assert "seeds=42" in manifest
    assert "recovery=slope-change" in manifest
    assert "legacy_validation=0" in manifest


def test_resume_with_a_matching_configuration_and_clean_probes_proceeds(tmp_path):
    output = seeded_run_directory(tmp_path, {"SEEDS": "42", "SENSORS": "dexcom",
                                             "LABELS": "diabetes"})
    (output / "probes" / "dexcom_diabetes.seed42.json").write_text("{}")
    result = run(output, tmp_path, {"RESUME": 1, "SEEDS": "42", "SENSORS": "dexcom",
                                    "LABELS": "diabetes"})
    # It gets past the guard and fails later, on the absent cohort data.
    assert "manifest verified" in result.stdout
    assert "not empty" not in result.stderr
