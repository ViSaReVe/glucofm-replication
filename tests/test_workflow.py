"""The workflow's output-directory guard, exercised without running the pipeline.

Every case here is settled by the guard, before conversion or training, so the data
roots can be empty directories and nothing expensive is launched. A stub `python` on
PATH records any invocation, which is how "no pipeline stage started" is asserted
rather than assumed.
"""

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "real_data.sh"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="workflow script absent")


def stub_python(tmp_path):
    """A `python` that records its arguments instead of running anything."""
    directory = tmp_path / "stub-bin"
    directory.mkdir(exist_ok=True)
    marker = tmp_path / "python-invocations.log"
    stub = directory / "python"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MARKER"\nexit 0\n')
    stub.chmod(0o755)
    return directory, marker


def run(output, tmp_path, environment=None, expect_failure=True):
    roots = tmp_path / "roots"
    (roots / "shanghai").mkdir(parents=True, exist_ok=True)
    (roots / "cgmacros").mkdir(parents=True, exist_ok=True)
    directory, marker = stub_python(tmp_path)
    marker.write_text("")  # per-invocation, so earlier calls do not leak in
    result = subprocess.run(
        ["bash", str(SCRIPT), str(roots / "shanghai"), str(roots / "cgmacros"), str(output)],
        capture_output=True, text=True,
        env={**os.environ, "PATH": f"{directory}{os.pathsep}{os.environ['PATH']}",
             "MARKER": str(marker),
             **{k: str(v) for k, v in (environment or {}).items()}})
    result.invocations = marker.read_text().splitlines() if marker.exists() else []
    if expect_failure:
        assert result.returncode != 0, f"expected refusal, got:\n{result.stdout}"
    return result


def fingerprint(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def completed_run(tmp_path, environment=None):
    """An output directory that looks like a finished run: manifest and artifacts."""
    output = tmp_path / "run"
    run(output, tmp_path, environment, expect_failure=False)  # writes the manifest
    for relative, content in (
            ("runs/seed42/last.pt", b"checkpoint bytes"),
            ("runs/seed42.log", b"epoch 120 train=0.0180 val=0.0164\n"),
            ("probes/dexcom_diabetes.seed42.json", b'{"fold_results": []}'),
            ("prepared/pretrain.npz", b"prepared windows"),
            ("aggregate.json", b'{"overall": {}}')):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
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
    assert fingerprint(output) == before
    assert result.invocations == []


def test_resume_is_refused_and_starts_no_pipeline_stage(tmp_path):
    """The regression for the removed RESUME mode.

    RESUME=1 used to pass the guard whenever the manifest matched, after which the
    script re-entered every stage: `pretrain` initialises a fresh model and optimizer
    -- there is no continuation path -- and `tee` truncated the previous run's log.
    A matching manifest is exactly the case that used to be accepted, so that is the
    case pinned here.
    """
    output = completed_run(tmp_path, {"SEEDS": "42", "SENSORS": "dexcom",
                                      "LABELS": "diabetes"})
    before = fingerprint(output)
    assert "run-manifest.txt" in before and "runs/seed42.log" in before
    result = run(output, tmp_path, {"RESUME": 1, "SEEDS": "42", "SENSORS": "dexcom",
                                    "LABELS": "diabetes"})
    assert "RESUME is not supported" in result.stderr
    # Nothing ran: no stage banner, and the stub python was never invoked, so
    # pretrain cannot have restarted.
    assert result.invocations == []
    assert "== 1." not in result.stdout and "== 3." not in result.stdout
    # Every artifact survives byte for byte -- checkpoint, log and probe included.
    assert fingerprint(output) == before
    assert (output / "runs/seed42.log").read_bytes() != b""


def test_resume_is_refused_even_for_an_empty_output_directory(tmp_path):
    # RESUME must never be a way through the guard, whatever the directory holds.
    output = tmp_path / "empty"
    output.mkdir()
    result = run(output, tmp_path, {"RESUME": 1})
    assert "RESUME is not supported" in result.stderr
    assert result.invocations == []
    assert list(output.iterdir()) == []


def test_a_matching_manifest_does_not_license_a_rerun(tmp_path):
    # Without the RESUME flag the same directory is still refused: a matching
    # manifest identifies a run, it does not authorise overwriting it.
    output = completed_run(tmp_path, {"SEEDS": "42"})
    before = fingerprint(output)
    result = run(output, tmp_path, {"SEEDS": "42"})
    assert "not empty" in result.stderr
    assert result.invocations == []
    assert fingerprint(output) == before


def test_a_fresh_directory_is_accepted_and_records_its_manifest(tmp_path):
    output = tmp_path / "fresh"
    result = run(output, tmp_path, {"SEEDS": "42"}, expect_failure=False)
    manifest = (output / "run-manifest.txt").read_text()
    assert "seeds=42" in manifest
    assert "recovery=slope-change" in manifest
    assert "legacy_validation=0" in manifest
    # It proceeds into the pipeline, which is what the stub records.
    assert any("glucofm.cli" in call for call in result.invocations)


def test_an_existing_empty_directory_is_accepted(tmp_path):
    output = tmp_path / "empty"
    output.mkdir()
    run(output, tmp_path, {"SEEDS": "42"}, expect_failure=False)
    assert (output / "run-manifest.txt").exists()
