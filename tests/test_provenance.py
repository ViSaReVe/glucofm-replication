"""Provenance must describe the code that ran, not the directory it ran from."""

import json
import subprocess
import sys
import textwrap

from glucofm import provenance


def git(*arguments, cwd):
    subprocess.run(("git", *arguments), cwd=cwd, check=True, capture_output=True)


def make_repository(path):
    path.mkdir(parents=True, exist_ok=True)
    git("init", "-q", cwd=path)
    git("config", "user.email", "fixture@example.invalid", cwd=path)
    git("config", "user.name", "Fixture", cwd=path)
    (path / "unrelated.txt").write_text("not glucofm\n")
    git("add", "-A", cwd=path)
    git("commit", "-q", "-m", "fixture", cwd=path)
    return subprocess.run(("git", "rev-parse", "HEAD"), cwd=path, check=True,
                          capture_output=True, text=True).stdout.strip()


def test_code_revision_ignores_the_callers_working_directory(tmp_path):
    # The bug this pins: running from another checkout used to record THAT
    # repository's commit as the provenance of a GlucoFM result.
    other = tmp_path / "unrelated-repo"
    foreign_commit = make_repository(other)
    source = textwrap.dedent("""
        import json, sys
        sys.path.insert(0, sys.argv[1])
        from glucofm import provenance
        print(json.dumps(provenance.code_revision()))
    """)
    script = tmp_path / "probe_revision.py"
    script.write_text(source)
    root = str(provenance.PACKAGE_ROOT.parent)
    result = subprocess.run([sys.executable, str(script), root], cwd=other,
                            capture_output=True, text=True, check=True)
    revision = json.loads(result.stdout)
    if revision["available"]:
        assert revision["commit"] != foreign_commit
        assert str(provenance.PACKAGE_ROOT) == revision["package_path"]
    else:
        # Acceptable outcome for an installed distribution -- but it must say so,
        # rather than reporting the unrelated repository's commit.
        assert "reason" in revision
    assert foreign_commit not in json.dumps(revision)


def test_code_revision_reports_availability_and_dirty_state():
    revision = provenance.code_revision()
    assert set(revision) >= {"available"}
    if revision["available"]:
        assert len(revision["commit"]) == 40
        assert isinstance(revision["dirty"], bool)
        assert revision["package_path"].endswith("glucofm")
    else:
        assert revision["reason"]


def test_environment_records_the_stage_it_describes():
    record = provenance.environment("probe")
    assert record["stage"] == "probe"
    assert "torch" in record["packages"] and "python" in record["runtime"]


def test_file_record_hashes_content_and_survives_a_missing_file(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"glucofm")
    record = provenance.file_record(path)
    assert record["bytes"] == 7 and len(record["sha256"]) == 64
    path.write_bytes(b"glucofm!")
    assert provenance.file_record(path)["sha256"] != record["sha256"]
    assert provenance.file_record(tmp_path / "absent") == {
        "path": str(tmp_path / "absent"), "available": False}
