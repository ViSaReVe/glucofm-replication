"""Evidence-chain metadata: which code, environment and inputs produced a result.

Provenance is captured **when an artifact is created** -- at training time into the
checkpoint, at probe time into the probe report -- and separately again at
aggregation time. Recording only the aggregation environment would describe the
machine that pooled the numbers as though it had produced them.
"""

import hashlib
import importlib
import platform
from pathlib import Path
import subprocess

PACKAGE_ROOT = Path(__file__).resolve().parent
TRACKED_PACKAGES = ("torch", "numpy", "sklearn", "scipy", "pandas", "openpyxl", "xlrd")


def _git(*arguments, cwd):
    return subprocess.run(("git", *arguments), cwd=cwd, capture_output=True,
                          text=True, check=True, timeout=30).stdout.strip()


def code_revision() -> dict:
    """Revision of the *package* checkout, never the caller's working directory.

    Resolving this from the process CWD is wrong: importing glucofm from an
    unrelated repository would record that repository's commit as the provenance of
    a GlucoFM result. The lookup is anchored at this file instead, and reports
    itself unavailable when the package is not inside a checkout -- for an installed
    distribution, say -- rather than guessing.
    """
    try:
        root = Path(_git("rev-parse", "--show-toplevel", cwd=PACKAGE_ROOT)).resolve()
    except (OSError, subprocess.SubprocessError):
        return {"available": False,
                "reason": "package is not inside a git checkout",
                "package_path": str(PACKAGE_ROOT)}
    if root not in PACKAGE_ROOT.parents and root != PACKAGE_ROOT:
        return {"available": False, "reason": "resolved repository does not contain the package",
                "package_path": str(PACKAGE_ROOT), "repository_root": str(root)}
    try:
        commit = _git("rev-parse", "HEAD", cwd=PACKAGE_ROOT)
        status = _git("status", "--porcelain", cwd=PACKAGE_ROOT).splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": f"git failed: {error}",
                "package_path": str(PACKAGE_ROOT), "repository_root": str(root)}
    modified = [line for line in status if not line.startswith("??")]
    return {
        "available": True, "commit": commit,
        # A dirty tree means the commit does not describe the code that ran.
        "dirty": bool(modified), "modified_paths": len(modified),
        "untracked_paths": sum(1 for line in status if line.startswith("??")),
        "repository_root": str(root), "package_path": str(PACKAGE_ROOT),
    }


def package_versions() -> dict:
    versions = {}
    for module in TRACKED_PACKAGES:
        try:
            versions[module] = str(importlib.import_module(module).__version__)
        except Exception:
            versions[module] = None
    return versions


def runtime() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform()}


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path) -> dict:
    """Identity of one consumed or produced file."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "available": False}
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def environment(stage: str) -> dict:
    """The code, interpreter and libraries in force at one pipeline stage."""
    return {"stage": stage, "code": code_revision(),
            "packages": package_versions(), "runtime": runtime()}
