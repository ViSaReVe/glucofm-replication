"""Cohort adapters that emit the canonical CSV schema `from_csv` reads.

Each adapter is responsible for resolving that cohort's units, its timestamp and
timezone convention, and its subject identifiers before anything reaches the
importer. `data.from_csv` requires timezone-naive local time and mg/dL, and will
not repair either. Download instructions and every label threshold are recorded
in `docs/datasets.md`; adapters never bundle or redistribute cohort data.
"""

from . import cgmacros, shanghai
from .cgmacros import CGMACROS_LABELS, CGMACROS_SENSORS

__all__ = ["cgmacros", "shanghai", "CGMACROS_LABELS", "CGMACROS_SENSORS"]
