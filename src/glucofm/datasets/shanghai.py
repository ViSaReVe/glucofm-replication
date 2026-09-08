"""ShanghaiT1DM / ShanghaiT2DM adapter (Scientific Data, figshare, CC BY 4.0).

Pretraining corpus. Emits the canonical schema `data.from_csv` reads. Download
instructions are in `docs/datasets.md`; no cohort data is bundled here.

Every window this adapter produces is unlabeled (`label = -1`). The cohort's own
diabetes type is a property of the folder, not a within-cohort contrast, and
clinical labels must not reach pretraining batches in any case.
"""

import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np

COHORTS = {"T1DM": "Shanghai_T1DM", "T2DM": "Shanghai_T2DM"}
SAMPLING_MINUTES = 15
GLUCOSE_RANGE_MG_DL = (20.0, 600.0)
# mmol/L would put a CGM series here instead; used to refuse a mislabeled column.
GLUCOSE_RANGE_MMOL_L = (1.0, 35.0)


def subject_id(cohort: str, number: str) -> str:
    """Globally unique subject id. Patient numbers repeat across the two cohorts."""
    return f"shanghai{cohort}-{int(number):04d}"


def _cohort_root(root, cohort: str) -> Path:
    folder = COHORTS[cohort]
    root = Path(root)
    # Accept the cohort directory itself, the archive root, or one level below it.
    for candidate in (root, root / folder, *sorted(root.glob(f"*/{folder}"))):
        if candidate.is_dir() and candidate.name == folder:
            return candidate
    raise FileNotFoundError(f"No {folder} directory under {root}; see docs/datasets.md")


def _workbooks(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir()
                  if path.suffix.lower() in {".xls", ".xlsx"}
                  and not path.name.startswith((".", "~$")))


def _resolve_columns(columns) -> tuple[str, str]:
    """Find the date and CGM columns.

    Header spelling is not uniform in the published files: 107 of 109 ShanghaiT2DM
    workbooks name the series 'CGM (mg / dl)' and two name it 'CGM ' with no unit
    at all, so the column is matched by prefix and its units are then checked
    against the values rather than trusted from the header.
    """
    date = [name for name in columns if str(name).strip().lower() == "date"]
    glucose = [name for name in columns if str(name).strip().upper().startswith("CGM")]
    if not date or not glucose:
        raise ValueError(f"Workbook lacks a Date or CGM column; found {list(columns)}")
    return date[0], glucose[0]


def _to_mg_dl(values: np.ndarray, column: str, name: str) -> np.ndarray:
    """Confirm mg/dL from the values; never infer the unit from the header alone.

    Both published cohorts already store mg/dL (the ShanghaiT2DM series spans
    39.6-468.0). A series that reads as mmol/L is refused rather than converted,
    because a silent x18 on a misread file is indistinguishable from real data.
    """
    finite = values[np.isfinite(values)]
    if not len(finite):
        return values
    low, high = float(finite.min()), float(finite.max())
    if GLUCOSE_RANGE_MG_DL[0] <= low and high <= GLUCOSE_RANGE_MG_DL[1]:
        return values
    if GLUCOSE_RANGE_MMOL_L[0] <= low and high <= GLUCOSE_RANGE_MMOL_L[1]:
        raise ValueError(
            f"{name} column {column!r} spans [{low}, {high}], which reads as mmol/L, "
            "not the mg/dL this cohort is documented to use. Refusing to convert; "
            "check the download against docs/datasets.md.")
    raise ValueError(f"{name} column {column!r} spans [{low}, {high}], outside "
                     f"plausible CGM values in either unit")


def read_workbook(path: Path) -> list[tuple[datetime, float]]:
    """Timestamped mg/dL readings from one recording period.

    Timestamps are timezone-naive local clock time (Beijing, UTC+8). Mainland China
    has observed no daylight saving since 1991, and every recording here starts in
    2019 or later, so local clock time is continuous across each period: no DST
    gap or repeated hour can occur. Nothing is converted; an offset-aware timestamp
    is rejected rather than coerced.
    """
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("Reading Shanghai workbooks needs pandas and an Excel engine: "
                          "pip install 'glucofm-replication[datasets]'") from error
    frame = pd.read_excel(path)
    date_column, glucose_column = _resolve_columns(frame.columns)
    times = pd.to_datetime(frame[date_column], errors="coerce")
    if getattr(times.dtype, "tz", None) is not None:
        raise ValueError(f"{path.name} carries a timezone offset; expected local naive")
    values = _to_mg_dl(pd.to_numeric(frame[glucose_column], errors="coerce").to_numpy(float),
                       glucose_column, path.name)
    keep = times.notna().to_numpy() & np.isfinite(values)
    return [(time.to_pydatetime(), float(value))
            for time, value in zip(times[keep], values[keep])]


def to_canonical_csv(root, output, cohort: str = "T2DM") -> Path:
    """Write one unlabeled canonical CSV for a whole cohort.

    Files are named `<patient>_<period>_<startdate>.xls[x]`; a patient may hold
    several recording periods, which share one subject id so that subject-disjoint
    partitioning cannot be defeated by splitting one person across partitions.
    """
    if cohort not in COHORTS:
        raise ValueError(f"Unknown cohort {cohort!r}; choose from {sorted(COHORTS)}")
    directory = _cohort_root(root, cohort)
    workbooks = _workbooks(directory)
    if not workbooks:
        raise FileNotFoundError(f"No .xls/.xlsx recording periods in {directory}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "timestamp", "glucose_mg_dl", "label"])
        for path in workbooks:
            match = re.match(r"(\d+)_(\d+)_", path.name)
            if match is None:
                raise ValueError(f"Unexpected workbook name {path.name!r}; "
                                 "expected <patient>_<period>_<startdate>")
            subject = subject_id(cohort, match.group(1))
            for time, value in read_workbook(path):
                # Unlabeled: clinical labels never enter pretraining.
                writer.writerow([subject, time.isoformat(), f"{value:g}", -1])
                written += 1
    if not written:
        raise ValueError(f"No readings survived for cohort {cohort!r}")
    return output
