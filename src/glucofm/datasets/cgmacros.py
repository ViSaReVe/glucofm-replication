"""CGMacros adapter (PhysioNet, open access, CC BY-NC-SA 4.0, 45 subjects).

Downstream cohort. Emits the canonical schema `data.from_csv` reads. Download
instructions, the published-file quirks relied on here, and the clinical source
of every label threshold are in `docs/datasets.md`.

Two sensors were worn concurrently -- Dexcom G6 Pro at five minutes and FreeStyle
Libre Pro at fifteen -- and they stay separate partitions. Merging them would put
two sampling rates, two calibrations and two dropout patterns into one window and
make the observation mask meaningless.
"""

import csv
from datetime import datetime
from pathlib import Path
import re

import numpy as np

# Column name, native sampling period in minutes. Both columns are already mg/dL
# per the published data dictionary; no unit conversion is applied or needed.
CGMACROS_SENSORS = {"dexcom": ("Dexcom GL", 5), "libre": ("Libre GL", 15)}

GLUCOSE_RANGE_MG_DL = (20.0, 600.0)
A1C_PERCENT_RANGE = (3.0, 20.0)
# bio.csv sentinels documented as calculation errors, not measurements.
BIO_SENTINELS = {"LDL (Cal)": 800.0, "VLDL (Cal)": 400.0, "Cho/HDL Ratio": 400.0}


def _panel_number(text, column=None):
    """bio.csv numerics, which may carry a lab flag such as '2.5 (low)'."""
    if text is None:
        return float("nan")
    cleaned = re.sub(r"\((?:low|high)\)", "", str(text), flags=re.I).strip()
    if not cleaned:
        return float("nan")
    try:
        value = float(cleaned)
    except ValueError:
        return float("nan")
    if column in BIO_SENTINELS and value == BIO_SENTINELS[column]:
        return float("nan")  # published as an erroneous reading, not a measurement
    return value


def read_bio_panel(root) -> dict[str, dict[str, float]]:
    """Per-subject fasting panel from bio.csv, keyed by canonical subject id."""
    path = _dataset_root(root) / "bio.csv"
    if not path.exists():
        raise FileNotFoundError(f"bio.csv not found at {path}; see docs/datasets.md")
    with open(path, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "subject" not in rows[0]:
        raise ValueError("bio.csv must carry a 'subject' column; see docs/datasets.md")
    panel = {}
    for row in rows:
        subject = subject_id(int(float(row["subject"])))
        panel[subject] = {key: _panel_number(value, key) for key, value in row.items()
                          if key and key != "subject"}
    return panel


def homa_ir(panel: dict[str, float]) -> float:
    """HOMA-IR = fasting insulin (uIU/mL) x fasting glucose (mg/dL) / 405."""
    return panel["Insulin "] * panel["Fasting GLU - PDL (Lab)"] / 405.0


def _a1c_percent(panel: dict[str, float]) -> float:
    """HbA1c as NGSP percent.

    The published data dictionary labels this column mmol/mol, but its stated range
    is 4.6-8.5, which is percent; mmol/mol would put the same cohort near 27-69.
    The label is treated as a documentation error and the guard below refuses any
    file whose values do not read as percent, rather than silently misclassifying.
    """
    value = panel["A1c PDL (Lab)"]
    if np.isfinite(value) and not A1C_PERCENT_RANGE[0] <= value <= A1C_PERCENT_RANGE[1]:
        raise ValueError(
            f"HbA1c {value} is outside the NGSP percent range {A1C_PERCENT_RANGE}; "
            "this adapter assumes percent (see docs/datasets.md)")
    return value


# Each entry is (predicate, threshold description). Thresholds and their clinical
# sources -- including the two that the literature does not settle -- are in
# docs/datasets.md; none of them is invented here.
CGMACROS_LABELS = {
    "insulin_resistance": (
        lambda p: homa_ir(p) >= 2.5,
        "HOMA-IR >= 2.5 (no consensus cutoff exists; see docs/datasets.md)"),
    "obesity": (
        lambda p: p["BMI"] >= 30.0,
        "BMI >= 30 kg/m^2 (WHO obesity class I)"),
    "hyperlipidemia": (
        lambda p: (p["Cholesterol"] >= 240.0 or p["LDL (Cal)"] >= 160.0
                   or p["Triglycerides"] >= 200.0),
        "NCEP ATP III 'high': total cholesterol >= 240, LDL >= 160, or "
        "triglycerides >= 200 mg/dL (the borderline-high set is also defensible; "
        "see docs/datasets.md)"),
    "diabetes": (
        lambda p: _a1c_percent(p) >= 6.5,
        "HbA1c >= 6.5% (ADA diagnostic threshold)"),
}


def subject_labels(root, label: str) -> dict[str, int]:
    """Binary subject-level labels. Subjects with a missing input are dropped."""
    if label not in CGMACROS_LABELS:
        raise ValueError(f"Unknown label {label!r}; choose from {sorted(CGMACROS_LABELS)}")
    predicate, _ = CGMACROS_LABELS[label]
    labels = {}
    for subject, panel in read_bio_panel(root).items():
        try:
            outcome = predicate(panel)
        except KeyError as error:
            raise ValueError(f"bio.csv is missing column {error} needed by {label!r}") from error
        if outcome != outcome:  # NaN propagated from a missing panel value
            continue
        labels[subject] = int(bool(outcome))
    if len(set(labels.values())) < 2:
        raise ValueError(f"Label {label!r} is constant across CGMacros subjects")
    return labels


def subject_id(number: int) -> str:
    """Globally unique subject id; CGMacros numbers alone would collide."""
    return f"cgmacros-{int(number):03d}"


def _dataset_root(root) -> Path:
    """Accept either the extracted archive root or the CGMacros folder itself."""
    root = Path(root)
    if (root / "bio.csv").exists():
        return root
    nested = root / "CGMacros"
    if (nested / "bio.csv").exists():
        return nested
    return root


def _parse_timestamp(text: str) -> datetime:
    """Published files use ISO; the data dictionary documents M/D/Y HH:MM."""
    text = text.strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for pattern in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%y %H:%M"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized CGMacros timestamp: {text!r}")


def sensor_readings(values: np.ndarray) -> np.ndarray:
    """Recover real sensor readings from the published one-minute series.

    CGMacros ships both sensors linearly interpolated onto a one-minute grid, so
    every minute carries a number while the sensors actually reported every five
    (Dexcom) or fifteen (Libre) minutes. Ingesting the grid as-is would present
    interpolated values as observations and leave the mask almost entirely true,
    which is precisely the information the model is built to use.

    A reading is recovered where the one-minute series changes slope, plus the
    first and last present sample of each run. Points interpolated across a sensor
    dropout lie on a straight line and are correctly excluded. The rule loses a
    real reading only when three consecutive readings are exactly collinear with a
    non-zero slope; measured against the sampling lattice on the published files
    that costs about 0.1% of Libre readings, and it errs toward marking a point
    missing, which the model handles natively.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("sensor_readings expects a one-dimensional series")
    present = np.isfinite(values)
    if len(values) < 3:
        return present
    before = np.r_[False, present[:-1]]
    after = np.r_[present[1:], False]
    earlier = np.r_[np.nan, values[:-1]]
    later = np.r_[values[1:], np.nan]
    curvature = np.nan_to_num(np.abs(later - 2 * values + earlier), nan=0.0)
    run_endpoint = present & (~before | ~after)
    slope_change = present & before & after & (curvature > 1e-6)
    return run_endpoint | slope_change


def _read_sensor(path: Path, column: str) -> list[tuple[datetime, float]]:
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = {name.strip(): name for name in (reader.fieldnames or []) if name}
        if "Timestamp" not in fields or column not in fields:
            raise ValueError(f"{path.name} lacks a Timestamp or {column!r} column")
        times, values = [], []
        for row in reader:
            raw = (row[fields[column]] or "").strip()
            times.append(_parse_timestamp(row[fields["Timestamp"]]))
            values.append(float(raw) if raw else float("nan"))
    series = np.asarray(values, dtype=float)
    finite = series[np.isfinite(series)]
    if len(finite) and not (GLUCOSE_RANGE_MG_DL[0] <= finite.min()
                            and finite.max() <= GLUCOSE_RANGE_MG_DL[1]):
        raise ValueError(f"{path.name} {column!r} leaves the mg/dL range "
                         f"{GLUCOSE_RANGE_MG_DL}: [{finite.min()}, {finite.max()}]")
    keep = sensor_readings(series)
    return [(times[i], float(series[i])) for i in np.flatnonzero(keep)]


def to_canonical_csv(root, output, sensor: str, label: str) -> Path:
    """Write one canonical CSV for a single sensor and a single binary label.

    Timestamps are already timezone-naive local clock time. CGMacros de-identifies
    by shifting whole dates by +/- 365-720 days, which moves the calendar date and
    leaves time of day intact, so the local clock index this model conditions on
    survives the shift. Nothing here converts a timezone or applies a DST rule; a
    timestamp carrying an offset is rejected rather than coerced.
    """
    if sensor not in CGMACROS_SENSORS:
        raise ValueError(f"Unknown sensor {sensor!r}; choose from {sorted(CGMACROS_SENSORS)}")
    column, _ = CGMACROS_SENSORS[sensor]
    root = _dataset_root(root)
    labels = subject_labels(root, label)
    files = sorted(root.glob("CGMacros-*/CGMacros-*.csv"))
    if not files:
        raise FileNotFoundError(f"No CGMacros-*/CGMacros-*.csv under {root}; see docs/datasets.md")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "timestamp", "glucose_mg_dl", "label"])
        for path in files:
            number = int(re.search(r"CGMacros-(\d+)\.csv$", path.name).group(1))
            subject = subject_id(number)
            if subject not in labels:
                continue  # panel value needed by this label is missing for the subject
            for time, value in _read_sensor(path, column):
                if time.utcoffset() is not None:
                    raise ValueError(f"{path.name} carries a timezone offset; expected local naive")
                writer.writerow([subject, time.isoformat(), f"{value:g}", labels[subject]])
                written += 1
    if not written:
        raise ValueError(f"No {sensor} readings survived for label {label!r}")
    return output
