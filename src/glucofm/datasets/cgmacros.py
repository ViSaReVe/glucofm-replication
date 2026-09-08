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
from dataclasses import dataclass
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


def _at_least(column: str, minimum: float):
    """One three-valued term: True, False, or None when the panel value is absent."""
    def term(panel):
        value = panel[column]
        return None if value != value else bool(value >= minimum)
    return term


def _derived_at_least(derive, minimum: float):
    def term(panel):
        value = derive(panel)
        return None if value != value else bool(value >= minimum)
    return term


@dataclass(frozen=True)
class LabelRule:
    """A disjunction of thresholds evaluated in three-valued logic.

    Missing panel values are `None`, not `False`. A comparison against a missing
    value used to collapse to False, which turned an undecidable panel into a
    confident negative -- exactly the direction that quietly inflates a control's
    apparent performance. `evaluate` keeps the two apart.
    """

    terms: tuple
    description: str

    def evaluate(self, panel: dict[str, float]) -> bool | None:
        outcomes = [term(panel) for term in self.terms]
        # A single satisfied threshold settles an OR rule even when others are
        # missing: a known positive stays positive on a partial panel.
        if any(outcome is True for outcome in outcomes):
            return True
        # Otherwise an absent term means the rule is undecided, never negative.
        if any(outcome is None for outcome in outcomes):
            return None
        return False


# Thresholds and their clinical sources -- including the two the literature does
# not settle -- are in docs/datasets.md; none of them is invented here.
CGMACROS_LABELS = {
    "insulin_resistance": LabelRule(
        (_derived_at_least(homa_ir, 2.5),),
        "HOMA-IR >= 2.5 (no consensus cutoff exists; see docs/datasets.md)"),
    "obesity": LabelRule(
        (_at_least("BMI", 30.0),),
        "BMI >= 30 kg/m^2 (WHO obesity class I)"),
    "hyperlipidemia": LabelRule(
        (_at_least("Cholesterol", 240.0), _at_least("LDL (Cal)", 160.0),
         _at_least("Triglycerides", 200.0)),
        "NCEP ATP III 'high': total cholesterol >= 240, LDL >= 160, or "
        "triglycerides >= 200 mg/dL (the borderline-high set is also defensible; "
        "see docs/datasets.md)"),
    "diabetes": LabelRule(
        (_derived_at_least(_a1c_percent, 6.5),),
        "HbA1c >= 6.5% (ADA diagnostic threshold)"),
}


def _rule(label: str) -> LabelRule:
    if label not in CGMACROS_LABELS:
        raise ValueError(f"Unknown label {label!r}; choose from {sorted(CGMACROS_LABELS)}")
    return CGMACROS_LABELS[label]


def resolve_labels(root, label: str) -> tuple[dict[str, int], list[str]]:
    """Binary subject labels, plus the subjects the rule could not decide.

    A subject is decided positive as soon as one threshold is met, even if other
    inputs of the same rule are missing. A subject with no satisfied threshold and
    at least one missing input is *undecided* and is returned separately rather
    than being labelled negative.
    """
    rule = _rule(label)
    labels, undecided = {}, []
    for subject, panel in read_bio_panel(root).items():
        try:
            outcome = rule.evaluate(panel)
        except KeyError as error:
            raise ValueError(f"bio.csv is missing column {error} needed by {label!r}") from error
        if outcome is None:
            undecided.append(subject)
        else:
            labels[subject] = int(outcome)
    if len(set(labels.values())) < 2:
        raise ValueError(f"Label {label!r} is constant across CGMacros subjects")
    return labels, undecided


def subject_labels(root, label: str) -> dict[str, int]:
    """Decided subject-level labels; undecided subjects are excluded, not negative."""
    return resolve_labels(root, label)[0]


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


RECOVERY_POLICIES = ("slope-change", "lattice")


def _slope_change_mask(values: np.ndarray) -> np.ndarray:
    """Points that are provably real: a change of slope, or the end of a run."""
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


def sensor_readings(values, minutes=None, period=None, policy="slope-change") -> np.ndarray:
    """Estimate which points of the published one-minute series are real readings.

    **This does not recover the physical observation mask, and cannot.** CGMacros
    ships both sensors linearly interpolated onto a one-minute grid, so every minute
    carries a number while the sensors reported every five (Dexcom) or fifteen
    (Libre) minutes. Interpolation is not invertible: a real reading that happens to
    fall on the straight line between its neighbours is bit-for-bit identical to an
    interpolated point, and nothing in the file distinguishes them. What follows is
    an estimate under a stated policy, not a reconstruction.

    What *is* decidable from the published file:

    - a point where the series changes slope is a real reading;
    - the first and last present point of a run are real readings;
    - a point off the sampling lattice was not reported by the sensor.

    Everything else is ambiguous. A point on the lattice, present, and collinear
    with its neighbours may be a real reading or an interpolated filler, and the
    file does not say which. Both policies therefore select *candidates*; neither
    partitions points into readings and non-readings.

    `"slope-change"` (default) keeps only the provably-real points and excludes
    every ambiguous candidate. It never admits an interpolated value. It certainly
    discards some real readings -- a plateau produces no slope change -- but how
    many is not identifiable from these files. Its exclusions are concentrated where
    the series is flat, so whatever the true rate, the loss is signal-dependent:
    missingness becomes correlated with glucose being flat, in a model that reads
    the mask as an input channel. `docs/datasets.md` gives the measured counts of
    excluded candidates by category.

    `"lattice"` needs `minutes` (minute-of-day per row) and `period`. It additionally
    admits an ambiguous candidate when the two bracketing readings are equal. Note
    what that does and does not buy: the admitted value equals its neighbours, but
    equal endpoints do not establish what an unobserved measurement between them
    would have been -- an excursion and return within one sampling interval is
    possible. The admitted number is an interpolated estimate, not a recovered
    measurement.

    Neither policy's true mask error rate is known. The recorded real-data
    experiment used `"slope-change"`, which is why it remains the default. See
    `docs/datasets.md`; changing it changes the prepared datasets.

    Phase inference needs at least three anchors. With fewer -- a wholly flat trace,
    say -- `"lattice"` cannot locate the sampling lattice and falls back to the
    `"slope-change"` result rather than guessing a phase, so plateau candidates are
    not admitted in that case.
    """
    if policy not in RECOVERY_POLICIES:
        raise ValueError(f"Unknown recovery policy {policy!r}; choose from {RECOVERY_POLICIES}")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("sensor_readings expects a one-dimensional series")
    anchors = _slope_change_mask(values)
    if policy == "slope-change":
        return anchors
    if minutes is None or period is None:
        raise ValueError("The 'lattice' policy needs minutes and period")
    minutes = np.asarray(minutes)
    if minutes.shape != values.shape:
        raise ValueError("minutes must align with values")
    index = np.flatnonzero(anchors)
    if len(index) < 3:
        # Too few anchors to infer the lattice phase; fall back rather than guess.
        return anchors
    phase = np.bincount(minutes[index] % period).argmax()
    on_lattice = np.isfinite(values) & (minutes % period == phase)
    candidate = np.flatnonzero(on_lattice & ~anchors)
    keep = anchors.copy()
    if len(candidate):
        previous = np.searchsorted(index, candidate) - 1
        following = np.searchsorted(index, candidate)
        inside = (previous >= 0) & (following < len(index))
        # Bracketing readings equal => the segment is flat, so admitting the point
        # cannot introduce a value the sensor did not report.
        flat = np.isclose(values[index[previous[inside]]], values[index[following[inside]]],
                          atol=1e-9)
        keep[candidate[inside][flat]] = True
    return keep


def _read_sensor(path: Path, column: str, period: int,
                 recovery: str = "slope-change") -> list[tuple[datetime, float]]:
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
    minutes = np.array([time.hour * 60 + time.minute for time in times])
    keep = sensor_readings(series, minutes, period, recovery)
    return [(times[i], float(series[i])) for i in np.flatnonzero(keep)]


def to_canonical_csv(root, output, sensor: str, label: str,
                     recovery: str = "slope-change") -> Path:
    """Write one canonical CSV for a single sensor and a single binary label.

    Timestamps are already timezone-naive local clock time. CGMacros de-identifies
    by shifting whole dates by +/- 365-720 days, which moves the calendar date and
    leaves time of day intact, so the local clock index this model conditions on
    survives the shift. Nothing here converts a timezone or applies a DST rule; a
    timestamp carrying an offset is rejected rather than coerced.
    """
    if sensor not in CGMACROS_SENSORS:
        raise ValueError(f"Unknown sensor {sensor!r}; choose from {sorted(CGMACROS_SENSORS)}")
    if recovery not in RECOVERY_POLICIES:
        raise ValueError(f"Unknown recovery policy {recovery!r}; choose from {RECOVERY_POLICIES}")
    column, period = CGMACROS_SENSORS[sensor]
    root = _dataset_root(root)
    labels, undecided = resolve_labels(root, label)
    if undecided:
        # Never silent: an undecidable panel is excluded, and says so.
        print(f"{label}: {len(undecided)} subject(s) excluded as undecidable "
              f"(missing panel input, no satisfied threshold): {', '.join(undecided)}")
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
                continue  # rule undecidable for this subject; excluded, not negative
            for time, value in _read_sensor(path, column, period, recovery):
                if time.utcoffset() is not None:
                    raise ValueError(f"{path.name} carries a timezone offset; expected local naive")
                writer.writerow([subject, time.isoformat(), f"{value:g}", labels[subject]])
                written += 1
    if not written:
        raise ValueError(f"No {sensor} readings survived for label {label!r}")
    return output
