"""Observed-only grid alignment, portable window files, and synthetic smoke data."""

from bisect import bisect_left
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import STEPS


def align_window(minutes, glucose, start_index: int, binning: str = "floor"):
    """Appendix C.1: elapsed minutes -> 288 bins; duplicate observations averaged.

    start_index is the absolute local-clock index of the window start. Binning
    rounds half up when requested; it never fills an unobserved bin by interpolation.
    """
    minutes, glucose = np.asarray(minutes, float), np.asarray(glucose, float)
    if minutes.ndim != 1 or minutes.shape != glucose.shape:
        raise ValueError("minutes and glucose must be equal-length one-dimensional arrays")
    if binning not in {"floor", "nearest"} or not 0 <= start_index < STEPS:
        raise ValueError("Invalid binning mode or clock index")
    valid = np.isfinite(minutes) & np.isfinite(glucose) & (minutes >= 0) & (minutes < 1440)
    indices = np.floor(minutes[valid] / 5 + (0.5 if binning == "nearest" else 0)).astype(int)
    inside = indices < STEPS
    indices, values = indices[inside], glucose[valid][inside]
    sums = np.bincount(indices, weights=values, minlength=STEPS)
    counts = np.bincount(indices, minlength=STEPS)
    observed = counts > 0
    aligned = np.divide(sums, counts, out=np.zeros(STEPS), where=observed).astype(np.float32)
    return aligned, observed, int(start_index)


@dataclass
class WindowSet(Dataset):
    glucose: np.ndarray
    observed: np.ndarray
    start: np.ndarray
    subjects: np.ndarray
    labels: np.ndarray
    source: str = "unspecified"

    def __post_init__(self):
        n = len(self.glucose)
        if n == 0 or self.glucose.shape != (n, STEPS) or self.observed.shape != (n, STEPS):
            raise ValueError("WindowSet requires nonempty [N, 288] arrays")
        if any(a.shape != (n,) for a in (self.start, self.subjects, self.labels)):
            raise ValueError("Metadata must have one entry per window")
        if self.observed.dtype != np.bool_ or not self.observed.any(1).all():
            raise ValueError("Each window needs at least one observation and a boolean mask")
        if not np.isfinite(self.glucose[self.observed]).all():
            raise ValueError("Observed glucose must be finite")
        if not np.issubdtype(self.start.dtype, np.integer) or ((self.start < 0) | (self.start >= STEPS)).any():
            raise ValueError("Clock indices must be integers in [0, 288)")
        if not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("Labels must be integers (-1 when unlabeled)")
        self.subjects = self.subjects.astype(str)
        if (self.subjects == "").any():
            raise ValueError("Subject IDs must be nonempty")
        for subject in np.unique(self.subjects):
            if len(np.unique(self.labels[self.subjects == subject])) != 1:
                raise ValueError("Subject-level labels must be consistent across that subject's windows")

    def __len__(self):
        return len(self.glucose)

    def __getitem__(self, index):
        # Labels/IDs deliberately stay out of the pretraining tensor batch.
        return {
            "glucose": torch.as_tensor(self.glucose[index], dtype=torch.float32),
            "observed": torch.as_tensor(self.observed[index], dtype=torch.bool),
            "start": torch.as_tensor(self.start[index], dtype=torch.long),
        }

    def subset(self, indices):
        return WindowSet(*(a[indices] for a in
                           (self.glucose, self.observed, self.start, self.subjects, self.labels)),
                         source=self.source)

    def save(self, path):
        np.savez_compressed(path, glucose=self.glucose, observed=self.observed, start=self.start,
                            subjects=self.subjects, labels=self.labels, source=np.array(self.source))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(*(data[key] for key in ("glucose", "observed", "start", "subjects", "labels")),
                       source=str(data["source"].item()))


def assert_subject_disjoint(*datasets: WindowSet):
    seen = set()
    for dataset in datasets:
        subjects = set(dataset.subjects.tolist())
        if seen & subjects:
            raise ValueError("Subject overlap between partitions: " + ", ".join(sorted(seen & subjects)[:5]))
        seen |= subjects


# Appendix A.2 pretraining sampling: consecutive window starts advance by a stride
# drawn uniformly over whole grid steps, so successive windows overlap by 20-80%
# of a day. Both endpoints of that overlap range map to the same stride range.
OVERLAP_RANGE = (0.2, 0.8)
STRIDE_STEPS = (round(STEPS * (1 - OVERLAP_RANGE[1])), round(STEPS * (1 - OVERLAP_RANGE[0])))


def _window_at(segment, times, beginning, binning, subject):
    """One 24-hour window starting at `beginning`; None when it holds no observation."""
    lo = bisect_left(times, beginning)
    hi = bisect_left(times, beginning + timedelta(hours=24))
    day = segment[lo:hi]
    if not day:
        return None
    elapsed = [(row[0] - beginning).total_seconds() / 60 for row in day]
    start = (beginning.hour * 60 + beginning.minute) // 5
    x, m, s = align_window(elapsed, [row[1] for row in day], start, binning)
    return x, m, s, subject, day[0][2]


def from_csv(path, binning="floor", *, sampling="non_overlapping", seed=0) -> WindowSet:
    """Prepare 24-hour windows from a canonical local-time CGM CSV.

    Required: subject_id,timestamp,glucose_mg_dl. Optional: label (-1 if absent).
    Timestamps must be timezone-naive local ISO times. Dataset adapters must handle
    timezone/DST conventions and unit conversion explicitly before this function.

    `sampling="non_overlapping"` (the default, and the only mode for anything
    downstream) tiles each segment in disjoint 24-hour windows. Because that stride
    is exactly one day, every window in a segment also inherits the segment's start
    clock index. `sampling="pretraining"` instead advances by a seeded random stride
    of 58-230 grid steps, giving the overlapping windows and mixed circadian phases
    Appendix A.2 asks for. `seed` makes that draw reproducible for a run.
    """
    if sampling not in {"non_overlapping", "pretraining"}:
        raise ValueError("sampling must be 'non_overlapping' or 'pretraining'")
    rng = np.random.default_rng(seed) if sampling == "pretraining" else None
    records = {}
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"subject_id", "timestamp", "glucose_mg_dl"} <= set(reader.fieldnames or []):
            raise ValueError("CSV needs subject_id,timestamp,glucose_mg_dl")
        for row in reader:
            time = datetime.fromisoformat(row["timestamp"])
            if time.utcoffset() is not None:
                raise ValueError("Normalize timestamps to local timezone-naive times before importing")
            value = float(row["glucose_mg_dl"]) if row["glucose_mg_dl"] else float("nan")
            label = int(row.get("label") or -1)
            if np.isfinite(value):
                records.setdefault(row["subject_id"], []).append((time, value, label))
    windows = []
    for subject, rows in sorted(records.items()):
        rows.sort(key=lambda item: item[0])
        if len({row[2] for row in rows}) != 1:
            raise ValueError(f"Inconsistent subject label: {subject}")
        boundaries = [0] + [i for i in range(1, len(rows))
                            if rows[i][0] - rows[i-1][0] > timedelta(hours=1)] + [len(rows)]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            segment = rows[lo:hi]
            times = [row[0] for row in segment]
            beginning = segment[0][0]
            # Require recording support through the last 15 minutes of each day.
            while segment[-1][0] >= beginning + timedelta(hours=24, minutes=-15):
                window = _window_at(segment, times, beginning, binning, subject)
                if window is not None:
                    windows.append(window)
                stride = STEPS if rng is None else int(rng.integers(*STRIDE_STEPS, endpoint=True))
                beginning += timedelta(minutes=5 * stride)
    if not windows:
        raise ValueError("No complete 24-hour windows survived gap segmentation")
    x, m, s, subject, labels = zip(*windows)
    return WindowSet(np.stack(x), np.stack(m), np.array(s), np.array(subject), np.array(labels),
                     source=f"canonical CSV: {Path(path).name}; {sampling} sampling"
                            + (f", seed={seed}" if rng is not None else ""))


def split_subjects(path, output_a, output_b, fraction_b=0.2, seed=0) -> tuple[Path, Path]:
    """Split a canonical CSV into two subject-disjoint canonical CSVs.

    Partitioning by subject *before* windowing is what lets each side choose its own
    sampling mode: overlapping windows for the pretraining side, non-overlapping for
    a validation or downstream side. Splitting after windowing forces one mode on
    both, which is how the recorded 2026-09-07 run ended up with overlapping
    validation windows (see docs/implementation-decisions.md, assumption 11).
    """
    if not 0 < fraction_b < 1:
        raise ValueError("fraction_b must lie strictly between 0 and 1")
    rows = []
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        if "subject_id" not in fields:
            raise ValueError("Canonical CSV needs a subject_id column")
        rows = list(reader)
    subjects = sorted({row["subject_id"] for row in rows})
    if len(subjects) < 2:
        raise ValueError("Need at least two subjects to split")
    order = np.random.default_rng(seed).permutation(len(subjects))
    count = max(1, min(len(subjects) - 1, round(fraction_b * len(subjects))))
    chosen = {subjects[i] for i in order[:count]}
    paths = []
    for output, wanted in ((output_a, False), (output_b, True)):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in rows if (row["subject_id"] in chosen) == wanted)
        paths.append(output)
    return tuple(paths)


def synthetic_windows(subjects=64, days=3, seed=42) -> WindowSet:
    """Toy regimes, NOT clinical diagnoses or validated simulated physiology.

    Subjects share baseline/response traits across days. Different seeds identify
    disjoint generated subjects. Sensor density is independent of the class label.
    """
    if subjects < 2 or days < 1:
        raise ValueError("At least two subjects and one day required")
    rng = np.random.default_rng(seed)
    xs, masks, starts, ids, labels = [], [], [], [], []
    for subject in range(subjects):
        label = subject % 2
        baseline = rng.normal(108 + 7 * label, 12)
        response = rng.normal(35 + 9 * label, 9)
        recovery = rng.uniform(12, 24) + 5 * label
        for _ in range(days):
            start = int(rng.integers(0, STEPS))
            clock = (start + np.arange(STEPS)) % STEPS
            x = baseline + rng.normal(0, 4) + 8 * np.sin(2 * np.pi * clock / STEPS)
            for meal in (96, 156, 228):
                elapsed = (clock - meal - rng.integers(-9, 10)) % STEPS
                bump = (1 - np.exp(-elapsed / 4)) * np.exp(-elapsed / recovery)
                x += response * rng.uniform(0.7, 1.3) * bump
            x += rng.normal(0, 4, STEPS)
            mask = rng.random(STEPS) > 0.04
            if rng.random() < 0.25:
                mask &= np.arange(STEPS) % 3 == rng.integers(3)
            gap_start = int(rng.integers(0, STEPS - 6))
            mask[gap_start:gap_start+6] = False
            xs.append(np.where(mask, x, 0).astype(np.float32))
            masks.append(mask)
            starts.append(start)
            ids.append(f"synthetic-{seed}-{subject}")
            labels.append(label)
    return WindowSet(np.stack(xs), np.stack(masks), np.array(starts), np.array(ids),
                     np.array(labels), source=f"synthetic regimes; seed={seed}")
