"""Cohort adapter invariants.

Fixtures reproduce the published layouts in miniature -- including the quirks the
adapters exist to absorb -- so the suite never needs the cohorts themselves.
"""

import csv

import numpy as np
import pytest

from glucofm.data import from_csv
from glucofm.datasets import cgmacros, shanghai

BIO_HEADER = ["subject", "Age", "Gender", "BMI", "Body weight ", "Height ", "Self-identify ",
              "A1c PDL (Lab)", "Fasting GLU - PDL (Lab)", "Insulin ", "Triglycerides",
              "Cholesterol", "HDL", "Non HDL ", "LDL (Cal)", "VLDL (Cal)", "Cho/HDL Ratio"]


def bio_row(subject, bmi=25.0, a1c=5.4, glucose=90.0, insulin="6.0",
            triglycerides=100.0, cholesterol=180.0, ldl=100.0, vldl=15.0):
    return [subject, 40, "F", bmi, 150.0, 65.0, "White", a1c, glucose, insulin,
            triglycerides, cholesterol, 60.0, 120.0, ldl, vldl, 3.0]


def write_cgmacros(root, rows, subjects=(1, 2), period=5, hours=30):
    """A CGMacros tree: bio.csv plus per-subject one-minute interpolated series."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "bio.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(BIO_HEADER)
        writer.writerows(rows)
    for number in subjects:
        folder = root / f"CGMacros-{number:03d}"
        folder.mkdir(exist_ok=True)
        minutes = hours * 60
        anchors = {t: 100.0 + 20.0 * np.sin(t / 97.0) for t in range(0, minutes, period)}
        keys = sorted(anchors)
        with (folder / f"CGMacros-{number:03d}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Timestamp", "Libre GL", "Dexcom GL", "Meal Type"])
            for minute in range(minutes):
                lo = max(k for k in keys if k <= minute)
                hi = min((k for k in keys if k >= minute), default=lo)
                # Linear interpolation between anchors, exactly as published.
                share = 0.0 if hi == lo else (minute - lo) / (hi - lo)
                value = anchors[lo] + share * (anchors[hi] - anchors[lo])
                stamp = (np.datetime64("2020-05-01T00:00:00") + np.timedelta64(minute, "m"))
                writer.writerow([str(stamp), f"{value:.6f}", f"{value:.6f}", ""])
    return root


def test_sensor_readings_recovers_anchors_and_drops_interpolated_points():
    # Five-minute anchors, linearly interpolated to one minute, with one dropout
    # where the published grid still carries a value at every minute.
    anchors = {0: 100.0, 5: 110.0, 10: 104.0, 20: 130.0, 25: 128.0}
    keys = sorted(anchors)
    series = []
    for minute in range(26):
        lo = max(k for k in keys if k <= minute)
        hi = min(k for k in keys if k >= minute)
        share = 0.0 if hi == lo else (minute - lo) / (hi - lo)
        series.append(anchors[lo] + share * (anchors[hi] - anchors[lo]))
    kept = cgmacros.sensor_readings(np.array(series))
    assert np.flatnonzero(kept).tolist() == keys
    # Minute 15 sits inside the 10->20 dropout and must not be called an observation.
    assert not kept[15]


def interpolate(anchors, span):
    """The published one-minute grid: linear interpolation between real readings."""
    keys = sorted(anchors)
    series = []
    for minute in range(span):
        lo = max(k for k in keys if k <= minute)
        hi = min((k for k in keys if k >= minute), default=lo)
        share = 0.0 if hi == lo else (minute - lo) / (hi - lo)
        series.append(anchors[lo] + share * (anchors[hi] - anchors[lo]))
    return np.array(series)


def test_slope_change_policy_drops_readings_inside_a_flat_stretch():
    # THE known bias, pinned rather than described. Five-minute readings that are
    # all 120: the one-minute series is constant, so no interior point changes
    # slope and the plateau's interior readings are not recovered.
    plateau = interpolate({0: 120.0, 5: 120.0, 10: 120.0, 15: 120.0, 20: 140.0}, 21)
    kept = cgmacros.sensor_readings(plateau)
    # Minute 15 survives only because the plateau ends there; the readings at 5 and
    # 10, interior to the flat stretch, are lost.
    assert np.flatnonzero(kept).tolist() == [0, 15, 20]
    # The lattice policy recovers them, because the bracketing readings are equal
    # and so the admitted value is the value a real reading would have carried.
    minutes = np.arange(21)
    recovered = cgmacros.sensor_readings(plateau, minutes, 5, policy="lattice")
    assert np.flatnonzero(recovered).tolist() == [0, 5, 10, 15, 20]


def test_neither_policy_admits_a_sloped_interpolated_point():
    # A real dropout: readings at 0 and 15 only, so minutes 5 and 10 are invented.
    ramp = interpolate({0: 100.0, 15: 160.0, 20: 150.0}, 21)
    minutes = np.arange(21)
    for policy, extra in (("slope-change", {}), ("lattice", {"minutes": minutes, "period": 5})):
        kept = cgmacros.sensor_readings(ramp, policy=policy, **extra)
        assert not kept[5] and not kept[10], policy
        assert kept[0] and kept[15] and kept[20], policy


def test_recovery_of_a_collinear_reading_is_undecidable_by_construction():
    # Three real readings exactly on a line are bit-for-bit identical to two real
    # readings with an interpolated point between them. No policy can separate
    # these, which is why the module claims estimation and not recovery.
    real = interpolate({0: 100.0, 5: 110.0, 10: 120.0, 15: 118.0}, 16)
    dropout = interpolate({0: 100.0, 10: 120.0, 15: 118.0}, 16)
    np.testing.assert_array_equal(real, dropout)
    assert cgmacros.sensor_readings(real)[5] == cgmacros.sensor_readings(dropout)[5]


def test_lattice_policy_needs_its_lattice_and_rejects_unknown_policies():
    values = interpolate({0: 100.0, 5: 110.0}, 6)
    with pytest.raises(ValueError, match="needs minutes and period"):
        cgmacros.sensor_readings(values, policy="lattice")
    with pytest.raises(ValueError, match="Unknown recovery policy"):
        cgmacros.sensor_readings(values, policy="interpolate-everything")


def test_sensor_readings_keeps_run_endpoints_around_absent_values():
    series = np.array([100.0, 101.0, 102.0, np.nan, np.nan, 120.0, 121.0, 122.0])
    kept = cgmacros.sensor_readings(series)
    assert kept.tolist() == [True, False, True, False, False, True, False, True]


def test_cgmacros_emits_canonical_schema_that_the_importer_accepts(tmp_path):
    root = write_cgmacros(tmp_path / "cgmacros",
                          [bio_row(1, bmi=34.0), bio_row(2, bmi=22.0)])
    path = cgmacros.to_canonical_csv(root, tmp_path / "obesity.csv", "dexcom", "obesity")
    with open(path) as stream:
        rows = list(csv.DictReader(stream))
    assert set(rows[0]) == {"subject_id", "timestamp", "glucose_mg_dl", "label"}
    labels = {row["subject_id"]: row["label"] for row in rows}
    assert labels == {"cgmacros-001": "1", "cgmacros-002": "0"}
    windows = from_csv(path, sampling="non_overlapping")
    assert len(windows) >= 2 and set(np.unique(windows.labels)) == {0, 1}


def test_cgmacros_sensors_stay_separate_partitions(tmp_path):
    root = write_cgmacros(tmp_path / "cgmacros", [bio_row(1, bmi=34.0), bio_row(2, bmi=22.0)],
                          period=5)
    dexcom = cgmacros.to_canonical_csv(root, tmp_path / "d.csv", "dexcom", "obesity")
    libre = cgmacros.to_canonical_csv(root, tmp_path / "l.csv", "libre", "obesity")
    count = lambda p: sum(1 for _ in open(p)) - 1
    # Same underlying five-minute fixture; the fifteen-minute sensor must not be
    # silently reinterpreted, and there is no mode that merges the two.
    assert count(dexcom) == count(libre)
    assert "merge" not in cgmacros.CGMACROS_SENSORS


def test_cgmacros_rejects_hba1c_that_is_not_ngsp_percent(tmp_path):
    # The published data dictionary mislabels this column as mmol/mol.
    root = write_cgmacros(tmp_path / "cgmacros", [bio_row(1, a1c=39.0), bio_row(2, a1c=48.0)])
    with pytest.raises(ValueError, match="NGSP percent"):
        cgmacros.subject_labels(root, "diabetes")


def test_cgmacros_treats_documented_error_sentinels_as_missing(tmp_path):
    root = write_cgmacros(tmp_path / "cgmacros",
                          [bio_row(1, ldl=800.0, cholesterol=100.0, triglycerides=50.0),
                           bio_row(2, ldl=200.0),
                           bio_row(3, ldl=90.0, cholesterol=150.0, triglycerides=80.0)],
                          subjects=(1, 2, 3))
    labels, undecided = cgmacros.resolve_labels(root, "hyperlipidemia")
    # LDL 800 is documented as a calculation error, so it must not read as very high
    # LDL -- and with no other threshold met the panel is undecidable, not negative.
    assert labels == {"cgmacros-002": 1, "cgmacros-003": 0}
    assert undecided == ["cgmacros-001"]


def test_partial_panel_keeps_a_known_positive_positive(tmp_path):
    # The real CGMacros case: cgmacros-012 has an unusable LDL but triglycerides
    # 1150, far over the threshold. One satisfied term settles an OR rule.
    root = write_cgmacros(tmp_path / "cgmacros",
                          [bio_row(1, ldl=800.0, cholesterol=213.0, triglycerides=1150.0),
                           bio_row(2, ldl=100.0, cholesterol=180.0, triglycerides=100.0)],
                          subjects=(1, 2))
    labels, undecided = cgmacros.resolve_labels(root, "hyperlipidemia")
    assert labels == {"cgmacros-001": 1, "cgmacros-002": 0}
    assert undecided == []


def test_insufficient_evidence_is_never_labelled_negative(tmp_path):
    # Every rule, including the single-term ones: a missing input must not collapse
    # to a confident negative. `nan >= threshold` is False, which is the trap.
    rows = [bio_row(1, bmi=float("nan"), a1c=float("nan"), insulin="",
                    ldl=float("nan"), cholesterol=100.0, triglycerides=50.0),
            bio_row(2, bmi=34.0, a1c=7.0, insulin="20.0", glucose=120.0,
                    ldl=200.0),
            bio_row(3, bmi=22.0, a1c=5.0, insulin="3.0", glucose=80.0,
                    ldl=90.0, cholesterol=150.0, triglycerides=80.0)]
    root = write_cgmacros(tmp_path / "cgmacros", rows, subjects=(1, 2, 3))
    for label in ("obesity", "diabetes", "insulin_resistance", "hyperlipidemia"):
        labels, undecided = cgmacros.resolve_labels(root, label)
        assert undecided == ["cgmacros-001"], label
        assert "cgmacros-001" not in labels, label
        assert labels["cgmacros-002"] == 1 and labels["cgmacros-003"] == 0, label


def test_undecided_subjects_never_reach_the_canonical_csv(tmp_path):
    rows = [bio_row(1, bmi=float("nan")), bio_row(2, bmi=34.0), bio_row(3, bmi=22.0)]
    root = write_cgmacros(tmp_path / "cgmacros", rows, subjects=(1, 2, 3))
    path = cgmacros.to_canonical_csv(root, tmp_path / "obesity.csv", "dexcom", "obesity")
    with open(path) as stream:
        subjects = {row["subject_id"] for row in csv.DictReader(stream)}
    assert subjects == {"cgmacros-002", "cgmacros-003"}


def test_label_rules_evaluate_in_three_valued_logic():
    rule = cgmacros.CGMACROS_LABELS["hyperlipidemia"]
    nan = float("nan")
    satisfied = {"Cholesterol": nan, "LDL (Cal)": nan, "Triglycerides": 300.0}
    undecidable = {"Cholesterol": 100.0, "LDL (Cal)": nan, "Triglycerides": 50.0}
    settled = {"Cholesterol": 100.0, "LDL (Cal)": 90.0, "Triglycerides": 50.0}
    assert rule.evaluate(satisfied) is True
    assert rule.evaluate(undecidable) is None
    assert rule.evaluate(settled) is False


def test_cgmacros_label_thresholds_sit_where_the_documentation_says(tmp_path):
    root = write_cgmacros(tmp_path / "cgmacros",
                          [bio_row(1, bmi=30.0, a1c=6.5), bio_row(2, bmi=29.9, a1c=6.4)])
    assert cgmacros.subject_labels(root, "obesity") == {"cgmacros-001": 1, "cgmacros-002": 0}
    assert cgmacros.subject_labels(root, "diabetes") == {"cgmacros-001": 1, "cgmacros-002": 0}


def test_cgmacros_parses_lab_flagged_panel_values():
    assert cgmacros._panel_number("2.5 (low)") == 2.5
    assert cgmacros._panel_number("46.4 (high)") == 46.4
    assert cgmacros._panel_number("") != cgmacros._panel_number("")  # NaN
    assert cgmacros.homa_ir({"Insulin ": 10.0, "Fasting GLU - PDL (Lab)": 81.0}) == 2.0


def test_subject_ids_cannot_collide_between_cohorts():
    assert cgmacros.subject_id(2) != shanghai.subject_id("T2DM", "2")
    assert shanghai.subject_id("T1DM", "1002") != shanghai.subject_id("T2DM", "1002")


def write_workbook(path, times, values, glucose_column="CGM (mg / dl)"):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    pd.DataFrame({"Date": times, glucose_column: values,
                  "CBG (mg / dl)": [None] * len(values)}).to_excel(path, index=False)


def shanghai_tree(tmp_path, glucose_column="CGM (mg / dl)", scale=1.0):
    pd = pytest.importorskip("pandas")
    folder = tmp_path / "shanghai" / "Shanghai_T2DM"
    folder.mkdir(parents=True)
    for patient, period in (("2000", "0"), ("2000", "1"), ("2001", "0")):
        times = pd.date_range("2021-05-13 08:00", periods=120, freq="15min")
        values = [(140.0 + 30.0 * np.sin(i / 9.0)) * scale for i in range(120)]
        write_workbook(folder / f"{patient}_{period}_20210513.xlsx", times, values, glucose_column)
    return tmp_path / "shanghai"


def test_shanghai_emits_unlabeled_rows_and_shares_ids_across_periods(tmp_path):
    root = shanghai_tree(tmp_path)
    path = shanghai.to_canonical_csv(root, tmp_path / "t2dm.csv", "T2DM")
    with open(path) as stream:
        rows = list(csv.DictReader(stream))
    assert {row["label"] for row in rows} == {"-1"}, "pretraining rows must stay unlabeled"
    # Two recording periods of one patient must land under a single subject id, or
    # subject-disjoint partitioning could be defeated by splitting one person.
    assert {row["subject_id"] for row in rows} == {"shanghaiT2DM-2000", "shanghaiT2DM-2001"}


def test_shanghai_reads_the_unlabeled_cgm_header_variant(tmp_path):
    # Two of the 109 published T2DM workbooks name the column 'CGM ' with no unit.
    root = shanghai_tree(tmp_path, glucose_column="CGM ")
    path = shanghai.to_canonical_csv(root, tmp_path / "t2dm.csv", "T2DM")
    assert sum(1 for _ in open(path)) - 1 == 360


def test_shanghai_refuses_a_series_that_reads_as_mmol_per_litre(tmp_path):
    root = shanghai_tree(tmp_path, scale=1 / 18.0)
    with pytest.raises(ValueError, match="mmol/L"):
        shanghai.to_canonical_csv(root, tmp_path / "t2dm.csv", "T2DM")


def test_shanghai_windows_carry_no_label_into_pretraining(tmp_path):
    root = shanghai_tree(tmp_path)
    path = shanghai.to_canonical_csv(root, tmp_path / "t2dm.csv", "T2DM")
    windows = from_csv(path, sampling="pretraining", seed=0)
    assert (windows.labels == -1).all()
    assert set(windows[0]) == {"glucose", "observed", "start"}
