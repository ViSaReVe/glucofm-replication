from datetime import datetime, timedelta
import csv

import numpy as np
import pytest
import torch

from glucofm.augment import augment, compression_profile
from glucofm.data import WindowSet, align_window, assert_subject_disjoint, from_csv, synthetic_windows
from glucofm.evaluate import probe, subject_splits, summary_features


def test_alignment_averages_duplicates_and_preserves_missingness():
    x, mask, start = align_window([0, 1, 5, 15, 1440, -1], [100, 120, 130, 140, 900, 900], 78)
    np.testing.assert_equal(x[:4], [110, 130, 0, 140])
    np.testing.assert_equal(mask[:4], [True, True, False, True])
    assert mask.sum() == 3 and start == 78


def test_nearest_binning_is_explicit_and_does_not_wrap():
    x, mask, _ = align_window([2.5, 1439], [100, 200], 0, "nearest")
    assert x[1] == 100 and mask.sum() == 1


def test_window_file_round_trip_without_pickle(tmp_path):
    data = synthetic_windows(4, 2)
    path = tmp_path / "windows.npz"
    data.save(path)
    restored = WindowSet.load(path)
    for name in ("glucose", "observed", "start", "subjects", "labels"):
        np.testing.assert_equal(getattr(data, name), getattr(restored, name))
    assert set(data[0]) == {"glucose", "observed", "start"}


def test_partition_overlap_is_rejected():
    data = synthetic_windows(4, 2)
    with pytest.raises(ValueError, match="overlap"):
        assert_subject_disjoint(data, data)
    assert_subject_disjoint(data, synthetic_windows(4, 2, seed=43))


def test_csv_segments_long_gaps_and_retains_local_start(tmp_path):
    path = tmp_path / "sample.csv"
    start = datetime(2026, 1, 1, 6, 30)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "timestamp", "glucose_mg_dl", "label"])
        for offset in (0, 60 * 48):
            for minute in range(0, 1440, 5):
                if minute == 20:
                    continue
                writer.writerow(["cohort-person-1", (start + timedelta(minutes=offset+minute)).isoformat(), 100, 0])
    data = from_csv(path)
    assert len(data) == 2 and (data.start == 78).all()
    assert (data.observed.sum(1) == 287).all()
    assert not data.observed[:, 4].any()


def test_augmentation_never_promotes_missing_values_and_is_reproducible():
    data = synthetic_windows(20, 2)
    x, mask = torch.from_numpy(data.glucose), torch.from_numpy(data.observed)
    torch.manual_seed(5)
    a, ma = augment(x, mask)
    torch.manual_seed(5)
    b, mb = augment(x, mask)
    torch.testing.assert_close(a, b)
    assert torch.equal(ma, mb) and not (ma & ~mask).any()
    assert torch.isfinite(a).all() and (a[~ma] == 0).all()
    c, mc = augment(x, mask)
    assert not torch.equal(a, c) or not torch.equal(ma, mc)


def test_subject_folds_cover_each_window_once_per_repeat():
    data = synthetic_windows(20, 3)
    counts = np.zeros((2, len(data)), dtype=int)
    for repeat, _, train, test in subject_splits(data, 5, 2):
        assert not set(data.subjects[train]) & set(data.subjects[test])
        counts[repeat, test] += 1
    assert (counts == 1).all()


def test_probe_shares_splits_and_reports_all_metrics():
    data = synthetic_windows(12, 2)
    features = summary_features(data)
    result = probe({"a": features, "b": features.copy()}, data, folds=3, repeats=1)
    assert result["summary"]["a"] == result["summary"]["b"]
    assert len(result["subject_splits"]) == 3
    assert set(result["summary"]["a"]) == {"average_precision", "roc_auc", "macro_f1"}


def test_compression_reaches_its_sampled_bottom_at_every_length():
    # linspace(-1, 1, L).abs() has no exact zero for even L, so an unrenormalised
    # envelope stopped at 0.52 / 0.4857 / 0.4667 / 0.4545 for L = 6 / 8 / 10 / 12.
    bottom = 0.40
    for length in range(6, 13):
        profile = compression_profile(length, bottom)
        assert profile.shape == (length,)
        assert abs(profile.min().item() - bottom) < 1e-6
        assert abs(profile.max().item() - 1.0) < 1e-6


def test_compression_leaves_the_already_correct_odd_lengths_unchanged():
    # Odd lengths already contained an exact zero; the renormalisation must be a
    # no-op there, so the fix cannot be masking a change to the correct case.
    for length in (7, 9, 11):
        old = 0.55 + 0.45 * torch.linspace(-1, 1, length).abs()
        torch.testing.assert_close(compression_profile(length, 0.55), old)


def test_compression_envelope_is_a_symmetric_v_anchored_at_one():
    profile = compression_profile(8, 0.4)
    torch.testing.assert_close(profile, profile.flip(0))
    assert profile[0] == profile[-1] == 1.0
    assert profile.argmin().item() in (3, 4)
