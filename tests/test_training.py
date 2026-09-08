import json

import numpy as np
import pytest
import torch

from glucofm.data import synthetic_windows
from glucofm.evaluate import embeddings
from glucofm.train import SELECTION_RULE, TrainConfig, effective_rank, fit, load_pretrainer


def test_train_save_reload_and_frozen_features(tmp_path):
    torch.set_num_threads(2)
    train = synthetic_windows(4, 1, seed=1)
    validation = synthetic_windows(4, 1, seed=2)
    model, summary = fit(train, validation, tmp_path, TrainConfig(epochs=1, batch_size=4, seed=3))
    restored, checkpoint = load_pretrainer(tmp_path / "last.pt")
    assert checkpoint["epoch"] == 1 and checkpoint["train_subjects"] == sorted(set(train.subjects))
    assert summary["trainable_parameters"] < summary["total_parameters"]
    before = {name: p.detach().clone() for name, p in model.online.named_parameters()}
    expected = embeddings(model.online, validation)
    actual = embeddings(restored.online, validation)
    np.testing.assert_allclose(expected, actual, atol=0, rtol=0)
    assert all(torch.equal(before[name], p) for name, p in model.online.named_parameters())
    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history) == 1 and np.isfinite(history[0]["validation_loss"])


def test_selection_keeps_the_last_epoch_and_still_logs_validation_loss(tmp_path):
    torch.set_num_threads(2)
    train = synthetic_windows(4, 1, seed=1)
    validation = synthetic_windows(4, 1, seed=2)
    _, summary = fit(train, validation, tmp_path, TrainConfig(epochs=3, batch_size=4, seed=3))
    _, checkpoint = load_pretrainer(tmp_path / "last.pt")
    # Validation loss measures how predictable the EMA teacher has become, so it does
    # not choose the checkpoint; the rule is recorded rather than left implicit.
    assert not (tmp_path / "best.pt").exists()
    assert checkpoint["selection_rule"] == SELECTION_RULE == "keep-last"
    assert checkpoint["epoch"] == len(checkpoint["history"]) == 3
    assert summary["selected_epoch"] == 3
    # The loss is still logged every epoch, and its minimum is reported, not used.
    assert all(np.isfinite(row["validation_loss"]) for row in checkpoint["history"])
    assert summary["minimum_validation_loss"] <= summary["final_validation_loss"]
    assert 1 <= summary["minimum_validation_loss_epoch"] <= 3


def test_epoch_diagnostics_report_rank_and_objective_coverage(tmp_path):
    torch.set_num_threads(2)
    train = synthetic_windows(4, 1, seed=1)
    validation = synthetic_windows(6, 1, seed=2)
    fit(train, validation, tmp_path, TrainConfig(epochs=1, batch_size=4, seed=3))
    record = json.loads((tmp_path / "history.json").read_text())[0]
    assert 1.0 <= record["validation_effective_rank"] <= 128.0
    for name in ("windows_missing_contextual_weight", "windows_missing_transition_weight",
                 "windows_missing_an_objective"):
        assert 0.0 <= record[name] <= 1.0
    assert record["windows_missing_an_objective"] >= max(
        record["windows_missing_contextual_weight"], record["windows_missing_transition_weight"])


def test_effective_rank_counts_spread_directions():
    line = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert abs(effective_rank(line) - 1.0) < 1e-9
    plane = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    assert abs(effective_rank(plane) - 2.0) < 1e-9
    # A spectrum concentrated on one direction scores near one even in many dimensions.
    skewed = torch.randn(64, 16) * torch.tensor([1e4] + [1e-4] * 15)
    assert effective_rank(skewed) < 1.01


def probe_report(values, folds=2, repeats=1, cv_seed=42, training_seed=None,
                 source="canonical CSV: fixture.csv", splits=None, dataset_sha=None):
    """A well-formed probe report: one finite row per model per declared fold."""
    rows = []
    for model, per_fold in values.items():
        for index, value in enumerate(per_fold):
            rows.append({"model": model, "repeat": index // folds, "fold": index % folds,
                         "average_precision": value, "roc_auc": value,
                         "macro_f1": value, "test_windows": 10})
    if splits is None:
        splits = [{"repeat": r, "fold": f, "train_subjects": ["a"], "test_subjects": ["b"]}
                  for r in range(repeats) for f in range(folds)]
    report = {"fold_results": rows, "summary": {}, "folds": folds, "repeats": repeats,
              "seed": cv_seed, "cv_seed": cv_seed, "source": source,
              "subject_splits": splits}
    if training_seed is not None:
        report["training_seed"] = training_seed
    if dataset_sha is not None:
        report["provenance"] = {"dataset": {"sha256": dataset_sha}}
    return report


def write_probe(path, values, **kwargs):
    path.write_text(json.dumps(probe_report(values, **kwargs)))


def two_seed_grid(tmp_path, overrides=None):
    overrides = overrides or {}
    for seed, offset in ((1, 0.0), (2, 0.10)):
        write_probe(tmp_path / f"taskA.seed{seed}.json",
                    {"pretrained_encoder": [0.60 + offset, 0.80 + offset],
                     "random_encoder": [0.50 + offset, 0.70 + offset]},
                    training_seed=seed, **overrides.get(seed, {}))


def test_aggregate_pools_over_seeds_and_pairs_on_identical_folds(tmp_path):
    from glucofm.evaluate import aggregate_probes
    two_seed_grid(tmp_path)
    report = aggregate_probes(tmp_path)
    assert report["groups"] == ["taskA"] and report["training_seeds"] == [1, 2]
    pooled = report["pooled"]["taskA/pretrained_encoder"]["average_precision"]
    assert pooled["mean"] == pytest.approx(0.75)          # (0.70 + 0.80) / 2
    gap = report["paired_ap_gap"]["taskA/vs_random_encoder"]
    assert gap["per_seed"] == pytest.approx([0.10, 0.10])
    assert gap["std_over_seeds"] == pytest.approx(0.0)
    assert report["overall"]["vs_random_encoder"]["groups_where_baseline_higher"] == 1
    assert report["source_provenance"]["training_seed_unverified"] == []


def test_aggregate_rejects_an_incomplete_seed_grid(tmp_path):
    from glucofm.evaluate import aggregate_probes
    for name, seed in (("taskA", 1), ("taskB", 1), ("taskA", 2)):
        write_probe(tmp_path / f"{name}.seed{seed}.json",
                    {"pretrained_encoder": [0.6, 0.7], "random_encoder": [0.5, 0.6]},
                    training_seed=seed)
    with pytest.raises(ValueError, match="Incomplete grid"):
        aggregate_probes(tmp_path)


def test_aggregate_rejects_a_report_that_does_not_cover_its_declared_grid(tmp_path):
    # The failure the review reproduced: five folds and ten repeats declared, one
    # fold actually present, previously averaged as though complete.
    from glucofm.evaluate import aggregate_probes
    thin = probe_report({"pretrained_encoder": [0.6], "random_encoder": [0.5]},
                        folds=2, repeats=1, training_seed=1)
    thin["folds"], thin["repeats"] = 5, 10
    (tmp_path / "taskA.seed1.json").write_text(json.dumps(thin))
    with pytest.raises(ValueError, match="does not cover the declared"):
        aggregate_probes(tmp_path)


def test_aggregate_rejects_duplicate_and_non_finite_rows(tmp_path):
    from glucofm.evaluate import aggregate_probes, validate_probe_report
    duplicated = probe_report({"pretrained_encoder": [0.6, 0.7],
                               "random_encoder": [0.5, 0.6]}, training_seed=1)
    duplicated["fold_results"].append(dict(duplicated["fold_results"][0]))
    with pytest.raises(ValueError, match="duplicate row"):
        validate_probe_report(duplicated)
    broken = probe_report({"pretrained_encoder": [0.6, float("nan")],
                           "random_encoder": [0.5, 0.6]}, training_seed=1)
    with pytest.raises(ValueError, match="non-finite"):
        validate_probe_report(broken)


def test_aggregate_rejects_seeds_evaluated_on_different_subjects(tmp_path):
    from glucofm.evaluate import aggregate_probes
    other = [{"repeat": 0, "fold": f, "train_subjects": ["x"], "test_subjects": ["y"]}
             for f in range(2)]
    two_seed_grid(tmp_path, {2: {"splits": other}})
    with pytest.raises(ValueError, match="different subject assignments"):
        aggregate_probes(tmp_path)


def test_aggregate_rejects_seeds_probed_on_different_datasets(tmp_path):
    from glucofm.evaluate import aggregate_probes
    two_seed_grid(tmp_path, {2: {"source": "canonical CSV: other.csv"}})
    with pytest.raises(ValueError, match="differs from seed 1 in 'source'"):
        aggregate_probes(tmp_path)
    for path in tmp_path.glob("*.json"):
        path.unlink()
    two_seed_grid(tmp_path, {1: {"dataset_sha": "a" * 64}, 2: {"dataset_sha": "b" * 64}})
    with pytest.raises(ValueError, match="probed a different dataset"):
        aggregate_probes(tmp_path)


def test_aggregate_rejects_a_mismatched_cv_seed(tmp_path):
    from glucofm.evaluate import aggregate_probes
    two_seed_grid(tmp_path, {2: {"cv_seed": 7}})
    with pytest.raises(ValueError, match="differs from seed 1 in 'cv_seed'"):
        aggregate_probes(tmp_path)


def test_aggregate_rejects_a_filename_that_contradicts_its_training_seed(tmp_path):
    from glucofm.evaluate import aggregate_probes
    two_seed_grid(tmp_path)
    write_probe(tmp_path / "taskA.seed2.json",
                {"pretrained_encoder": [0.7, 0.9], "random_encoder": [0.6, 0.8]},
                training_seed=99)
    with pytest.raises(ValueError, match="filename says training seed 2"):
        aggregate_probes(tmp_path)


def test_aggregate_flags_archived_reports_without_training_seed_metadata(tmp_path):
    # Archived reports predate the metadata. They must still aggregate, and the
    # unverifiable filename claim must be stated rather than quietly accepted.
    from glucofm.evaluate import aggregate_probes
    two_seed_grid(tmp_path)
    for seed in (1, 2):
        path = tmp_path / f"taskA.seed{seed}.json"
        report = json.loads(path.read_text())
        del report["training_seed"]
        path.write_text(json.dumps(report))
    report = aggregate_probes(tmp_path)
    assert report["source_provenance"]["training_seed_unverified"] == [
        "taskA.seed1.json", "taskA.seed2.json"]
    assert report["pooled"]["taskA/pretrained_encoder"]["average_precision"]["mean"] \
        == pytest.approx(0.75)


def test_aggregate_refuses_to_pair_across_mismatched_folds(tmp_path):
    from glucofm.evaluate import aggregate_probes
    rows = [{"model": "pretrained_encoder", "repeat": 0, "fold": 0,
             "average_precision": 0.6, "roc_auc": 0.6, "macro_f1": 0.6},
            {"model": "random_encoder", "repeat": 0, "fold": 1,
             "average_precision": 0.5, "roc_auc": 0.5, "macro_f1": 0.5}]
    (tmp_path / "taskA.seed1.json").write_text(json.dumps(
        {"fold_results": rows, "folds": 2, "repeats": 1, "seed": 42,
         "training_seed": 1, "source": "s"}))
    with pytest.raises(ValueError, match="does not cover the declared"):
        aggregate_probes(tmp_path)
