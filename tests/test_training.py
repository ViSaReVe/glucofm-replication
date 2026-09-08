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


def write_probe(path, values):
    """A probe report shaped like evaluate.probe's, with fixed per-fold values."""
    rows = []
    for model, per_fold in values.items():
        for index, value in enumerate(per_fold):
            rows.append({"model": model, "repeat": index // 2, "fold": index % 2,
                         "average_precision": value, "roc_auc": value,
                         "macro_f1": value, "test_windows": 10})
    path.write_text(json.dumps({"fold_results": rows, "summary": {}}))


def test_aggregate_pools_over_seeds_and_pairs_on_identical_folds(tmp_path):
    from glucofm.evaluate import aggregate_probes
    for seed, offset in ((1, 0.0), (2, 0.10)):
        write_probe(tmp_path / f"taskA.seed{seed}.json",
                    {"pretrained_encoder": [0.60 + offset, 0.80 + offset],
                     "random_encoder": [0.50 + offset, 0.70 + offset]})
    report = aggregate_probes(tmp_path)
    assert report["groups"] == ["taskA"] and report["seeds"] == [1, 2]
    pooled = report["pooled"]["taskA/pretrained_encoder"]["average_precision"]
    assert pooled["mean"] == pytest.approx(0.75)          # (0.70 + 0.80) / 2
    gap = report["paired_ap_gap"]["taskA/vs_random_encoder"]
    assert gap["per_seed"] == pytest.approx([0.10, 0.10])
    assert gap["std_over_seeds"] == pytest.approx(0.0)
    assert report["overall"]["vs_random_encoder"]["groups_where_baseline_higher"] == 1


def test_aggregate_rejects_an_incomplete_seed_grid(tmp_path):
    from glucofm.evaluate import aggregate_probes
    write_probe(tmp_path / "taskA.seed1.json", {"pretrained_encoder": [0.6],
                                                "random_encoder": [0.5]})
    write_probe(tmp_path / "taskB.seed1.json", {"pretrained_encoder": [0.6],
                                                "random_encoder": [0.5]})
    write_probe(tmp_path / "taskA.seed2.json", {"pretrained_encoder": [0.6],
                                                "random_encoder": [0.5]})
    with pytest.raises(ValueError, match="Incomplete grid"):
        aggregate_probes(tmp_path)


def test_aggregate_refuses_to_pair_across_mismatched_folds(tmp_path):
    from glucofm.evaluate import aggregate_probes
    rows = [{"model": "pretrained_encoder", "repeat": 0, "fold": 0,
             "average_precision": 0.6, "roc_auc": 0.6, "macro_f1": 0.6},
            {"model": "random_encoder", "repeat": 0, "fold": 1,
             "average_precision": 0.5, "roc_auc": 0.5, "macro_f1": 0.5}]
    (tmp_path / "taskA.seed1.json").write_text(json.dumps({"fold_results": rows}))
    with pytest.raises(ValueError, match="identical folds"):
        aggregate_probes(tmp_path)
