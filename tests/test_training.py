import json

import numpy as np
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
