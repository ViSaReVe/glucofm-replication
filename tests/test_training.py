import json

import numpy as np
import torch

from glucofm.data import synthetic_windows
from glucofm.evaluate import embeddings
from glucofm.train import TrainConfig, fit, load_pretrainer


def test_train_save_reload_and_frozen_features(tmp_path):
    torch.set_num_threads(2)
    train = synthetic_windows(4, 1, seed=1)
    validation = synthetic_windows(4, 1, seed=2)
    model, summary = fit(train, validation, tmp_path, TrainConfig(epochs=1, batch_size=4, seed=3))
    restored, checkpoint = load_pretrainer(tmp_path / "best.pt")
    assert checkpoint["epoch"] == 1 and checkpoint["train_subjects"] == sorted(set(train.subjects))
    assert summary["trainable_parameters"] < summary["total_parameters"]
    before = {name: p.detach().clone() for name, p in model.online.named_parameters()}
    expected = embeddings(model.online, validation)
    actual = embeddings(restored.online, validation)
    np.testing.assert_allclose(expected, actual, atol=0, rtol=0)
    assert all(torch.equal(before[name], p) for name, p in model.online.named_parameters())
    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history) == 1 and np.isfinite(history[0]["validation_loss"])
