"""Small reproducible training loop. Clinical labels are never read during fit."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import augment
from .data import WindowSet, assert_subject_disjoint
from .model import GlucoFMPretrainer, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 120
    batch_size: int = 128
    learning_rate: float = 1e-4
    bandwidth_learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0 or self.bandwidth_learning_rate <= 0:
            raise ValueError("Training counts and learning rates must be positive")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_batch(batch, device):
    return {name: tensor.to(device) for name, tensor in batch.items()}


def runtime_info():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": str(np.__version__)}


def fit(train_data: WindowSet, validation_data: WindowSet, output,
        config: TrainConfig | None = None, dynamics_weight=1.0):
    config = config or TrainConfig()
    assert_subject_disjoint(train_data, validation_data)
    seed_all(config.seed)
    device = torch.device(config.device)
    model = GlucoFMPretrainer(dynamics_weight=dynamics_weight).to(device)
    normal_parameters = [parameter for name, parameter in model.named_parameters()
                         if parameter.requires_grad and name != "online.filter.rho"]
    optimizer = torch.optim.AdamW([
        {"params": normal_parameters, "lr": config.learning_rate, "weight_decay": config.weight_decay},
        {"params": [model.online.filter.rho], "lr": config.bandwidth_learning_rate, "weight_decay": 0.0},
    ])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed), num_workers=0)
    validation = DataLoader(validation_data, batch_size=config.batch_size, num_workers=0)
    history = []
    best = float("inf")
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = dict.fromkeys(("loss", "contextual", "dynamics"), 0.0)
        for batch in loader:
            batch = move_batch(batch, device)
            batch["glucose"], batch["observed"] = augment(batch["glucose"], batch["observed"])
            losses = model(**batch)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            model.update_target()
            for name in totals:
                totals[name] += losses[name].item() * len(batch["glucose"])
        model.eval()
        validation_total = 0.0
        # Fixed CPU-generated hiding patterns make validation comparable across epochs.
        generator = torch.Generator().manual_seed(config.seed + 10000)
        with torch.no_grad():
            for batch in validation:
                batch = move_batch(batch, device)
                b = len(batch["glucose"])
                ratios = torch.empty(b).uniform_(0.5, 0.6, generator=generator)
                ranks = torch.rand(b, 24, generator=generator).argsort(-1).argsort(-1)
                hidden = (ranks < (ratios * 24).floor().long()[:, None]).to(device)
                value = model(**batch, hidden=hidden)["loss"]
                if not torch.isfinite(value):
                    raise RuntimeError("Non-finite validation loss")
                validation_total += value.item() * b
        record = {"epoch": epoch, **{name: value / len(train_data) for name, value in totals.items()},
                  "validation_loss": validation_total / len(validation_data),
                  "sigma_steps": model.online.filter.sigma.item()}
        history.append(record)
        improved = record["validation_loss"] < best
        best = min(best, record["validation_loss"])
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "model_config": asdict(model.online.config), "train_config": asdict(config),
            "dynamics_weight": dynamics_weight, "momentum": model.momentum,
            "epoch": epoch, "best_validation_loss": best, "history": history,
            "train_subjects": np.unique(train_data.subjects).tolist(),
            "validation_subjects": np.unique(validation_data.subjects).tolist(),
            "data_sources": [train_data.source, validation_data.source], "runtime": runtime_info(),
        }
        torch.save(checkpoint, output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch {epoch:03d} train={record['loss']:.4f} val={record['validation_loss']:.4f} "
              f"sigma={record['sigma_steps']:.3f}", flush=True)
    summary = {"seconds": time.perf_counter() - started, "config": asdict(config),
               "runtime": runtime_info(), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "total_parameters": sum(p.numel() for p in model.parameters()),
               "best_validation_loss": best, "dynamics_weight": dynamics_weight}
    (output / "training-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return load_pretrainer(output / "best.pt", device)[0], summary


def load_pretrainer(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = GlucoFMPretrainer(ModelConfig(**checkpoint["model_config"]),
                             checkpoint["momentum"], checkpoint["dynamics_weight"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint
