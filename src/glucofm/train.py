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
from .model import GlucoFMPretrainer, ModelConfig, objective_support
from . import provenance


SELECTION_RULE = "keep-last"
"""Checkpoint selection rule.

Validation loss is an EMA-target objective: it measures how predictable the teacher
has become, not how good the representation is. In a paired experiment over 4 seeds,
10 vs 60 epochs on identical data and folds, validation loss improved in 4/4 runs by
2.3-3.9x while downstream AP got *worse* in 4/4, -1.77 AP with 95% CI [-2.68, -0.87];
the loss-selected checkpoint was the final epoch in every run anyway. Selection
therefore keeps the last epoch and says so. The loss is still logged every epoch,
alongside two diagnostics that do track representation health: the effective rank of
the validation embeddings, and the fraction of windows that carry no weight for one
or both objectives.
"""


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


def effective_rank(features: torch.Tensor) -> float:
    """Entropy of the singular-value spectrum of the centred matrix (Roy & Vetterli).

    Singular values are normalized to sum to one and the Shannon entropy of that
    distribution is exponentiated: 1 when all variance lies in a single direction,
    the full dimensionality when the spectrum is flat. A representation that is
    quietly collapsing loses effective rank while its loss keeps falling.
    """
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("Effective rank needs at least two feature rows")
    centred = features - features.mean(0, keepdim=True)
    values = torch.linalg.svdvals(centred.to(torch.float64))
    total = values.sum()
    if total <= 0:
        return 0.0
    share = values / total
    share = share[share > 0]
    return float(torch.exp(-(share * share.log()).sum()))


def runtime_info():
    """Kept for the checkpoint/summary field of the same name in archived runs.

    Richer, stage-tagged provenance lives in `glucofm.provenance`.
    """
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
    training_provenance = {
        "stage": "training", "training_seed": config.seed,
        **provenance.environment("training"),
        "train_data": {"source": train_data.source, "windows": len(train_data),
                       "subjects": len(np.unique(train_data.subjects))},
        "validation_data": {"source": validation_data.source, "windows": len(validation_data),
                            "subjects": len(np.unique(validation_data.subjects))},
        "dynamics_weight": dynamics_weight,
    }
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
        unsupported = dict.fromkeys(("contextual", "transition", "either"), 0)
        features = []
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
                contextual_weight, transition_weight = objective_support(batch["observed"], hidden)
                unsupported["contextual"] += int((contextual_weight == 0).sum())
                unsupported["transition"] += int((transition_weight == 0).sum())
                unsupported["either"] += int(((contextual_weight == 0) | (transition_weight == 0)).sum())
                # Diagnose the representation actually used downstream: no patch hiding.
                features.append(model.online(**batch)["embedding"])
        n = len(validation_data)
        record = {"epoch": epoch, **{name: value / len(train_data) for name, value in totals.items()},
                  "validation_loss": validation_total / n,
                  "sigma_steps": model.online.filter.sigma.item(),
                  "validation_effective_rank": effective_rank(torch.cat(features)),
                  "windows_missing_contextual_weight": unsupported["contextual"] / n,
                  "windows_missing_transition_weight": unsupported["transition"] / n,
                  "windows_missing_an_objective": unsupported["either"] / n}
        history.append(record)
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "model_config": asdict(model.online.config), "train_config": asdict(config),
            "dynamics_weight": dynamics_weight, "momentum": model.momentum,
            "epoch": epoch, "selection_rule": SELECTION_RULE, "history": history,
            "train_subjects": np.unique(train_data.subjects).tolist(),
            "validation_subjects": np.unique(validation_data.subjects).tolist(),
            "data_sources": [train_data.source, validation_data.source], "runtime": runtime_info(),
            # Captured here, at the moment the checkpoint is written, so a later
            # aggregation cannot be mistaken for the environment that trained it.
            "provenance": training_provenance,
        }
        # Keep-last: `last.pt` is the selected checkpoint (see SELECTION_RULE).
        torch.save(checkpoint, output / "last.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch {epoch:03d} train={record['loss']:.4f} val={record['validation_loss']:.4f} "
              f"sigma={record['sigma_steps']:.3f} rank={record['validation_effective_rank']:.2f} "
              f"unsupported={record['windows_missing_an_objective']:.3f}", flush=True)
    validation_losses = [row["validation_loss"] for row in history]
    summary = {"seconds": time.perf_counter() - started, "config": asdict(config),
               "runtime": runtime_info(), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "total_parameters": sum(p.numel() for p in model.parameters()),
               "selection_rule": SELECTION_RULE, "selected_epoch": config.epochs,
               "final_validation_loss": validation_losses[-1],
               # Logged for monitoring only; deliberately not a selection criterion.
               "minimum_validation_loss": min(validation_losses),
               "minimum_validation_loss_epoch": 1 + validation_losses.index(min(validation_losses)),
               "final_validation_effective_rank": history[-1]["validation_effective_rank"],
               "final_windows_missing_an_objective": history[-1]["windows_missing_an_objective"],
               "dynamics_weight": dynamics_weight, "provenance": training_provenance}
    (output / "training-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return load_pretrainer(output / "last.pt", device)[0], summary


def load_pretrainer(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = GlucoFMPretrainer(ModelConfig(**checkpoint["model_config"]),
                             checkpoint["momentum"], checkpoint["dynamics_weight"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint
