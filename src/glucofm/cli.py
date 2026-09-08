"""Commands for CSV preparation, training, probing, and a complete toy demo."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import WindowSet, assert_subject_disjoint, from_csv, synthetic_windows
from .evaluate import embeddings, probe, summary_features
from .model import GlucoFMEncoder
from .train import TrainConfig, fit, load_pretrainer, seed_all


def save_report(report, output):
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unofficial GlucoFM v1 implementation")
    parser.add_argument("--threads", type=int, default=2, help="CPU intra-op threads")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Canonical CSV -> 24-hour windows")
    prepare.add_argument("--csv", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--binning", choices=["floor", "nearest"], default="floor")
    prepare.add_argument("--sampling", choices=["non-overlapping", "pretraining"],
                         default="non-overlapping",
                         help="non-overlapping: disjoint days, one clock phase per segment "
                              "(required for validation and downstream partitions). "
                              "pretraining: Appendix A.2 overlapping random windows.")
    prepare.add_argument("--sampling-seed", type=int, default=0,
                         help="Seed for pretraining window sampling")
    train = sub.add_parser("pretrain", help="Pretrain on subject-disjoint NPZ partitions")
    train.add_argument("--train", required=True)
    train.add_argument("--validation", required=True)
    train.add_argument("--output", default="runs/pretrain")
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    evaluate = sub.add_parser("probe", help="Binary subject-disjoint linear probe")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--output", default="probe.json")
    evaluate.add_argument("--folds", type=int, default=5)
    evaluate.add_argument("--repeats", type=int, default=10)
    demo = sub.add_parser("demo", help="Synthetic smoke experiment, not a clinical reproduction")
    demo.add_argument("--output", default="runs/synthetic-demo")
    demo.add_argument("--epochs", type=int, default=10)
    demo.add_argument("--subjects", type=int, default=64, help="Generated pretraining subjects")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    demo.add_argument("--ablation", action="store_true", help="Also train with dynamics-loss weight zero")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "prepare":
        # The sampling mode is always stated here; it is not left to a default.
        sampling = args.sampling.replace("-", "_")
        dataset = from_csv(args.csv, args.binning, sampling=sampling, seed=args.sampling_seed)
        dataset.save(args.output)
        print(f"Saved {len(dataset)} windows from {len(np.unique(dataset.subjects))} subjects "
              f"using {sampling} sampling")
    elif args.command == "pretrain":
        config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=args.device)
        fit(WindowSet.load(args.train), WindowSet.load(args.validation), args.output, config)
    elif args.command == "probe":
        model, checkpoint = load_pretrainer(args.checkpoint)
        data = WindowSet.load(args.data)
        used = set(checkpoint["train_subjects"]) | set(checkpoint["validation_subjects"])
        if used & set(data.subjects):
            raise ValueError("Probe dataset overlaps pretraining or model-selection subjects")
        seed_all(checkpoint["train_config"]["seed"])
        random_encoder = GlucoFMEncoder(model.online.config)
        features = {"glucose_summaries": summary_features(data), "random_encoder": embeddings(random_encoder, data),
                    "pretrained_encoder": embeddings(model.online, data)}
        report = probe(features, data, args.folds, args.repeats)
        report["checkpoint_epoch"] = checkpoint["epoch"]
        report["pretraining_sources"] = checkpoint["data_sources"]
        save_report(report, args.output)
    else:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        training = synthetic_windows(args.subjects, 3, args.seed)
        validation = synthetic_windows(16, 3, args.seed+1)
        evaluation = synthetic_windows(80, 3, args.seed+2)
        assert_subject_disjoint(training, validation, evaluation)
        evaluation.save(output / "evaluation.npz")
        config = TrainConfig(epochs=args.epochs, batch_size=32, seed=args.seed, device=args.device)
        model, summary = fit(training, validation, output / "full", config)
        seed_all(args.seed)
        random_encoder = GlucoFMEncoder().to(args.device)
        features = {"glucose_summaries": summary_features(evaluation),
                    "random_encoder": embeddings(random_encoder, evaluation),
                    "pretrained_encoder": embeddings(model.online, evaluation)}
        ablation_summary = None
        if args.ablation:
            ablated, ablation_summary = fit(training, validation, output / "without-dynamics", config, dynamics_weight=0.0)
            features["without_dynamics"] = embeddings(ablated.online, evaluation)
        report = probe(features, evaluation, folds=5, repeats=2, seed=args.seed)
        report.update({"experiment": "SYNTHETIC SOFTWARE SMOKE TEST; no clinical labels or clinical validity",
                       "training": summary, "ablation_training": ablation_summary,
                       "pretraining_subjects": np.unique(training.subjects).tolist(),
                       "validation_subjects": np.unique(validation.subjects).tolist(),
                       "embedding_mean_coordinate_std": float(features["pretrained_encoder"].std(0).mean())})
        save_report(report, output / "evaluation.json")


if __name__ == "__main__":
    main()
