"""Commands for CSV preparation, training, probing, and a complete toy demo."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .data import (
    WindowSet, assert_subject_disjoint, from_csv, split_subjects, synthetic_windows,
)
from .datasets import cgmacros, shanghai
from .evaluate import aggregate_probes, embeddings, probe, summary_features
from .model import GlucoFMEncoder
from .train import TrainConfig, fit, load_pretrainer, package_versions, runtime_info, seed_all


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit():
    """Records which revision produced a result; None outside a checkout."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
    convert = sub.add_parser("convert", help="Cohort archive -> canonical CSV")
    convert.add_argument("--dataset", required=True, choices=["cgmacros", "shanghai"])
    convert.add_argument("--root", required=True, help="Extracted cohort directory")
    convert.add_argument("--output", required=True)
    convert.add_argument("--sensor", choices=sorted(cgmacros.CGMACROS_SENSORS),
                         help="CGMacros only; the two sensors stay separate partitions")
    convert.add_argument("--label", choices=sorted(cgmacros.CGMACROS_LABELS),
                         help="CGMacros only; thresholds are documented in docs/datasets.md")
    convert.add_argument("--cohort", choices=sorted(shanghai.COHORTS), default="T2DM",
                         help="Shanghai only; emitted unlabeled for pretraining")
    convert.add_argument("--recovery", choices=list(cgmacros.RECOVERY_POLICIES),
                         default="slope-change",
                         help="CGMacros only; how to estimate real readings from the "
                              "published interpolated one-minute grid. slope-change "
                              "(the recorded experiment's policy) keeps only "
                              "provably-real points; lattice also recovers plateaus. "
                              "See docs/datasets.md.")
    partition = sub.add_parser(
        "partition", help="Split a canonical CSV into subject-disjoint CSVs")
    partition.add_argument("--csv", required=True)
    partition.add_argument("--output-a", required=True)
    partition.add_argument("--output-b", required=True)
    partition.add_argument("--fraction-b", type=float, default=0.2,
                           help="Share of subjects in B")
    partition.add_argument("--seed", type=int, default=0)
    split = sub.add_parser("split", help="Split an NPZ into subject-disjoint partitions")
    split.add_argument("--data", required=True)
    split.add_argument("--output-a", required=True)
    split.add_argument("--output-b", required=True)
    split.add_argument("--fraction-b", type=float, default=0.2, help="Share of subjects in B")
    split.add_argument("--seed", type=int, default=0)
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
    summarise = sub.add_parser(
        "aggregate", help="Pool probe reports named '<group>.seed<N>.json'")
    summarise.add_argument("--probes", required=True, help="Directory of probe reports")
    summarise.add_argument("--output", required=True)
    summarise.add_argument("--baseline", default="pretrained_encoder")
    summarise.add_argument("--provenance", nargs="*", default=(),
                           help="Files to hash into the report for provenance")
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
    elif args.command == "convert":
        if args.dataset == "cgmacros":
            if not args.sensor or not args.label:
                parser.error("--sensor and --label are required for cgmacros")
            path = cgmacros.to_canonical_csv(args.root, args.output, args.sensor,
                                             args.label, args.recovery)
            description = cgmacros.CGMACROS_LABELS[args.label].description
            print(f"Wrote {path} ({args.sensor} sensor, {args.recovery} recovery); "
                  f"label {args.label}: {description}")
        else:
            if args.sensor or args.label:
                parser.error("--sensor/--label do not apply to shanghai; it is unlabeled")
            path = shanghai.to_canonical_csv(args.root, args.output, args.cohort)
            print(f"Wrote {path} (Shanghai {args.cohort}, unlabeled for pretraining)")
    elif args.command == "partition":
        # Partition before windowing so each side can pick its own sampling mode.
        a, b = split_subjects(args.csv, args.output_a, args.output_b, args.fraction_b, args.seed)
        count = lambda path: len({row.split(",")[0] for row in
                                  open(path).read().splitlines()[1:] if row})
        print(f"A: {count(a)} subjects -> {a}")
        print(f"B: {count(b)} subjects -> {b}")
    elif args.command == "split":
        if not 0 < args.fraction_b < 1:
            parser.error("--fraction-b must lie strictly between 0 and 1")
        data = WindowSet.load(args.data)
        subjects = np.unique(data.subjects)
        order = np.random.default_rng(args.seed).permutation(len(subjects))
        count = max(1, min(len(subjects) - 1, round(args.fraction_b * len(subjects))))
        chosen = set(subjects[order[:count]].tolist())
        in_b = np.isin(data.subjects, list(chosen))
        a, b = data.subset(~in_b), data.subset(in_b)
        assert_subject_disjoint(a, b)
        a.save(args.output_a)
        b.save(args.output_b)
        print(f"A: {len(a)} windows / {len(np.unique(a.subjects))} subjects -> {args.output_a}")
        print(f"B: {len(b)} windows / {len(np.unique(b.subjects))} subjects -> {args.output_b}")
    elif args.command == "pretrain":
        config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=args.device)
        fit(WindowSet.load(args.train), WindowSet.load(args.validation), args.output, config)
    elif args.command == "aggregate":
        report = aggregate_probes(args.probes, args.baseline)
        report["provenance"] = {
            "inputs": {str(path): _sha256(path) for path in sorted(args.provenance)},
            "environment": runtime_info(), "packages": package_versions(),
            "commit": _git_commit(),
        }
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        for name, value in report["overall"].items():
            print(f"{report['baseline']} {name}: {100*value['mean_ap_gap']:+.2f} AP "
                  f"(sd over groups {100*value['std_over_groups']:.2f}); higher on "
                  f"{value['groups_where_baseline_higher']}/{value['groups']} groups")
        print(f"Wrote {args.output}")
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
