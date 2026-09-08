"""Frozen binary linear probing with identical repeated subject-grouped splits."""

import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import provenance
from .data import WindowSet
from .train import move_batch


@torch.no_grad()
def embeddings(encoder, data: WindowSet, batch_size=128):
    encoder.eval()
    device = next(encoder.parameters()).device
    return np.concatenate([encoder(**move_batch(batch, device))["embedding"].cpu().numpy()
                           for batch in DataLoader(data, batch_size=batch_size)], axis=0)


def summary_features(data: WindowSet):
    """Simple observed-glucose baseline: six level/variability summaries."""
    features = []
    for x, mask in zip(data.glucose, data.observed, strict=True):
        values = x[mask]
        features.append([values.mean(), values.std(), values.min(), values.max(),
                         np.quantile(values, 0.1), np.quantile(values, 0.9)])
    return np.asarray(features)


def subject_splits(data: WindowSet, folds=5, repeats=10, seed=42):
    if repeats < 1 or folds < 2:
        raise ValueError("Need at least two folds and one repeat")
    if set(np.unique(data.labels)) != {0, 1}:
        raise ValueError("This first probe implementation requires binary labels 0 and 1")
    subjects, first = np.unique(data.subjects, return_index=True)
    labels = data.labels[first]
    if min(np.bincount(labels)) < folds:
        raise ValueError("Need at least one subject per class per fold")
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed+repeat)
        for fold, (train, test) in enumerate(splitter.split(subjects, labels)):
            train_ids, test_ids = subjects[train], subjects[test]
            yield repeat, fold, np.flatnonzero(np.isin(data.subjects, train_ids)), np.flatnonzero(np.isin(data.subjects, test_ids))


def probe(feature_sets: dict[str, np.ndarray], data: WindowSet, folds=5, repeats=10, seed=42):
    for name, values in feature_sets.items():
        if values.ndim != 2 or len(values) != len(data) or not np.isfinite(values).all():
            raise ValueError(f"Invalid feature matrix: {name}")
    results, assignments = [], []
    for repeat, fold, train, test in subject_splits(data, folds, repeats, seed):
        assignments.append({"repeat": repeat, "fold": fold,
                            "train_subjects": sorted(set(data.subjects[train])),
                            "test_subjects": sorted(set(data.subjects[test]))})
        for name, values in feature_sets.items():
            # Standardization is fit only on each fold's training subjects.
            classifier = make_pipeline(StandardScaler(), LogisticRegression(
                solver="lbfgs", C=1.0, max_iter=1000, random_state=seed,
            ))
            classifier.fit(values[train], data.labels[train])
            probability = classifier.predict_proba(values[test])[:, 1]
            prediction = classifier.predict(values[test])
            results.append({"model": name, "repeat": repeat, "fold": fold,
                            "average_precision": average_precision_score(data.labels[test], probability),
                            "roc_auc": roc_auc_score(data.labels[test], probability),
                            "macro_f1": f1_score(data.labels[test], prediction, average="macro"),
                            "test_windows": len(test)})
    summary = {}
    for name in feature_sets:
        rows = [row for row in results if row["model"] == name]
        summary[name] = {metric: {"mean": float(np.mean([row[metric] for row in rows])),
                                  "std": float(np.std([row[metric] for row in rows], ddof=1))}
                         for metric in ("average_precision", "roc_auc", "macro_f1")}
    return {"summary": summary, "fold_results": results, "subject_splits": assignments,
            "metric_note": "Average precision (AP), not trapezoidal PR-AUC. Mean/std across repeated folds; not confidence intervals.",
            "evaluation_unit": "held-out daily windows; subjects never cross train/test folds",
            "folds": folds, "repeats": repeats, "seed": seed, "source": data.source}


PROBE_FILE_PATTERN = re.compile(r"^(?P<group>.+)\.seed(?P<seed>\d+)\.json$")


METRICS = ("average_precision", "roc_auc", "macro_f1")


def _fold_values(report, model, metric):
    """Metric per (repeat, fold). `validate_probe_report` guarantees uniqueness."""
    return {(row["repeat"], row["fold"]): row[metric]
            for row in report["fold_results"] if row["model"] == model}


def validate_probe_report(report, name="report"):
    """Reject a report that cannot be pooled, rather than summarising it anyway.

    Checks the rows actually present against the grid the report *declares*. A
    report claiming five folds and ten repeats while carrying one fold would
    otherwise be averaged as though complete, and a duplicated row would silently
    overwrite its twin.
    """
    for field in ("folds", "repeats", "fold_results"):
        if field not in report:
            raise ValueError(f"{name}: probe report lacks {field!r}")
    folds, repeats = int(report["folds"]), int(report["repeats"])
    if folds < 2 or repeats < 1:
        raise ValueError(f"{name}: declares folds={folds}, repeats={repeats}")
    expected = {(repeat, fold) for repeat in range(repeats) for fold in range(folds)}
    models = sorted({row["model"] for row in report["fold_results"]})
    if not models:
        raise ValueError(f"{name}: no fold results")
    for model in models:
        seen = {}
        for row in report["fold_results"]:
            if row["model"] != model:
                continue
            key = (row["repeat"], row["fold"])
            if key in seen:
                raise ValueError(f"{name}: duplicate row for {model} at repeat "
                                 f"{key[0]} fold {key[1]}")
            seen[key] = row
            for metric in METRICS:
                if metric not in row:
                    raise ValueError(f"{name}: {model} row {key} lacks {metric!r}")
                if not np.isfinite(row[metric]):
                    raise ValueError(f"{name}: {model} row {key} has non-finite {metric}")
        if set(seen) != expected:
            missing = sorted(expected - set(seen))
            unexpected = sorted(set(seen) - expected)
            raise ValueError(
                f"{name}: {model} does not cover the declared {repeats}x{folds} grid; "
                f"missing {missing[:4]}, unexpected {unexpected[:4]}")
    if "subject_splits" in report and len(report["subject_splits"]) != len(expected):
        raise ValueError(f"{name}: {len(report['subject_splits'])} subject splits for a "
                         f"declared {repeats}x{folds} grid")
    return models


def _fold_identity(report):
    """What must match across the seeds pooled for one group."""
    return {
        "folds": report.get("folds"), "repeats": report.get("repeats"),
        # `seed` is the CV seed in archived reports; new reports also carry cv_seed.
        "cv_seed": report.get("cv_seed", report.get("seed")),
        "source": report.get("source"),
        "subject_splits": report.get("subject_splits"),
    }


def _dataset_identity(report):
    dataset = (report.get("provenance") or {}).get("dataset") or {}
    return dataset.get("sha256")


def aggregate_probes(directory, baseline="pretrained_encoder"):
    """Pool probe reports written as `<group>.seed<N>.json` into one summary.

    Each group is one evaluation partition (a sensor/task pair); each seed is one
    pretraining run. Metrics are averaged over folds within a seed and then over
    seeds, and every control is differenced against `baseline` on identical folds.
    The spread reported is across seeds, never across folds: repeated folds over one
    small subject pool are dependent, so a fold-level interval would be far too
    narrow to mean anything.
    """
    directory = Path(directory)
    reports, sources, unverified = {}, {}, []
    for path in sorted(directory.glob("*.json")):
        match = PROBE_FILE_PATTERN.match(path.name)
        if not match:
            continue
        key = (match["group"], int(match["seed"]))
        report = json.loads(path.read_text())
        validate_probe_report(report, path.name)
        declared = report.get("training_seed")
        if declared is None:
            # Archived reports predate training-seed metadata. Accepted, but the
            # filename's claim about which run produced them is not verifiable, and
            # saying so beats rewriting the files to look verified.
            unverified.append(path.name)
        elif int(declared) != key[1]:
            raise ValueError(f"{path.name}: filename says training seed {key[1]} but the "
                             f"report records {declared}")
        reports[key] = report
        sources[key] = {**provenance.file_record(path),
                        "training_seed": declared, "cv_seed": report.get("cv_seed", report.get("seed")),
                        "checkpoint_epoch": report.get("checkpoint_epoch"),
                        "probe_provenance": report.get("provenance")}
    if not reports:
        raise ValueError(f"No probe reports named '<group>.seed<N>.json' in {directory}")
    groups = sorted({group for group, _ in reports})
    seeds = sorted({seed for _, seed in reports})
    missing = [(g, s) for g in groups for s in seeds if (g, s) not in reports]
    if missing:
        raise ValueError(f"Incomplete grid; missing {missing[:5]}")

    # Within a group, seeds differ only in pretraining. Everything defining the
    # evaluation -- dataset, fold count, CV seed, actual subject assignments -- must
    # be identical, or the pooled mean mixes protocols.
    for group in groups:
        reference = _fold_identity(reports[(group, seeds[0])])
        reference_data = _dataset_identity(reports[(group, seeds[0])])
        for seed in seeds[1:]:
            other = _fold_identity(reports[(group, seed)])
            for field in ("folds", "repeats", "cv_seed", "source"):
                if reference[field] != other[field]:
                    raise ValueError(
                        f"{group}: seed {seed} differs from seed {seeds[0]} in {field!r} "
                        f"({other[field]!r} vs {reference[field]!r}); these are not "
                        "repetitions of one evaluation")
            if reference["subject_splits"] != other["subject_splits"]:
                raise ValueError(f"{group}: seed {seed} used different subject assignments "
                                 f"from seed {seeds[0]}; folds are not comparable")
            other_data = _dataset_identity(reports[(group, seed)])
            if reference_data and other_data and reference_data != other_data:
                raise ValueError(f"{group}: seed {seed} probed a different dataset "
                                 f"({other_data[:12]} vs {reference_data[:12]})")

    models = sorted({row["model"] for report in reports.values()
                     for row in report["fold_results"]})
    for key, report in reports.items():
        present = sorted({row["model"] for row in report["fold_results"]})
        if present != models:
            raise ValueError(f"{key[0]} seed {key[1]}: models {present} do not match {models}")
    if baseline not in models:
        raise ValueError(f"Baseline {baseline!r} is absent; found {models}")
    metrics = METRICS

    pooled, per_seed, paired = {}, {}, {}
    for group in groups:
        for model in models:
            values = {metric: [float(np.mean(list(_fold_values(reports[(group, seed)],
                                                               model, metric).values())))
                               for seed in seeds] for metric in metrics}
            per_seed[f"{group}/{model}"] = values
            pooled[f"{group}/{model}"] = {
                metric: {"mean": float(np.mean(v)),
                         "std_over_seeds": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                for metric, v in values.items()}
        for model in models:
            if model == baseline:
                continue
            gaps = []
            for seed in seeds:
                report = reports[(group, seed)]
                a = _fold_values(report, baseline, "average_precision")
                b = _fold_values(report, model, "average_precision")
                if a.keys() != b.keys():
                    raise ValueError(f"{group} seed {seed}: {model} and {baseline} "
                                     "were not evaluated on identical folds")
                gaps.append(float(np.mean([a[k] - b[k] for k in sorted(a)])))
            paired[f"{group}/vs_{model}"] = {
                "per_seed": gaps, "mean": float(np.mean(gaps)),
                "std_over_seeds": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0}

    overall = {}
    for model in models:
        if model == baseline:
            continue
        every = [paired[f"{group}/vs_{model}"]["mean"] for group in groups]
        overall[f"vs_{model}"] = {
            "mean_ap_gap": float(np.mean(every)),
            "std_over_groups": float(np.std(every, ddof=1)) if len(every) > 1 else 0.0,
            "groups_where_baseline_higher": int(sum(gap > 0 for gap in every)),
            "groups": len(every)}
    return {"groups": groups, "training_seeds": seeds, "seeds": seeds,
            "models": models, "baseline": baseline,
            "pooled": pooled, "per_seed": per_seed, "paired_ap_gap": paired,
            "overall": overall,
            "sources": {f"{group}.seed{seed}": record
                        for (group, seed), record in sorted(sources.items())},
            "source_provenance": {
                "reports": len(reports),
                "training_seed_verified": sorted(
                    f"{g}.seed{s}" for (g, s) in reports if f"{g}.seed{s}.json" not in unverified),
                "training_seed_unverified": sorted(unverified),
                "note": "Reports without a training_seed field predate that metadata. "
                        "Their filename's seed claim is taken at face value and listed "
                        "here; the files themselves are not modified.",
            },
            "note": "`seeds`/`training_seeds` are pretraining seeds; the CV seed is "
                    "fixed within each group and checked. Spread is across pretraining "
                    "seeds. Repeated folds over one subject pool are dependent; no "
                    "confidence intervals are implied."}
