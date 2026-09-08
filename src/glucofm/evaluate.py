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


def _fold_values(report, model, metric):
    """Metric per (repeat, fold), keyed so models pair on identical folds."""
    return {(row["repeat"], row["fold"]): row[metric]
            for row in report["fold_results"] if row["model"] == model}


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
    reports = {}
    for path in sorted(directory.glob("*.json")):
        match = PROBE_FILE_PATTERN.match(path.name)
        if match:
            reports[(match["group"], int(match["seed"]))] = json.loads(path.read_text())
    if not reports:
        raise ValueError(f"No probe reports named '<group>.seed<N>.json' in {directory}")
    groups = sorted({group for group, _ in reports})
    seeds = sorted({seed for _, seed in reports})
    missing = [(g, s) for g in groups for s in seeds if (g, s) not in reports]
    if missing:
        raise ValueError(f"Incomplete grid; missing {missing[:5]}")
    models = sorted({row["model"] for report in reports.values()
                     for row in report["fold_results"]})
    if baseline not in models:
        raise ValueError(f"Baseline {baseline!r} is absent; found {models}")
    metrics = ("average_precision", "roc_auc", "macro_f1")

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
    return {"groups": groups, "seeds": seeds, "models": models, "baseline": baseline,
            "pooled": pooled, "per_seed": per_seed, "paired_ap_gap": paired,
            "overall": overall,
            "note": "Spread is across pretraining seeds. Repeated folds over one "
                    "subject pool are dependent; no confidence intervals are implied."}
