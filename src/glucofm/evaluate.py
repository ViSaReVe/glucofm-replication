"""Frozen binary linear probing with identical repeated subject-grouped splits."""

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
