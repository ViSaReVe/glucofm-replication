# GlucoFM: an unofficial PyTorch implementation

An independent implementation of the main mechanisms in **GlucoFM: A Dual-Stream Foundation Model for Continuous Glucose Monitoring**, [arXiv:2605.30865v1](https://arxiv.org/abs/2605.30865v1).

The project asks how ideas familiar from biosignal processing—windowing, decomposition, missing-data handling, and subject-separated evaluation—fit into self-supervised representation learning.

**Status:** core implementation and synthetic demonstration. This is not Google's code, an official checkpoint, a clinical model, or a reproduction of the reported clinical results. The paper's private Wear-CGM pretraining data are not included. V2's additional post-meal response experiments are outside this implementation's scope.

## What it does

```text
24-hour glucose window, physical observation mask, local start time
    │
    ├── Mask-aware normalization → learnable causal Gaussian filter
    │                                │                  │
    │                             state              residual
    │                                │                  │
    │                      local state tokens   local event tokens
    │                         [24 × 64]            [24 × 64]
    │                                └────────┬─────────┘
    │                                  fuse + time
    │                                   [24 × 128]
    │                                        │
    │                              3-layer Transformer
    │                                        │
    │                            mean → 128-D daily vector
    │                                        │
    │                              frozen linear probe
    │
    └── Pretraining: masked contextual prediction + local transition prediction
         Online encoder learns by gradients; target encoder tracks its EMA.
```

Selected online patches are hidden **before** normalization, statistics, and filtering. Both streams are fused **before** the global Transformer. Next-patch state/event prediction uses local tokens **before** global attention. These placements matter to the actual learning problem.

## Run locally

Tested runtime and measurements are recorded in [the experiment report](reports/synthetic-smoke.md). Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q

# Train on generated unlabeled windows; probe on separate generated subjects.
# Include a matched run with the temporal-dynamics loss disabled.
glucofm demo --epochs 10 --ablation --output runs/synthetic-demo
```

The default demo uses the documented main Transformer dimensions. It shortens training and uses synthetic data; it is not a small alias for the paper's full experiment. CPU is the default. `--device mps` and `--device cuda` are supported command options but require validation on the intended hardware.

The demo writes best/last checkpoints, training histories, runtime/configuration metadata, a portable evaluation window file, fold assignments, and evaluation metrics. Model selection uses separate pretraining-validation subjects. Probe labels never enter the pretraining batches.

## Use your own data

Supply a canonical CSV with `subject_id,timestamp,glucose_mg_dl` and an optional integer `label`. Timestamps are local, timezone-naive ISO values. Normalize units and resolve timezone/DST conventions in the dataset adapter first. Subject identifiers must be globally consistent across files and cohorts.

```csv
subject_id,timestamp,glucose_mg_dl,label
cohortA-person001,2026-01-01T06:30:00,104,0
cohortA-person001,2026-01-01T06:35:00,108,0
```

The two-row example illustrates the schema; a real import needs at least a complete day. Missing readings may be absent or blank. The importer averages duplicate time bins, keeps an observation mask, splits at gaps longer than one hour, and extracts non-overlapping days.

```bash
glucofm prepare --csv pretrain.csv --output pretrain.npz
glucofm prepare --csv validation.csv --output validation.npz
glucofm prepare --csv downstream.csv --output downstream.npz

glucofm pretrain --train pretrain.npz --validation validation.npz \
  --epochs 120 --batch-size 128 --output runs/real-data

glucofm probe --checkpoint runs/real-data/best.pt --data downstream.npz \
  --folds 5 --repeats 10 --output probe.json
```

Create the three CSV partitions by **subject**, before preparing windows. Train and validation subject overlap is rejected; probing also rejects overlap with either pretraining partition. The current probe supports binary labels `0` and `1`; unlabeled pretraining data use `-1`. Multiclass glucotypes and cohort-specific label construction are future work.

For each downstream fold, a standardizer and L2 logistic regression are fit only on training subjects. All days of a subject remain together. The same folds compare six simple glucose summaries, a frozen random encoder, and the frozen pretrained encoder. Reported AP is scikit-learn average precision, not trapezoidal PR-AUC.

## Evidence and fidelity

- [Synthetic smoke experiment](reports/synthetic-smoke.md): actual measurements, configuration, baselines and ablation; no clinical interpretation.
- [Equation-to-code map and implementation decisions](docs/implementation-decisions.md): what follows v1, what is assumed, and what remains outside scope.
- [From biosignals to this model](docs/biosignal-connections.md): windowing, intersubject variability, and representation learning.
- `tests/`: hidden-input isolation, missingness invariance, causal filtering, loss weights, gradient paths, EMA behavior, grouped folds and checkpoint round trips.

Some layer details are not fully specified in v1. The repository records its choices and actual parameter count instead of claiming exact source equivalence. A synthetic result, even a high score, cannot establish clinical validity or reproduce the paper's benchmark. The central real-data milestone is a subject-separated experiment on an accessible cohort, with the same baselines and a documented data-access protocol.

## Citation and license

Credit the research paper when using or discussing its method; [CITATION.bib](CITATION.bib) contains its reference. Implementation code is under the [MIT license](LICENSE). The original paper, datasets, and any third-party weights retain their own terms and are not redistributed here.
