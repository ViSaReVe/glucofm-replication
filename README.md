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

The demo writes the selected checkpoint, training histories, runtime/configuration metadata, a portable evaluation window file, fold assignments, and evaluation metrics. Probe labels never enter the pretraining batches.

Checkpoint selection keeps the **last** epoch, and `last.pt` is that checkpoint. Validation loss is not a selection criterion here: the target is an EMA of the online encoder, so the loss measures how predictable the teacher has become rather than how good the representation is. In a paired experiment over 4 seeds, 10 vs 60 epochs on identical data and folds, validation loss improved in 4/4 runs by 2.3-3.9x while downstream AP got *worse* in 4/4, -1.77 AP with 95% CI [-2.68, -0.87]. The loss is still logged every epoch, together with two diagnostics that do track representation health: the effective rank of the validation embeddings (entropy of the centred singular-value spectrum) and the fraction of windows carrying no weight for one or both objectives.

## Use your own data

Supply a canonical CSV with `subject_id,timestamp,glucose_mg_dl` and an optional integer `label`. Timestamps are local, timezone-naive ISO values. Normalize units and resolve timezone/DST conventions in the dataset adapter first. Subject identifiers must be globally consistent across files and cohorts.

```csv
subject_id,timestamp,glucose_mg_dl,label
cohortA-person001,2026-01-01T06:30:00,104,0
cohortA-person001,2026-01-01T06:35:00,108,0
```

The two-row example illustrates the schema; a real import needs at least a complete day. Missing readings may be absent or blank. The importer averages duplicate time bins, keeps an observation mask, and splits at gaps longer than one hour.

`--sampling` chooses how windows are cut from each segment and is always stated explicitly. `non-overlapping` (the default) tiles disjoint days; because that stride is exactly 24 hours, every window in a segment also inherits the segment's start clock index. `pretraining` follows Appendix A.2 instead, advancing by a seeded random stride so windows overlap by 20-80% of a day and cover mixed circadian phases. Use it only for the pretraining partition — validation and downstream partitions must stay non-overlapping so that near-duplicate days cannot straddle a fold.

```bash
glucofm prepare --csv pretrain.csv --output pretrain.npz \
  --sampling pretraining --sampling-seed 0
glucofm prepare --csv validation.csv --output validation.npz --sampling non-overlapping
glucofm prepare --csv downstream.csv --output downstream.npz --sampling non-overlapping

glucofm pretrain --train pretrain.npz --validation validation.npz \
  --epochs 120 --batch-size 128 --output runs/real-data

glucofm probe --checkpoint runs/real-data/last.pt --data downstream.npz \
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
