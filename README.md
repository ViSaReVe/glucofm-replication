# GlucoFM: an unofficial PyTorch implementation

An independent implementation of the main mechanisms in **GlucoFM: A Dual-Stream Foundation Model for Continuous Glucose Monitoring**, [arXiv:2605.30865v1](https://arxiv.org/abs/2605.30865v1).

The project asks how ideas familiar from biosignal processing—windowing, decomposition, missing-data handling, and subject-separated evaluation—fit into self-supervised representation learning.

**Status:** core implementation, plus a completed subject-disjoint evaluation on real public CGM data. This is not Google's code, an official checkpoint, a clinical model, or a reproduction of the reported clinical results. The paper's private Wear-CGM pretraining data are not included. V2's additional post-meal response experiments are outside this implementation's scope.

### Headline result

Pretrained on **ShanghaiT2DM** (80 subjects, 23,356 hours), evaluated subject-disjoint on **CGMacros** (45 subjects, four phenotype tasks, two sensors, three pretraining seeds):

**Frozen pretrained representations beat a frozen random encoder of identical architecture by +1.16 average-precision points, on 6 of 8 sensor/task pairs, with most per-task gaps inside the spread across seeds.** That does not establish an advantage.

The wider margin over six glucose summaries (+3.75 AP) is mostly not pretraining: on the obesity tasks the *random* encoder already beats the summaries by 15.1 and 10.4 AP.

This run changes corpus size, population, clinical setting and sensor sampling rate relative to the paper simultaneously, so it cannot attribute the outcome to any one of them — "not enough data" is a hypothesis it does not test. An independent evaluation reran the probes from the preserved checkpoints and reproduced every number exactly. Full protocol, limitations and the reasons these numbers are **not** directly comparable to the paper's Table 3 are in [reports/real-data.md](reports/real-data.md).

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

Python 3.10+ is required. The real-data measurements are in [reports/real-data.md](reports/real-data.md); the earlier synthetic software demonstration is in [reports/synthetic-smoke.md](reports/synthetic-smoke.md).

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

`--sampling` chooses how windows are cut from each segment and is always stated explicitly. `non-overlapping` (the default) tiles disjoint days; because that stride is exactly 24 hours, every window in a segment also inherits the segment's start clock index. `pretraining` follows Appendix A.2 instead, advancing by a seeded random stride so windows overlap by 20-80% of a day and cover mixed circadian phases. Use it only for the pretraining partition. Validation and downstream partitions stay non-overlapping — not to prevent leakage, since subject-grouped folds already keep a subject's windows together, but to stop the same hours of a subject being counted repeatedly and over-weighting densely recorded subjects.

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

## Public cohorts

Two open cohorts, both downstream cohorts in the paper itself. Neither is redistributed here; `src/glucofm/datasets/` converts each archive into the canonical CSV above, resolving units, timestamps and subject identifiers inside the adapter. Download instructions, the published-file quirks the adapters absorb, and the clinical source of every label threshold are in [docs/datasets.md](docs/datasets.md).

- **CGMacros** (PhysioNet, CC BY-NC-SA 4.0, 45 subjects, ten days each). Downstream. A Dexcom G6 Pro at five minutes and a FreeStyle Libre Pro at fifteen were worn at the same time; they stay **separate partitions** and there is no merged mode. The published series are linearly interpolated onto a one-minute grid, so the adapter reconstructs the real sensor samples rather than presenting interpolation as observation. Four subject-level labels come from the bio panel.
- **ShanghaiT2DM** (figshare, CC BY 4.0, 100 patients, 109 recording periods, ~28,095 hours at fifteen minutes). Pretraining, emitted unlabeled.

```bash
python -m pip install -e '.[datasets]'   # Shanghai ships Excel workbooks

glucofm convert --dataset shanghai --root shanghai --cohort T2DM --output shanghai-t2dm.csv
glucofm prepare --csv shanghai-t2dm.csv --output shanghai-t2dm.npz \
  --sampling pretraining --sampling-seed 0
glucofm split --data shanghai-t2dm.npz --output-a pretrain.npz --output-b preval.npz \
  --fraction-b 0.2 --seed 0

glucofm convert --dataset cgmacros --root cgmacros \
  --sensor dexcom --label diabetes --output cgm-dexcom-diabetes.csv
glucofm prepare --csv cgm-dexcom-diabetes.csv --output cgm-dexcom-diabetes.npz \
  --sampling non-overlapping

glucofm pretrain --train pretrain.npz --validation preval.npz \
  --epochs 120 --batch-size 128 --output runs/shanghai
glucofm probe --checkpoint runs/shanghai/last.pt --data cgm-dexcom-diabetes.npz \
  --folds 5 --repeats 10 --output probe.json
```

`glucofm partition` splits the canonical CSV by subject *before* windowing, so the pretraining and validation sides can take different sampling modes; `glucofm split` does the same to an already-windowed NPZ, which forces one mode on both. Both reject overlap, and probing rejects overlap with either pretraining partition.

### Reproducible workflow

The whole experiment — partitioning, preparation, training, probing and aggregation, with fixed seeds — is one script:

```bash
./experiments/real_data.sh <shanghai-root> <cgmacros-root> runs/real-data
```

It writes `aggregate.json` with pooled and paired metrics plus a provenance record: SHA-256 of every prepared dataset, package versions and the git commit. Cohorts must be downloaded first ([docs/datasets.md](docs/datasets.md)); nothing is redistributed here.

It does **not** reproduce the recorded 2026-09-07 numbers, by design: it partitions before windowing so validation is non-overlapping, whereas the recorded run split an already-windowed NPZ. `LEGACY_VALIDATION=1` reproduces the original partitioning. Pretraining is stochastic across BLAS versions and thread counts in any case, so expect close rather than identical numbers; [reports/real-data.md](reports/real-data.md) is the record of the original run and is not regenerated.

## Evidence and fidelity

- [Real-data experiment](reports/real-data.md): pretraining on ShanghaiT2DM, subject-disjoint evaluation on CGMacros, with both controls in every table.
- [Synthetic smoke experiment](reports/synthetic-smoke.md): actual measurements, configuration, baselines and ablation; no clinical interpretation.
- [Equation-to-code map and implementation decisions](docs/implementation-decisions.md): what follows v1, what is assumed, and what remains outside scope.
- [Cohort adapters](docs/datasets.md): downloads, unit and timezone conventions, and every label threshold with its clinical source.
- [From biosignals to this model](docs/biosignal-connections.md): windowing, intersubject variability, and representation learning.
- `tests/`: hidden-input isolation, missingness invariance, causal filtering, loss weights, gradient paths, EMA behavior, grouped folds and checkpoint round trips.

Some layer details are not fully specified in v1. The repository records its choices and actual parameter count instead of claiming exact source equivalence. A synthetic result, even a high score, cannot establish clinical validity or reproduce the paper's benchmark. The central real-data milestone is a subject-separated experiment on an accessible cohort, with the same baselines and a documented data-access protocol.

## Citation and license

Credit the research paper when using or discussing its method; [CITATION.bib](CITATION.bib) contains its reference. Implementation code is under the [MIT license](LICENSE). The original paper, datasets, and any third-party weights retain their own terms and are not redistributed here.
