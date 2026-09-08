# Real-data experiment: pretrain on ShanghaiT2DM, evaluate on CGMacros

Run date: 2026-09-07. This is the first experiment in this repository on real CGM
recordings. It does not replace [the synthetic smoke experiment](synthetic-smoke.md),
which remains the record of the software demonstration on generated data.

**What this measures.** What frozen GlucoFM representations are worth on CGMacros
when the only pretraining data available is public. The paper's headline result
depends on a private corpus that is not obtainable, so nothing here can confirm or
refute it, and nothing below should be read as trying to.

**What it does not isolate.** This experiment changes corpus size, population,
clinical setting and sensor sampling rate relative to the paper, all at once. It
therefore cannot attribute the outcome to any one of them. "Not enough data" is a
*hypothesis* consistent with the result, not an explanation this experiment
establishes; a controlled data-size sweep holding population and evaluation fixed
would be needed to test it.

## Result

Pretraining on ShanghaiT2DM produced a **small and inconsistent** improvement over a
frozen random encoder on CGMacros: **+1.16 AP points on average across eight
sensor/task pairs, higher on 6 of 8**. Most per-task differences are within the
spread across pretraining seeds. On this corpus, at this scale, **pretraining does
not establish an advantage over the random-encoder control.**

This measures the reimplementation on this corpus under this protocol. It is not evidence that the
method fails: a different implementation, corpus or training procedure could give a
different answer, and this run does not test any of them. Nor is it evidence that
more data would fix it — see *What it does not isolate* above.

Two absolute numbers are worth reading alongside the gains. Dexcom/hyperlipidemia has
the largest gain over random (+4.37 AP) but a pretrained ROC-AUC of 48.72%, which is
chance; a relative improvement there is not a useful classifier. And the high
insulin-resistance AP sits on a high positive rate — 32 of 45 Dexcom subjects and 31
of 44 Libre subjects are positive — so AP near 92 is closer to the base rate than it
looks.

The larger-looking margin over the six glucose summaries (+3.75 AP, higher on 7 of 8)
is **not** attributable to pretraining. It is carried almost entirely by the two
obesity tasks, where the *random* encoder also beats the summaries by 15.1 and 10.4
AP. That gap is the architecture and the probe, not anything learned.

## Setup

**Pretraining corpus: ShanghaiT2DM**, converted by `glucofm.datasets.shanghai`,
windowed with the Appendix A.2 overlapping sampler.

| | subjects | windows | hours | source readings |
|---|---:|---:|---:|---:|
| Gradient updates | 80 | 1,797 | 23,356 | 93,500 |
| Model-selection monitoring | 20 | 360 | 4,736 | 18,962 |
| Whole cohort available | 100 | 2,157 | 28,092 | 112,462 |

15-minute sampling, so window density is 0.333 on the 5-minute grid. Overlapping
sampling gives 288 distinct clock phases against 69 for non-overlapping tiling.

**Downstream cohort: CGMacros**, converted by `glucofm.datasets.cgmacros`,
non-overlapping windows, the two sensors kept as separate partitions. Non-overlap is
not a leakage guard here — cross-validation is subject-grouped, so a subject's windows
never straddle a fold — it stops the same hours of a subject being counted repeatedly
within a fold. Either way the window counts below are not independent sample counts.

| Partition | subjects | windows | mean density | positives / subjects |
|---|---:|---:|---:|---|
| Dexcom (5 min) | 45 | 403 | 0.832 | diabetes 14, IR 32, hyperlip. 12, obesity 23 |
| Libre (15 min) | 44 | 437 | 0.306 | diabetes 14, IR 31, hyperlip. 12, obesity 23 |

Subject separation is total and structural: pretraining subjects are Chinese
inpatients from ShanghaiT2DM, evaluation subjects are CGMacros participants. The
probe rejects any overlap with either pretraining partition, and `glucofm split`
rejects overlap between the pretraining and monitoring partitions.

**Training.** The paper's stated configuration: 120 epochs, batch 128, lr 1e-4,
weight decay 1e-2, bandwidth lr 1e-3. Three pretraining seeds (42, 43, 44).
Checkpoint selection is keep-last. Each run took about 29 minutes on CPU.

**Probing.** Frozen encoder, sklearn `LogisticRegression` (L2, lbfgs, C=1,
max_iter 1000), 5-fold subject-grouped cross-validation × 10 repeats, identical
folds across all three feature sets, per fold standardization fit on training
subjects only. Every table below carries both controls.

## Average precision

Percentages. Mean over 3 pretraining seeds; each seed's value is itself the mean over
5 folds × 10 repeats. The ± figure is the standard deviation **across the three
seeds**. The glucose summaries do not depend on the pretraining seed, so their
spread is zero by construction — a useful check that the folds really are identical.

| Sensor / task | Six glucose summaries | Frozen random encoder | Pretrained encoder |
|---|---:|---:|---:|
| Dexcom / diabetes | 74.81 | 71.78 ± 2.23 | **73.92 ± 2.01** |
| Dexcom / insulin resistance | 90.40 | 91.98 ± 0.69 | **92.56 ± 0.41** |
| Dexcom / hyperlipidemia | 31.45 | 27.39 ± 0.75 | **31.76 ± 1.30** |
| Dexcom / obesity | 60.45 | 75.58 ± 0.45 | **76.41 ± 0.53** |
| Libre / diabetes | 77.75 | **78.24 ± 0.49** | 78.16 ± 1.40 |
| Libre / insulin resistance | 92.11 | 92.77 ± 0.41 | **92.91 ± 0.93** |
| Libre / hyperlipidemia | 37.84 | 37.40 ± 1.39 | **39.14 ± 3.16** |
| Libre / obesity | 57.10 | **67.52 ± 0.72** | 67.02 ± 3.24 |

### Paired differences

Pretrained minus control, paired on identical folds within each seed, then averaged.

| Sensor / task | vs random encoder | vs glucose summaries |
|---|---:|---:|
| Dexcom / diabetes | +2.15 ± 1.52 | −0.88 ± 2.01 |
| Dexcom / insulin resistance | +0.58 ± 0.35 | +2.16 ± 0.41 |
| Dexcom / hyperlipidemia | +4.37 ± 2.02 | +0.31 ± 1.30 |
| Dexcom / obesity | +0.83 ± 0.91 | +15.96 ± 0.53 |
| Libre / diabetes | −0.08 ± 1.88 | +0.42 ± 1.40 |
| Libre / insulin resistance | +0.15 ± 1.33 | +0.80 ± 0.93 |
| Libre / hyperlipidemia | +1.74 ± 1.97 | +1.30 ± 3.16 |
| Libre / obesity | −0.50 ± 3.65 | +9.92 ± 3.24 |
| **Mean over the eight** | **+1.16** (6/8 positive) | **+3.75** (7/8 positive) |

Three tasks exceed their own seed spread — Dexcom/hyperlipidemia (+4.37 ± 2.02),
Dexcom/diabetes (+2.15 ± 1.52) and Dexcom/insulin resistance (+0.58 ± 0.35), the last
of these a gain too small to matter. Two tasks are negative. **No confidence intervals are quoted
anywhere in this report.** The 50 folds per seed are 10 repeated 5-fold partitions
of the same 45 subjects, so fold-level errors are strongly dependent and any interval
computed across them would be far too narrow. The ± figures are seed-to-seed spread
and nothing more; with three seeds they are indicative, not inferential.

## ROC-AUC and Macro-F1

| Sensor / task | ROC-AUC: summaries / random / pretrained | Macro-F1: summaries / random / pretrained |
|---|---|---|
| Dexcom / diabetes | 85.59 / 83.82 ± 1.23 / 84.90 ± 0.97 | 69.19 / 74.34 ± 1.10 / 75.21 ± 1.61 |
| Dexcom / insulin resistance | 81.78 / 85.09 ± 1.26 / 86.00 ± 0.59 | 70.94 / 75.23 ± 0.54 / 75.40 ± 0.10 |
| Dexcom / hyperlipidemia | 48.07 / 43.31 ± 1.24 / 48.72 ± 1.02 | 43.81 / 44.39 ± 1.07 / 47.00 ± 1.59 |
| Dexcom / obesity | 60.64 / 77.40 ± 0.47 / 78.29 ± 0.07 | 57.26 / 69.48 ± 0.41 / 70.58 ± 0.81 |
| Libre / diabetes | 86.75 / 88.54 ± 0.19 / 87.57 ± 1.04 | 70.14 / 75.38 ± 0.38 / 72.70 ± 2.13 |
| Libre / insulin resistance | 81.65 / 84.10 ± 0.80 / 84.06 ± 1.53 | 65.90 / 70.15 ± 1.15 / 70.30 ± 1.78 |
| Libre / hyperlipidemia | 55.99 / 55.66 ± 1.53 / 59.20 ± 3.97 | 45.86 / 52.10 ± 0.70 / 52.77 ± 1.67 |
| Libre / obesity | 52.06 / 62.84 ± 0.44 / 63.22 ± 2.27 | 50.15 / 58.32 ± 0.18 / 59.02 ± 1.79 |

The same picture: the pretrained encoder is close to the random encoder throughout,
ahead on most tasks by under a point, behind on Libre/diabetes.

## Independent reproduction

An independent evaluation reran the CGMacros probes from the preserved seed-42/43/44
checkpoints, using this repository's code and the prepared datasets. Across every
sensor, task, seed and metric the **maximum absolute difference from the numbers in
this report was 0.0**. It also verified that no training or validation subject from
any checkpoint appears in the downstream data.

That evaluation produced the corrections applied above (seed 42's minimum epoch, the
scope of the objective-support diagnostic, and the data-scale framing) and the
missing-label sensitivity analysis referenced under *Limitations*. It changed no
repository code and retrained no model.

## Scale, against the paper's corpus

The paper pretrains on 477 dataset-defined subjects and 109,066 hours across five
cohorts (arXiv:2605.30865v1, Table 2; identical in v2). Wear-CGM, 192 subjects and
75,330 hours of it, is private and is not included in this repository.

| Measure | This run | Paper | Share |
|---|---:|---:|---:|
| Subjects | 80 | 477 | 16.8% |
| Recording hours | 23,356 | 109,066 | 21.4% |
| CGM readings | 93,500 | ~1,209,480 | 7.7% |

ShanghaiT2DM samples every 15 minutes, so a fifth of the paper's *hours* is under a tenth of its *readings*.
Corpus composition differs more than size does. The paper mixes five cohorts spanning
two sampling rates and several populations; this run has one cohort, one rate, one
country, one clinical setting.

These ratios describe the gap. They do not explain the result: size and composition
moved together here, so neither can be credited with the outcome.

## Why these numbers are not directly comparable to the paper's Table 3

The paper reports, for GlucoFM (0.72M) on CGMacros: diabetes 65.9 PR-AUC / 78.7 AUC /
58.3 F1; insulin resistance 91.9 / 81.2 / 69.6; hyperlipidemia 36.1 / 54.7 / 50.2;
obesity 64.9 / 62.6 / 59.4.

Several of this report's absolute numbers sit at or above those. **That comparison is
not meaningful, and it is not evidence that this implementation matched the paper.**

1. **The metric differs.** This repository reports scikit-learn average precision. The
   paper reports PR-AUC. They are not the same statistic.
2. **The label definitions are probably not the same.** The paper does not state the
   HOMA-IR or hyperlipidemia cutoffs behind its CGMacros columns. This repository's
   choices, and the fact that two of the four have no settled clinical consensus, are
   documented in [docs/datasets.md](../docs/datasets.md). Different thresholds move
   the class balance and therefore move AP directly.
3. **The sensor is unstated.** The paper's Table 2 lists CGMacros at both 5 and 15
   minutes, but Table 3 has a single CGMacros column and does not say which sensor it
   reports, or whether the two were pooled.
4. **The folds and the preprocessing differ**, including the reconstruction of real
   sensor samples from the published one-minute interpolated grid, which the paper
   does not describe doing.

The one comparison that *is* internally controlled is the one this repository exists
to make: same encoder, same folds, same probe, pretrained weights against random
weights. The paper's Table 3 contains no random-encoder control, so its absolute
numbers cannot by themselves separate a learned representation from the architecture
and the probe. On CGMacros a frozen random encoder already reaches 92.8 AP on insulin
resistance (Libre) and 75.6 on obesity (Dexcom), without having been trained at all.

## Training diagnostics

| Seed | Validation loss, epoch 1 → 120 | Minimum (epoch) | Effective rank, 1 → 120 | σ, 1 → 120 |
|---|---|---|---|---|
| 42 | 0.2216 → 0.016140 | 0.016085 (117) | 8.52 → 4.21 | 5.96 → 4.83 |
| 43 | 0.2131 → 0.029049 | 0.020614 (51) | 7.67 → 5.96 | 5.96 → 4.76 |
| 44 | 0.2084 → 0.020399 | 0.015773 (52) | 6.96 → 4.62 | 5.96 → 4.71 |

Two things the per-epoch diagnostics show that the loss alone does not.

**Validation loss falls by 7–14× while the effective rank of the validation
embeddings falls by 22–51%.** The representation contracts into fewer directions as
the objective improves — the loss going down is compatible with the representation
getting narrower, which is what the effective-rank diagnostic exists to surface.
Whether that contraction causes the weak downstream result is not established here.

**The loss-minimising epoch is not the last one in any of the three runs** (epochs
117, 51 and 52). On this cohort the old lowest-validation-loss rule would have
selected a different checkpoint every time, unlike the synthetic run where the
minimum was always the final epoch. For seed 42 the difference is negligible
(0.016085 at epoch 117 against 0.016140 at 120); for seeds 43 and 44 it is not. The
keep-last rule and its justification are in
[docs/implementation-decisions.md](../docs/implementation-decisions.md), assumption 10.

*(An earlier version of this table recorded seed 42's minimum at epoch 120. That was
a transcription error in the table only; `real-data/pretraining-runs.json` recorded
117 throughout.)*

**The objective-support diagnostic covers the validation set only.** No *validation*
window in any epoch lacked weight for either objective
(`windows_missing_an_objective` = 0.000 throughout). Training windows are augmented
and hidden independently, and the training loop does not measure their support, so
this run does not establish that every training window carried both objectives. That
is a gap in the instrumentation, not a measurement.

## Limitations

- **The pretraining corpus is 15-minute data only.** The Libre partition is
  rate-matched to it; the **Dexcom partition is out-of-rate** for this encoder. The
  paper's mixture deliberately spans both rates. Pretraining happened to help
  slightly more on Dexcom, which is the opposite of what a sampling-rate explanation
  predicts, but with three seeds and eight tasks that is not a finding.
- **One cohort, one population, one clinical setting.** ShanghaiT2DM is hospitalised
  Chinese T2DM patients; CGMacros is free-living US participants across healthy,
  pre-diabetic and T2DM. The domain shift is large and is not separable here from the
  scale effect.
- **This run uses all 100 ShanghaiT2DM patients for pretraining.** The paper splits
  the same cohort, 44 recording-visit entries into pretraining and 65 into downstream
  evaluation. Since nothing here is evaluated on ShanghaiT2DM, subject separation
  holds for this experiment, but the corpus is not the paper's pretraining subset.
- **Three seeds**, and variation across seeds is comparable to the effect being
  measured. Three initialization seeds do not establish statistical significance, and
  the direction a larger seed sample would settle on is unknown.
- **The eight sensor/task pairs are not eight independent cohorts.** Dexcom and Libre
  were worn simultaneously by largely the same participants, so the two sensor rows
  for a task are paired measurements, not replication.
- **Metrics are over held-out daily windows, not one prediction per participant**, so
  participants contributing more retained windows carry more weight.
- **45 downstream subjects**, 12–32 positives per task. Small.
- **No hyperparameter search**, on either the pretraining or the probe side. The
  paper's stated configuration was used as-is.
- **CPU only.** MPS and CUDA were not exercised.
- **No ablation** of the dynamics objective on real data; the synthetic report has
  that comparison and this one does not.
- **The CGMacros observation mask is an estimate, not the physical mask.** The
  published files are interpolated onto a one-minute grid and the sampling mask is
  not recoverable from them. This run used the `slope-change` policy, which excludes
  every ambiguous candidate: 4.85% of on-lattice Dexcom and 2.84% of Libre points
  are excluded with equal bracketing readings. How many of those were real readings
  is not identifiable from the files, so the policy's error rate against the physical
  mask is unknown. What is clear is the shape of the bias — exclusions concentrate
  where the series is flat, in a model that reads the mask as an input channel — and
  its effect on these scores is unmeasured.
  See [docs/datasets.md](../docs/datasets.md).
- **The pretraining-validation partition used overlapping windows**, contrary to the
  protocol written in `docs/implementation-decisions.md` assumption 11. The split is
  by subject, so nothing leaks between training and validation; the cost is that
  correlated windows shrink the effective sample size behind the validation loss and
  the effective-rank diagnostic. The run is reported as it happened.

## What would change the answer

First settle preprocessing, so that later runs are comparable: the mask policy and
the missing-input policy are now explicit and documented, and should not move again.

Then the cheapest informative experiment is **the no-dynamics ablation** — full
training against `--dynamics-weight 0` on the same saved subject partitions, the same
random controls and a predeclared epoch-120 rule. It asks whether the temporal
objective earns its place before any larger run is paid for.

A data-size sweep is worth running only in controlled form: hold the evaluation
participants fixed and vary pretraining hours alone, since changing population and
hours together — as this run does — cannot isolate a scale effect.

One caution about these particular numbers. They have now been used to characterise
the implementation. Repeatedly tuning against them would turn CGMacros from a held-out
test into development data; a future headline claim should be made on a cohort that
has not been looked at this many times.

## Reproducing

Downloads and conversion are documented in [docs/datasets.md](../docs/datasets.md).

```bash
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
  --epochs 120 --batch-size 128 --seed 42 --output runs/seed42
glucofm probe --checkpoint runs/seed42/last.pt --data cgm-dexcom-diabetes.npz \
  --folds 5 --repeats 10 --output probe.json
```

Repeat the last two lines for seeds 43 and 44 and for each of the eight
sensor/label partitions. `experiments/real_data.sh` drives all of it, including
aggregation and a provenance record, and requires a fresh output directory; run it
with `LEGACY_VALIDATION=1` to reproduce this run's partitioning, since its default
corrects the overlapping-validation deviation described under *Limitations*. Pretraining is stochastic across BLAS
versions and thread counts, so expect close rather than identical numbers.

The CGMacros mask policy in force for this run was `slope-change`
(`--recovery slope-change`, the default).

Machine-readable evidence is in `real-data/`: `probe-aggregate.json` (all pooled and
paired numbers above), `pretraining-runs.json` (per-epoch history and summary for the
three seeds), and `environment-pins.txt`. Checkpoints and the converted cohort files
are not in the repository; the cohorts are redistributed by their own sources under
their own licences.

Runtime: Python 3.12.4, PyTorch 2.9.1, NumPy 1.26.4, scikit-learn 1.8.0, macOS 26.6.2
on arm64. This differs from the environment recorded for the synthetic run, so the
two reports' timings are not comparable.

## What this report does not claim

It does not reproduce the paper's benchmark, and cannot: the pretraining corpus is
different and most of the paper's is unavailable. It does not show that GlucoFM's
method fails — a null result at a fifth of the hours from a single cohort is a
statement about this corpus, not about the method. It does not establish clinical
validity of anything. And it does not show that pretraining is useless here; it shows
that on this data the advantage over an untrained encoder of identical architecture is
small enough that three seeds cannot separate it from noise.
