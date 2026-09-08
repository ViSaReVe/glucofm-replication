# Synthetic smoke experiment

Run date: 2026-09-07. This is a software and evaluation-pipeline demonstration using generated regimes. It is not a clinical experiment or a reproduction of the GlucoFM benchmark.

> **This page is the record of that run, not a statement of current behaviour.** The
> measurements below were produced before the compression-envelope, window-sampling,
> mask-channel and checkpoint-selection changes recorded in
> `docs/implementation-decisions.md`. Every number in the tables and in the JSON
> artifacts belongs to the 2026-09-07 run and is left untouched; where the current code
> differs, that is stated inline. The demo has not been re-run since, so there are no
> measured current metrics to report here.

## Protocol

```bash
glucofm demo --epochs 10 --ablation --output runs/synthetic-demo
```

- Unlabeled pretraining: 64 generated subjects × 3 days = 192 windows (generation seed 42).
- Pretraining validation: 16 separate generated subjects × 3 days = 48 windows (seed 43).
- Downstream evaluation: 80 additional generated subjects × 3 days = 240 windows (seed 44).
- Downstream labels: two synthetic regimes with deliberately different distributions of baseline, response amplitude and recovery. These are not disease labels.
- Five subject-grouped folds, repeated twice. All model comparisons use identical folds. Metrics are over held-out daily windows.
- Logistic regression with training-fold standardization, L2 regularization, C=1, lbfgs, maximum 1000 iterations. No probe hyperparameter search.
- Full model and no-dynamics ablation use the same initialization/training seed, data, ten epochs and batch size 32. Each selects its best pretraining-validation-loss checkpoint; both selected epoch 10.
- The frozen random encoder uses the same encoder initialization seed as training. Its classifier is trained on the downstream training folds, just like the other methods.

## Results

Values are percentages, mean ± standard deviation across ten folds. AP means scikit-learn average precision; it is not trapezoidal integration of a precision–recall curve. Fold standard deviations are not confidence intervals.

| Representation | Average precision | ROC-AUC | Macro-F1 |
|---|---:|---:|---:|
| Six glucose summaries | 87.26 ± 5.08 | 85.07 ± 8.05 | 76.11 ± 6.09 |
| Frozen random encoder | 88.08 ± 4.96 | 86.58 ± 6.32 | 77.82 ± 8.31 |
| Pretrained encoder, both losses | 88.74 ± 4.21 | 86.94 ± 5.65 | 77.80 ± 7.77 |
| Pretrained encoder, dynamics weight zero | 89.42 ± 4.53 | 87.59 ± 5.70 | 77.71 ± 8.33 |

The full model gains only about 0.65 AP points over the random encoder and 1.47 over glucose summaries. Removing the dynamics objective produces slightly higher AP in this run. Macro-F1 is effectively similar between the learned and random encoders. These results do not establish a reliable advantage from pretraining or the dynamics objective. The toy data contain easy level/shape cues and do not substitute for real metabolic cohorts.

The purpose of preserving this table is to distinguish a functioning training pipeline from evidence of a useful clinical model. No tuning was performed to make the full model win this demonstration.

## Training and implementation evidence

The full model's average training loss decreased from 0.5224 to 0.1830, and its validation loss from 0.4189 to 0.1612. The learned Gaussian bandwidth changed from six grid steps at initialization to approximately 5.869 at epoch 10. Loss values cannot be compared directly against the ablation's totals because the objectives differ.

The model in this run had **638,466 trainable parameters** and **1,077,636 total pretraining parameters**.

Since this run, the patch convolution takes the observation mask as a second input channel, which adds 3 x (64 + 16 + 48 + 48) = 528 weights per encoder. The **current architecture measures 638,994 trainable and 1,078,692 total parameters** (+528 trainable, +1,056 total, the latter counting the EMA target copy), versus the paper's reported 0.72M and 1.18M. The difference from the paper reflects the explicitly documented low-level architectural assumptions. It is not claimed to be an exact parameter-for-parameter replica.

The full and ablation training loops took approximately 6.14 and 6.04 seconds, respectively, on the recorded local CPU runtime with two PyTorch intra-op threads. These times include the small training/validation/checkpoint loops but exclude Python startup, dataset generation and downstream probe evaluation. They are not hardware benchmarks or estimates of the paper's training cost.

Runtime: Python 3.12.4, PyTorch 2.14.0, NumPy 2.5.3, macOS 26.6.2 on arm64. Exact installed dependencies are recorded in `environment-pins.txt`. MPS and CUDA execution were not tested.

## Verification

All **25 tests passed** on CPU at the time of this run. Tests exercised mask and hidden-input isolation, causal filtering with fixed normalized input, rate-of-change gaps, weighted targets, gradients through both streams and fusion, EMA freezing/update behavior, subject separation, and a train/save/reload/feature-extraction round trip. The current suite and CI status are available from the [README](../README.md).

The standalone `glucofm probe` command also loaded the saved best checkpoint and reproduced the demo's three non-ablation metric summaries exactly. Python compilation and command help were checked. GitHub Actions had not run at the time of this historical experiment; current runs are listed in [Actions](https://github.com/ViSaReVe/glucofm-replication/actions/workflows/tests.yml).

Machine-readable evidence is in `synthetic-evaluation.json`, `full-history.json`, `without-dynamics-history.json`, and the two training summary files. Subject IDs in these artifacts refer only to generated subjects. Model checkpoints remain in the local ignored `runs/synthetic-demo/` directory and are not included in the Git snapshot.

## Subsequent real-data evaluation

The subsequent [ShanghaiT2DM-to-CGMacros experiment](real-data.md) completed a subject-disjoint evaluation on public CGM recordings. That report contains the real-data results, dataset provenance, baselines, variability, and protocol limitations. This page preserves the earlier synthetic experiment.
