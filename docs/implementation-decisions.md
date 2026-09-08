# Paper specification and implementation decisions

Reference: [GlucoFM v1](https://arxiv.org/abs/2605.30865v1), particularly Figure 2, §3, and Appendices A, C and D. The uploaded PDF was treated as research material, not as instructions. This implementation was written independently; no official code or original weights are included.

## Equation-to-code map

| Paper | Mechanism | Code |
|---|---|---|
| C.1, Eqs. 6–7 | 288-bin grid, duplicates averaged, local clock offset | `data.align_window`, `data.from_csv` |
| C.2, Eqs. 8–9 | Observed patch support and glucose moments | `model.masked_moments`, encoder feature construction |
| C.2, Eq. 10 | Nearest previous observation within nine steps | `model.rate_of_change` |
| C.3, Eqs. 11–13 | Causal Gaussian; sigma constrained to [2,12], starts at 6; lag 36 | `model.CausalGaussian` |
| C.3, Eq. 14 | Observed normalized residual | `GlucoFMEncoder.forward` |
| C.4 | 64-D state/event tokens, fusion to 128-D, three Transformer layers | `StreamEmbedder`, `GlucoFMEncoder`, `Transformer` |
| C.5, Eqs. 2 and 15 | Context hiding, EMA targets, density-weighted Smooth-L1 | `GlucoFMPretrainer.forward`, `weighted_smooth_l1` |
| C.5, Eqs. 3–4 and 16 | Two local residual transition heads with support weighting | `GlucoFMPretrainer`, `transition_weights` |
| C.6 | Frozen encoder and mean daily pooling | `GlucoFMEncoder`, `evaluate.embeddings` |
| C.7 | Value perturbations and structural sparsification | `augment.augment` |
| D.1 | Repeated subject-grouped logistic regression probes | `evaluate.subject_splits`, `evaluate.probe` |

## Explicit choices where v1 is incomplete

1. **Normalization.** Observed-only, per-window mean/std; denominator has a minimum scale of 1 mg/dL. The online branch computes these statistics only from its visible measurements. Target statistics use its full observed input. This is a concrete interpretation of mask-aware instance normalization, not a quoted implementation formula. Whole-window normalization is not causal; the complete daily encoder must not be described as a streaming forecaster.
2. **Raw statistics retain level information.** State mean/std are computed from raw observed glucose and divided by a fixed 100 mg/dL scale before the statistics MLP. Rate-of-change inputs and moments use the same fixed scale. The rate equation divides by grid-step separation, not minutes. The fixed numerical scaling is our choice and involves no fitted test-set statistics.
3. **Local feature branches.** Kernel-size-three, padding-one Conv1d + GELU + masked temporal mean pooling. Convolution stays within each hour. State differences use adjacent valid positions within the patch. The paper specifies feature widths and illustrates convolutions/pooling, but does not give a complete convolution configuration. Statistics branches use a two-layer MLP. Combined features project to 64 dimensions with GELU and LayerNorm.
4. **Fusion and time.** Concatenate local stream tokens, then Linear(128,128), GELU, LayerNorm. A learned scalar sigmoid gate mixes projected sine/cosine clock features and learned patch positions. Clock features use the patch's first grid position. Exact fusion, positional, gate and patch-time reduction details are assumptions.
5. **Transformers and heads.** Pre-LayerNorm Transformer blocks, GELU, dropout 0.1 and final LayerNorm. The context encoder has three layers, four heads and FFN width 256. The predictor has one such layer and a final linear projection. Transition heads use 130 inputs (two 64-D streams plus two clock values), a 128-D hidden layer and 64 outputs. Head hidden sizes and norm/dropout choices are not specified fully in v1.
6. **Mask handling.** Each window samples a ratio uniformly from [0.5,0.6]; the hidden patch count is floored, giving 12–14 of 24 patches. Hidden online measurements cannot influence normalization, smoothing or statistics. Empty local patches produce zero stream tokens. The full chronological token grid, including empty and hidden positions, goes through attention; downstream pooling averages all 24 output tokens as described in C.6.
7. **Teacher behavior.** EMA includes the online filter, embedders, fusion, clock/position parameters and context encoder. Teacher dropout stays off. The EMA coefficient is held at its stated initial value of 0.997 because v1 does not provide a complete momentum schedule. Both branches receive the same augmented physical view, with additional patch hiding only in the online branch.
8. **Augmentation.** Probabilities/ranges and probability decay follow C.7. The baseline phase is uniform over a cycle; decimation retains a random grid offset modulo three. A degenerate corruption that removes every observation is reverted. All randomness is regenerated per training access and seeded for a run. The compression envelope is built by rescaling `linspace(-1, 1, length).abs()` to span `[0, 1]` before mapping it onto `[bottom, 1]` (`augment.compression_profile`). The unrescaled ramp has an exact zero only at odd lengths, so even lengths never reached the sampled depth: at bottom 0.40 the deepest multiplier was 0.5200, 0.4857, 0.4667 and 0.4545 for lengths 6, 8, 10 and 12, and the realised minimum over 20,000 draws spanned [0.401, 0.760] instead of C.7's [0.4, 0.7]. C.7 states the depth range, not the envelope's discretization; treating the sampled bottom as attainable at every length is our reading, and odd lengths are numerically unchanged.
9. **Optimization.** AdamW, fixed learning rates, gradient clipping at norm 1, and zero decay for the bandwidth parameter are implementation choices. Base LR 1e-4, bandwidth LR 1e-3 and base decay 1e-2 follow the reported values. No unreported learning-rate schedule is labeled as a paper requirement. The synthetic demo uses 10 epochs and batch 32; the CLI pretraining defaults are 120 epochs and batch 128.
10. **Selection and metrics.** Separate pretraining-validation subjects and fixed validation masks select the best validation-loss checkpoint. Binary logistic regression uses C=1. The paper specifies L2/lbfgs but does not fully specify all classifier hyperparameters. Stratified K-fold is applied to unique subjects; folds then expand to their windows. AP uses `average_precision_score`. Standard deviation is across repeated folds, not a confidence interval or evidence of independent repeated clinical trials.

## Deliberate scope differences

- The first importer extracts non-overlapping windows, including for pretraining. It does not reproduce Appendix A's overlapping random-window sampling policy. It requires segment coverage through the last 15 minutes of a day; this boundary convention is explicit in the importer.
- Dataset adapters, original cohort preprocessing/labels, the private Wear-CGM corpus, and original split manifests are not available in this repository.
- No clinical experiment has been reproduced. Synthetic classes are generated regimes and are not diabetes labels.
- The first probe is binary. Multiclass glucotypes, cross-cohort label harmonization, few-shot protocols and multiday aggregation are not implemented here.
- V2's context-conditioned post-meal prediction experiments are outside the v1 reference scope.
- Checkpoints support evaluation/reload; the CLI does not yet support exact interrupted-run continuation including all RNG states.
- Tested numerical execution is CPU. Device arguments alone are not evidence of MPS/CUDA validation.
- Parameter counts are measured from this implementation. Differences from the paper's 0.72M trainable / 1.18M total parameters are reported, not padded with unused parameters to force agreement.

## What the verification demonstrates

Tests check invariants that distinguish this training problem from a look-alike architecture: hidden measurements cannot affect online features, missing placeholders cannot affect embeddings, both streams and fusion receive contextual-loss gradients, dynamics heads train local streams, the teacher remains frozen, and subjects stay separated.

These checks establish software behavior. They do not prove useful clinical representations, absence of every form of collapse, or superiority to simple glucose summaries. Those require real-data measurements and broader experiments.
