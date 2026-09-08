# Connecting GlucoFM to biosignal processing

Sliding windows are still part of this approach. They define the unit the model sees: a 24-hour glucose window, split into one-hour patches. The foundation-model contribution is how the model learns a reusable description inside that window and how that description is evaluated across tasks and people.

| Familiar biosignal idea | Connection here | Important distinction |
|---|---|---|
| Windowing a recording | Extract days, then hourly patches | Window length determines which dynamics fit in context; it is not the entire learning method |
| Baseline/envelope and faster variation | Causal smoothing and a residual stream | The residual is not a labeled physiological event or an identified cause |
| Handcrafted features | Observed means, variability and rate of change help the tokenizers | The model combines these with learned waveform features |
| Sensor loss or artifacts | Preserve masks and augment drift/dropout patterns | A filled number must not become a physical observation |
| Intersubject variability | Keep subjects separated during evaluation | Randomly splitting windows can reward person recognition rather than transferable task information |
| Supervised classification | Fit a simple classifier on daily embeddings | Pretraining learns without these task labels; the encoder stays frozen during probing |

CGM and sEMG differ in physical origin, relevant timescales, measurement conditions and task definitions. Their preprocessing choices should not be copied mechanically. The transferable reasoning is to identify the signal's structure, preserve what was actually measured, choose a learning objective, and evaluate on the intended unseen population.

## One small exercise before reading the whole training loop

Suppose a one-hour patch is deliberately hidden during pretraining. We first normalize the complete day, smooth it, and compute all patch means. Then we replace that patch's token with a mask token.

Which intermediate quantities could still contain information from the hidden hour? Draw where the mask must be applied instead. Compare your answer with `GlucoFMEncoder.forward` and `test_hidden_readings_cannot_change_online_features_or_tokens`.

This links the signal-processing pipeline directly to the self-supervised task. Correct tensor shapes alone would not catch that information leak.

## Evaluating the representation

The [real-data experiment](../reports/real-data.md) compares pretrained embeddings with a random encoder of the same architecture and six glucose summaries. Using the same subject partitions and classifier for every representation helps distinguish the effect of pretraining from the features available before training.

The results apply to the documented cohorts and preprocessing. Testing transfer to additional populations and sensors requires separate experiments with explicit subject separation.
