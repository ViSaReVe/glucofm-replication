# Cohort adapters: download, conventions, and label thresholds

Two public cohorts, both downstream cohorts in the GlucoFM paper itself. Neither is
redistributed here. `src/glucofm/datasets/` converts each archive into the canonical
CSV schema (`subject_id,timestamp,glucose_mg_dl,label`) that `data.from_csv` reads;
units, timestamps and subject identifiers are resolved inside the adapter, because
`from_csv` deliberately repairs none of them.

Reading Shanghai workbooks needs an Excel engine:

```bash
python -m pip install -e '.[datasets]'
```

CGMacros is plain CSV and needs nothing beyond the base install.

## CGMacros — downstream cohort

45 participants, ten consecutive days each, two CGMs worn concurrently.

- Source: <https://physionet.org/content/cgmacros/1.0.0/>
- Licence: CC BY-NC-SA 4.0. Open access, no credentialing.
- Citation: Kissiov et al., *CGMacros: a pilot scientific dataset for personalized
  nutrition and diet monitoring*, Scientific Data (2025).

```bash
curl -O https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip
unzip CGMacros_dateshifted365.zip -d cgmacros
```

The archive is 657 MB, almost all of it meal photographs. Only the CSV members are
needed; a partial extraction of just those is about 7 MB.

```bash
glucofm convert --dataset cgmacros --root cgmacros \
  --sensor dexcom --label diabetes --output cgm-dexcom-diabetes.csv
glucofm prepare --csv cgm-dexcom-diabetes.csv --output cgm-dexcom-diabetes.npz \
  --sampling non-overlapping
```

### The two sensors stay separate

A Dexcom G6 Pro (5-minute) and a FreeStyle Libre Pro (15-minute) were worn at the
same time, on the abdomen and the upper arm. `--sensor` selects one; there is no
merged mode. Merging them would put two sampling rates, two calibrations and two
dropout patterns into a single window, and the observation mask — which this model
reads directly — would no longer mean one thing.

### Published series are interpolated, and the observation mask is not recoverable

Every per-subject CSV is on a **one-minute** grid with both sensors linearly
interpolated between their real samples. Row 2 of `CGMacros-001.csv` reads
`84.133333…` one minute after a reading of `84.0`, one fifteenth of the way to the
next Libre value.

Ingesting that grid directly would present interpolated values as observations and
leave the mask almost entirely true, destroying exactly the signal the encoder is
built around. So the adapter estimates which points are real. **It does not recover
the physical observation mask, and no method can:** interpolation is not invertible,
and a real reading that happens to lie on the straight line between its neighbours
is bit-for-bit identical to an interpolated point.

What the published file *does* decide:

| Case | Verdict |
|---|---|
| The series changes slope at the point | real reading |
| First or last present point of a run | real reading |
| Point off the sampling lattice | not reported by the sensor |
| On-lattice, present, collinear with its neighbours | **ambiguous** |

The ambiguous case is genuinely undecidable, and it is where the policies differ.
`--recovery` selects one. Both select *candidate* points; **neither partitions the
series into readings and non-readings, and neither yields a true mask error rate.**

**`slope-change` (default).** Keeps only the provably-real points and excludes every
ambiguous candidate. It never admits an interpolated value.

The counts below are **excluded on-lattice candidates, classified by the geometry of
the published series** — not by known sampling status. How many were real readings is
not identifiable from these files.

| | on-lattice present points | excluded, bracketing readings equal | excluded, spanned by a sloped straight segment longer than one sampling period | excluded, other |
|---|---:|---:|---:|---:|
| Dexcom | 126,025 | 6,115 (4.85%) | 14,989 (11.89%) | 1,019 (0.81%) |
| Libre | 45,894 | 1,304 (2.84%) | 3,139 (6.84%) | 304 (0.66%) |

The middle column is *consistent with* interpolation across a sensor dropout, but
does not prove it: a run of real readings that happen to lie on a line produces the
same geometry. The first column is *consistent with* a plateau of real readings, and
equally does not prove one.

What the numbers do establish is the **shape** of the bias, which does not depend on
resolving the ambiguity: exclusions concentrate where the series is flat. So whatever
the true rate, the loss is signal-dependent — missingness becomes correlated with
glucose being flat, in a model that reads the mask as an input channel. That is a
real defect of the default policy, and its magnitude is unquantified.

An earlier version of this document put the loss at "about 0.1% of Libre readings",
called the middle column "correctly excluded as dropout interpolation", and stated an
exact mask undercount. All three were wrong: the first came from one subject, and the
others asserted a distinction the same paragraph had just called unidentifiable.

**`lattice`.** Additionally admits an ambiguous candidate when the two bracketing
readings are equal, which yields 5.8% more Dexcom and 3.1% more Libre candidate
points than the default. Note precisely what that buys: the admitted value equals its
neighbours, but **equal endpoints do not establish what an unobserved measurement
between them would have been** — an excursion and return inside one sampling interval
is possible. The admitted number is an interpolated estimate, not a recovered
measurement, and admitting it over-counts the mask wherever the sensor did not in
fact sample there.

Phase inference needs at least three anchors. A wholly flat trace yields only its two
run endpoints, so `lattice` cannot locate the sampling lattice and falls back to the
`slope-change` result rather than guessing a phase; plateau candidates are not
admitted in that case. It does not recover all plateaus.

**The recorded experiment used `slope-change`**, which is why it is the default:
changing it changes the prepared datasets and would require regenerating
[the real-data results](../reports/real-data.md). Neither policy recovers ground
truth, and neither one's error rate against the physical mask is identified by these
files. What is stated above is what each policy selects and how its exclusions are
distributed. `reports/real-data.md` records the policy in force.

Yields under the default: 105,895 Dexcom readings and 41,510 Libre readings, from
629,825 and 687,360 interpolated one-minute points respectively.

### Units and timestamps

Both glucose columns are already mg/dL per the published data dictionary; no
conversion is applied. The adapter still range-checks every series and refuses one
that leaves 20–600 mg/dL rather than guessing at a unit.

Timestamps are timezone-naive local clock time. De-identification shifted whole
dates by ±365–720 days, which moves the calendar date and **leaves time of day
intact**, so the local clock index this model conditions on survives the shift. No
timezone conversion and no DST rule is applied; an offset-aware timestamp is
rejected rather than coerced.

Subject ids become `cgmacros-001` … `cgmacros-049`. Numbering runs to 49 with 45
completers, so 024, 025, 037 and 040 are absent. The bare integers would collide
with Shanghai patient numbers, hence the prefix.

### Labels

All four come from `bio.csv`, are subject-level, and are binarised for the binary
probe. Values carrying a lab flag (`2.5 (low)`) are parsed; the sentinels the data
dictionary documents as calculation errors — `LDL (Cal) = 800`, `VLDL (Cal) = 400`,
`Cho/HDL Ratio = 400` — are treated as missing.

#### Partial panels

Rules are evaluated in **three-valued logic**, so a missing input is `None` rather
than `False`:

- **A satisfied threshold settles the rule.** Hyperlipidemia is a disjunction, so a
  subject over any one of the three cutoffs is positive even if another input of the
  same rule is missing. A known positive stays positive on a partial panel.
- **Insufficient evidence is undecided, never negative.** With no threshold met and
  at least one input missing, the subject is excluded from that label's partition —
  not imputed, and not labelled negative.

This matters because the natural spelling of the rule gets it wrong: `nan >= 160.0`
is `False` in Python, so a plain comparison silently converts an undecidable panel
into a confident negative. `LabelRule.evaluate` keeps the two apart, and
`resolve_labels` returns the undecided subjects alongside the decided ones so the
exclusions are visible; `glucofm convert` prints them.

On the published cohort this policy changes nothing. Exactly one subject
(`cgmacros-012`) has an unusable input — the `LDL (Cal) = 800` sentinel — and its
triglycerides of 1150 mg/dL clear the 200 threshold outright, so it is decided
positive either way. All 45 subjects are decided for all four labels. Excluding that
subject instead is a sensitivity analysis, not a correction; doing so moves
hyperlipidemia AP by several points in a 45-subject cohort, which says more about the
cohort's size than about the label.

| Label | Rule | Source | Positives / 45 |
|---|---|---|---:|
| `diabetes` | HbA1c ≥ 6.5% | ADA diagnostic threshold | 14 |
| `obesity` | BMI ≥ 30 kg/m² | WHO obesity class I | 23 |
| `insulin_resistance` | HOMA-IR ≥ 2.5 | **no consensus cutoff — see below** | 32 |
| `hyperlipidemia` | TC ≥ 240 **or** LDL ≥ 160 **or** TG ≥ 200 mg/dL | NCEP ATP III "high" | 12 |

HOMA-IR is computed as fasting insulin (µIU/mL) × fasting glucose (mg/dL) / 405,
the standard formula and the one the dataset authors use in their own analysis
notebook.

**Two of these thresholds are genuinely unsettled, and neither is invented here.**

1. **HOMA-IR.** There is no consensus cutoff for insulin resistance. 2.5 is widely
   used and is what this adapter applies, but the literature also supports 2.6,
   2.73, and population-specific values as high as 3.8 (Ascaso et al. 2003). The
   cohort's HOMA-IR runs 0.56–16.96 with a median of 4.24, so the choice moves the
   class balance materially: at 2.5 it is 32/45 positive. Any comparison across
   papers must state the cutoff. This label is the least portable of the four.
2. **Hyperlipidemia.** NCEP ATP III defines both a "high" and a "borderline-high"
   band, and "hyperlipidemia" is used in the literature for either. This adapter
   uses the "high" set above. The borderline-high set (TC ≥ 200, LDL ≥ 130,
   TG ≥ 150) is equally defensible and would move far more subjects into the
   positive class.

`diabetes` and `obesity` are not ambiguous: ADA and WHO fix them.

**One documentation error in the source.** `DataDictionary_Bio.csv` labels
`A1c PDL (Lab)` as mmol/mol, but gives its range as 4.6–8.5, which is NGSP percent;
mmol/mol would put the same cohort near 27–69. The observed values are 4.6–8.5. The
adapter treats the unit label as an error, reads the column as percent, and raises
if any value falls outside 3–20% rather than silently misclassifying a cohort.

## ShanghaiT2DM — pretraining corpus

- Source: <https://figshare.com/collections/Diabetes_Datasets_ShanghaiT1DM_and_ShanghaiT2DM/6310860>
  (DOI [10.6084/m9.figshare.21600933](https://doi.org/10.6084/m9.figshare.21600933))
- Licence: CC BY 4.0.
- Citation: Zhao et al., *Chinese diabetes datasets for data-driven machine
  learning*, Scientific Data 10, 35 (2023).

```bash
curl -L -o diabetes_datasets.zip https://ndownloader.figshare.com/files/42966622
unzip diabetes_datasets.zip -d shanghai

glucofm convert --dataset shanghai --root shanghai --cohort T2DM --output shanghai-t2dm.csv
glucofm prepare --csv shanghai-t2dm.csv --output shanghai-t2dm.npz \
  --sampling pretraining --sampling-seed 0
```

The archive is 3.7 MB. `--cohort T1DM` reads the 12-patient type 1 folder instead.

### Structure and conventions

Per-recording Excel workbooks named `<patient>_<period>_<startdate>.xls[x]`. A
patient may hold several recording periods; they share one subject id, so
subject-disjoint partitioning cannot be defeated by splitting one person across
partitions. Measured: **100 patients, 109 recording periods, 112,462 CGM readings,
about 28,095 recording hours**, sampled every 15 minutes (112,355 of 112,366
inter-sample gaps are exactly 15 minutes).

Header spelling is not uniform. 107 of the 109 T2DM workbooks name the series
`CGM (mg / dl)`; two (`2045_0_20201216.xls`, `2095_0_20201116.xls`) name it `CGM `
with no unit at all. The adapter matches the column by prefix and then checks the
units against the values: both unlabeled files span 64.8–325.8 and 45.0–394.2, which
is unambiguously mg/dL. A series that reads as mmol/L is **refused, not converted** —
a silent ×18 on a misread file is indistinguishable from real data.

Timestamps are timezone-naive local clock time (Beijing, UTC+8). Mainland China has
observed no daylight saving since 1991 and every recording starts in 2019 or later,
so local clock time is continuous across each period: no DST gap or repeated hour
can occur, and nothing is converted.

Subject ids become `shanghaiT2DM-2000` … ; patient numbers repeat between the T1DM
and T2DM folders, hence the cohort in the prefix.

Every emitted row has `label = -1`. The cohort's diabetes type is a property of the
folder rather than a within-cohort contrast, and clinical labels must not reach
pretraining batches regardless.

### Sampling mode

Pretraining uses `--sampling pretraining` (Appendix A.2 overlapping windows). On
this cohort that is the difference between 1,093 windows covering 69 distinct clock
phases and 2,157 windows covering all 288. Downstream CGMacros partitions stay
`non-overlapping`. That is not a leakage guard — cross-validation is subject-grouped,
so a subject's windows never straddle a fold however they are cut. It controls
*weighting*: overlapping windows would count the same hours of a subject repeatedly
within a fold, over-weighting densely recorded subjects.

Partition by subject **before** windowing, with `glucofm partition`, so the
pretraining and validation sides can take different sampling modes:

```bash
glucofm partition --csv shanghai-t2dm.csv \
  --output-a shanghai-pretrain.csv --output-b shanghai-validation.csv \
  --fraction-b 0.2 --seed 0
glucofm prepare --csv shanghai-pretrain.csv --output pretrain.npz \
  --sampling pretraining --sampling-seed 0
glucofm prepare --csv shanghai-validation.csv --output validation.npz \
  --sampling non-overlapping
```

Splitting an already-windowed NPZ instead forces one mode on both sides. The
recorded run did exactly that and so used overlapping validation windows; see
`docs/implementation-decisions.md`, assumption 11.

## What the adapters do not do

- No imputation. A missing reading stays missing; the model is mask-aware.
- No merging of sensors or cohorts.
- No unit inference from a header alone; values decide, and an implausible series
  raises.
- No timezone conversion, and no DST handling, because neither cohort needs one —
  both are already local naive. An offset-aware timestamp is rejected.
- No redistribution of cohort data, and no cached copies in this repository.
