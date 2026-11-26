# Sharing Aggregated Patient Counts in Place of Line-Level EHR Data

This repository implements a Bayesian count-inference pipeline for "cube" releases — tables of aggregate counts over every subset of a set of categorical grouping variables, with cells below a suppression threshold withheld. 

The pipeline infers the suppressed cells, and the same reconstruction is used to (1) measure statistical fidelity of the cube against the original line-level data, (2) evaluate analytical utility on descriptive, inferential, and predictive tasks, and (3) quantify how much suppressed information the reconstruction recovers, i.e., the privacy risk of threshold suppression. CTGAN synthetic data serves as the comparator throughout, averaged over as many synthetic datasets as there are posterior draws so that it parallels the cube's posterior-mean reconstruction.

## Repository structure

```
src/
├── infer_counts.py    # Count inference pipeline
└── evaluate.py    # Fidelity & Utility Evaluation
```

## Installation

Python ≥ 3.12:

```
pip install -r requirements.txt
```

`cmdstanpy` additionally requires a one-time CmdStan installation:

```python
import cmdstanpy
cmdstanpy.install_cmdstan()
```

## How it works

`run_complete_bayesian_reconstruction()` in `src/infer_counts.py` converts line-level data to cube (at the specified threshold) and runs the full count inference pipeline.

Every fidelity/utility statistic uses one common uncertainty procedure (`src/evaluate.py`): cube reconstruction or CTGAN sample is held fixed while the reference data are resampled with replacement over shared bootstrap indices; each statistic is reported as a bootstrap mean with a 95% percentile confidence interval.

The two releases are reduced to a single fixed surrogate in the same way:

| | Draws | Surrogate |
|---|---|---|
| Cube | x posterior atomic-count vectors | mean counts, re-aggregated to one reconstruction |
| CTGAN | x synthetic datasets of n rows each | mean counts per variable combination, rounded to rows exactly as the cube is |

where x is the number of posterior count vectors (`evaluate.n_posterior_draws(rec)`) and n the cohort size. Every comparator accepts the CTGAN argument either as a single synthetic DataFrame or as the averaged counts returned by `evaluate.ctgan_posterior_mean`.

## Usage

The input is a pandas DataFrame in which each row is one record and each column a categorical variable.

### 1. Reconstruct a suppressed cube

```python
import src.infer_counts as ifc

rec = ifc.run_complete_bayesian_reconstruction(
    df_line_level=df,                     # line-level DataFrame, categorical columns only
    demographic_cols=df.columns.tolist(), # the cube's grouping variables
    suppress_threshold=9,                 # counts <= 9 (i.e., below k = 10) are suppressed
    iter_sampling=200,                    # NUTS sampling iterations per chain
    iter_warmup=100,                      # NUTS warmup iterations per chain
    chains=2,
    mh_iters=2000,                        # constrained MH iterations (thinned by 10)
)

# Per-cell summary: Published / Deterministic (exactly recovered) / Uncertain (inferred)
table = ifc.display_complete_powerset_table(rec)
table["Status"].value_counts()
```

The returned dict `rec` contains the constraint system, ILP bounds, posterior samples of the atomic-cell counts, and the original line-level DataFrame; it is the object every evaluation function below takes. It can be pickled and reloaded to avoid re-running the (expensive) reconstruction.

### 2. Generate the CTGAN comparator

```python
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)
synthesizer = CTGANSynthesizer(metadata)   # default hyperparameters
synthesizer.fit(df)

import src.evaluate as evaluate

# Mean counts over as many CTGAN datasets (n = len(df) rows each) as the cube has posterior draws
synthetic_df = evaluate.ctgan_posterior_mean(
    synthesizer,
    n_rows=len(df),
    n_draws=evaluate.n_posterior_draws(rec),
    demographic_cols=df.columns.tolist(),
)
```

`synthetic_df` is a `pd.Series` of mean counts indexed by variable combination. Passing a single sample (`synthesizer.sample(len(df))`) instead reproduces a one-draw comparison.

### 3. Evaluate fidelity

```python
cols = df.columns.tolist()

# One shared set of bootstrap resample indices for all statistics
idx = evaluate.make_boot_indices(len(df), n_boot=2000, seed=0)

# Suppressed-count MAE: reconstruction vs. impute-0 / impute-5 / random / CTGAN baselines
mae = evaluate.compare_suppressed_mae(df, rec, synthetic_df, cols,
                                suppress_threshold=9, n_boot=2000, seed=0)

# Jensen–Shannon divergence, per marginal and joint
dist = evaluate.compare_distribution_fidelity(df, rec, synthetic_df, cols, boot_idx=idx)
fig = evaluate.plot_distribution_fidelity_compare(df, rec, synthetic_df, cols,
                                            labels=("Original vs. Cube", "Original vs. CTGAN"),
                                            boot_idx=idx)

# Cramér's V per variable pair, and mean |difference| across pairs
assoc = evaluate.compare_association_fidelity(df, rec, synthetic_df, cols, boot_idx=idx)
assoc_mae = evaluate.compare_association_mae(df, rec, synthetic_df, cols, boot_idx=idx)

# R² between original and surrogate atomic-cell count vectors
r2 = evaluate.compare_r2(df, rec, synthetic_df, cols, boot_idx=idx)
```

### 4. Evaluate utility

```python
# Marginal probabilities per category, and MAE across categories
probs = evaluate.compare_marginal_probs(df, rec, synthetic_df, "column_name", cols, boot_idx=idx)
prob_mae = evaluate.compare_marginal_prob_mae(df, rec, synthetic_df, cols, boot_idx=idx)

# Subgroup odds ratios for a binary outcome column: log-OR MAE and direction flips,
# plus how many flips involve associations significant in the original (Fisher's exact)
odds = evaluate.compare_odds_ratios(df, rec, synthetic_df, cols, "outcome_col", boot_idx=idx)
flip_summary, flip_detail = evaluate.flipped_odds_ratio_significance(
    df, rec, synthetic_df, cols, "outcome_col", alpha=0.05, boot_idx=idx)

# Classification: train on original / reconstructed cube / CTGAN, evaluate on one held-out test set
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42,
                                     stratify=df["outcome_col"])
rec_train = ifc.run_complete_bayesian_reconstruction(
    df_line_level=train_df, demographic_cols=train_df.columns.tolist(), suppress_threshold=9)
# ... fit CTGAN on train_df as in step 2, then average over
#     evaluate.n_posterior_draws(rec_train) draws of len(train_df) rows -> synthetic_train_df ...
tidx = evaluate.make_boot_indices(len(test_df), n_boot=2000, seed=0)
clf = evaluate.compare_classification(train_df, test_df, rec_train, synthetic_train_df,
                                "outcome_col", cat_cols=["categorical_col_1", "categorical_col_2"],
                                demographic_cols=train_df.columns.tolist(), boot_idx=tidx)
```
`cat_cols` lists only the **multi-level** categorical predictors (those are the columns that need label-encoding before the logistic regression).
