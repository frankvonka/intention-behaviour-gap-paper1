# Phase 16 — Intention–Behavior Gap Profile Validation

## Purpose
Determine whether observed LPA profiles represent meaningful
INT–BE configurations or mainly reflect overall response level.
Descriptive/computational only. No K selection, no labels, no
interpretation, no plots, no literature.

## Input Result Files
- results/02_measurement/construct_scores.csv
- results/03_gap_analysis/gap_scores.npy
- results/04_lpa_estimation/K_{K}/posterior_probabilities.csv

## Formulas
- GAP_i = z(INT_i) - z(BE_i), where z = (x - mean) / SD
- LEVEL = (z_INT + z_BE) / 2
- Variance decomposition: Var_total = Var_between + weighted Var_within
  - weights = profile sample proportions
  - Var_between = sum(w_p * (mean_p - grand_mean)^2)
  - Var_within = sum(w_p * var_p)
- Cohen's d = (M1 - M2) / pooled_SD
  - pooled_SD = sqrt[((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2)]
- Euclidean distance between centroids: sqrt(d1^2 + d2^2)

## Gap-only clustering
- GaussianMixture(n_components=K_go, covariance_type='full', n_init=1000,
  random_state=42, max_iter=500)
- INDICATOR: respondent-level GAP (1D)
- Model selection: MINIMUM BIC (corrected procedure)
- K_go = 1..7

## Parameters
- N = 1166
- SEED = 42 (explicit, documented)
- N_INIT = 1000
- BIC minimization (NOT maximization — historical bug corrected)

## Output Files
- profile_int_be_by_k.csv
- gap_variability_by_profile.csv
- gap_variance_decomposition.csv
- pairwise_gap_separation.csv
- gap_only_model_comparison.csv
- gap_only_K{1..7}.csv
- gap_only_selected.csv
- level_gap_relationship.csv
- level_gap_correlation.csv
- k6_centroid_geometry.csv
- k6_centroid_distances.csv
- k6_gap_variance_explained.csv
- phase16_master.csv
- README.md

## No Modifications
All calculations read existing result files. No refitting of the
primary multivariate LPA. No dataset changes. No model-selection
change. No plots.
