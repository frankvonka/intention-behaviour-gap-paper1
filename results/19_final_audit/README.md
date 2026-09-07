# Phase 19 — Final Numerical Audit and Results Freeze

## Dataset
- File: data.xls (read-only, never modified)
- N = 1166
- Missing values: 0
- Duplicate rows: 42 (reported, not removed)
- Psychometric items range: 1..5

## Measurement (10 constructs)
Cronbach alpha, mean, SD verified for each construct.
INT-BE Pearson r reproduced at r = 0.6515, p = 8.83e-142.

## GAP Definition
GAP = z(INT) - z(BE), respondent-level.
mean = -0.000000, SD = 0.834916.
n INT > BE = 524 (44.94%)
n BE > INT = 642 (55.06%)

## Primary LPA Specification
- Estimator: sklearn.mixture.GaussianMixture, full covariance
- n_init = 1000, SEED = 42
- K = 2..7 estimated; K = 7 selected

## K-Selection Criterion
Minimum BIC among non-degenerate fits under the primary specified
model (BIC minimization; historical maximization bug corrected),
per results/05_lpa_selection/selected_model.csv.
BIC = -2 LL + k log(n), where k = parameters, n = sample size.

## K=7 Profile Sizes
- Profile 0: N = 281 (24.10%)
- Profile 1: N = 125 (10.72%)
- Profile 2: N = 108 (9.26%)
- Profile 3: N = 264 (22.64%)
- Profile 4: N = 245 (21.01%)
- Profile 5: N = 70 (6.00%)
- Profile 6: N = 73 (6.26%)
- Total = 1166 (100.00%)


## Classification Quality
- Mean max posterior: 0.9105
- Median max posterior: 0.9995
- % >= 0.90: 77.02
- % >= 0.80: 82.16
- % >= 0.70: 84.91
- % <  0.70: 15.09
- % <  0.50: 0.00

## Stability (Phase 14)
Bootstrap resampling (Hungarian-matched profile assignment) and
split-sample validation. Both recorded in results/14_k6_stability/.

## Alternative Specifications (Phase 17)
Covariance, score-representation, and random-seed sensitivity. All
fits converged. Profile matching distances reported.

## Predictor Model (Phase 18)
- 13 predictors x 6 non-reference profiles = 78 tests
- MNLogit, BFGS, converged
- Joint Benjamini-Hochberg FDR across all 78 tests
- 19 / 78 FDR-significant

## Multicollinearity Diagnostics (Phase 18B)
- VIF range: 1.99..67.44
- Condition number (corr): 18.40
- Smallest eigenvalue: 0.2609
- High-corr pairs (|r|>=0.70): 1
- Primary model reproduces Phase 18 to numerical precision

## Known Limitations (Computational)
- Construct predictors exhibit substantial multicollinearity
  (VIF > 26 for all 8 construct predictors)
- Coefficient interpretation should account for this
- Demographic predictors (age, Education, income) VIF > 10
- All numerics reproducible from data.xls and the listed source files

## No Modifications
- No phase outputs were modified.
- No observations removed.
- No predictors removed from the primary model.
- No plots produced.
- No literature search.
- No manuscript text.
