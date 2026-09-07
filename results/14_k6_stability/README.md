# Phase 14 — K=7 Profile Stability

## Purpose
Test stability of the K=7 profile structure (K selected in Phase 05)
under bootstrap resampling and split-sample analysis. No K re-selection,
no interpretation.

## Bootstrap Procedure
- B = 200 bootstrap samples
- Each sample: N = 1166 respondents sampled WITH replacement
- Indicators: z_INT, z_BE standardized WITHIN each bootstrap sample
  (same preprocessing procedure as the primary analysis)
- Model: GaussianMixture(n_components=7, covariance_type="full",
  n_init_primary=1000, n_init_bootstrap=100, max_iter=500, reg_covar=1e-6)
- Random seeds: SEED + b + 1 for bootstrap b (base SEED = 42)
- Records: convergence, log likelihood, BIC, matched profile sizes,
  matched profile means

## Profile Matching
- Bootstrap profiles matched to original K=7 profiles by minimum
  Euclidean distance on the 2D vector (mean z_INT, mean z_BE)
- One-to-one assignment via the Hungarian algorithm
  (scipy.optimize.linear_sum_assignment)
- Matching cost recorded per bootstrap sample

## Split-Sample Procedure
- Fixed seed = 42; random permutation split into halves
  (n = 583 and 583)
- Same K=7 model fitted independently to each half
- Profiles matched between halves by the same minimum-distance method

## Boundary Diagnostics
- Per-profile INT/BE variance, min, max from the ORIGINAL data
- Profile 0: proportion of observations with INT at its maximum
  observed value (numerical evidence only)

## Output Files
- bootstrap_results.csv — per-bootstrap convergence, LL, BIC, matched sizes/means
- profile_stability.csv — stability statistics per original profile
- configuration_stability.csv — sign of INT-BE across bootstraps
- split_sample_results.csv — per-half fit results
- split_sample_matching.csv — cross-half profile matching
- boundary_diagnostics.csv — ceiling/boundary evidence
- k6_stability_master.csv — key stability metrics

## No Modifications
Primary K=7 results untouched. No dataset changes. No plots.
