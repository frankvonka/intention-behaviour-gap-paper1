# Phase 15 — Profile Structure Comparison (K=2 to K=7)

## Purpose
Numerically compare the existing K=2..K=7 solutions
(dynamically read from Phase 04 model_fit.csv). Descriptive and
computational only. No K selection, no profile labels, no interpretation.

## Input Result Files
- results/02_measurement/construct_scores.csv (construct scores)
- results/04_lpa_estimation/K_7/posterior_probabilities.csv (class assignments)
- results/04_lpa_estimation/K_7/profile_sizes.csv (profile sizes)
- results/04_lpa_estimation/model_fit.csv (fit statistics)

## Calculations
1. **all_k_profile_structure.csv** — per-K, per-profile N, %, mean INT, mean BE, INT-BE
2. **all_k_gap_configuration.csv** — signed INT-BE differences with direction flags
3. **profile_split_comparison.csv** — consecutive-K splitting via nearest
   standardized profile mean vector (Euclidean distance, 10 constructs)
4. **gap_information_by_k.csv** — counts of INT>BE / BE>INT profiles, range
   and variance of profile-level INT-BE means
5. **profile_size_by_k.csv** — min/max N and percentage per K
6. **boundary_by_k.csv** — per-profile INT/BE variance, min, max; zero-variance flags
7. **model_fit_summary.csv** — LL, AIC, BIC, entropy, n_params, sizes (existing results)
8. **phase15_master.csv** — key per-K summary

## Distance Metric and Standardization
- Profile mean vectors computed across the 10 constructs
  (ATT, CON, SNO, COVID, INT, BE, PU, PEU, PO, PRI)
- Standardized using respondent-level grand mean and SD per construct:
  z = (profile_mean - grand_mean) / grand_SD
- Euclidean distance between standardized vectors
- Nearest-neighbor assignment (argmin distance); no manual assignment

## No Data Modification / No Model-Selection Change
All numbers read from existing verified result files. No refitting.
No dataset changes. No plots. No literature.
