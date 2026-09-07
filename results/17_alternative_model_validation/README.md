# Phase 17 — Alternative Model Validation

## Purpose
Sensitivity analysis of the selected-K (K = 7, minimum BIC among
non-degenerate fits per results/05_lpa_selection/selected_model.csv)
INT-BE profile structure under alternative statistical specifications.
No model selection, no interpretation.

## Specifications Tested
1. **Covariance type** (full / diag / spherical), K = 7, indicators =
   all 10 constructs (INT, BE, ATT, CON, SNO, COVID, PU, PEU, PO, PRI),
   standardized across respondents
2. **Score representation** (arithmetic mean scores vs sklearn
   FactorAnalysis 1-component scores for INT and BE), K = 7 full
   covariance on the 2 standardized indicators
3. **Random seeds** 1, 2, 3, 4, 5 — primary 2-indicator (z_INT, z_BE)
   full-covariance specification

## Parameters
- K = 7 (fixed; no re-selection)
- n_init = 1000 per fit
- Base seed = 42; seed list = [1, 2, 3, 4, 5]
- max_iter = 500, reg_covar = 1e-6
- Scaling: respondent-level z-standardization per indicator

## Profile Matching
- Alternative profiles matched to the primary K = 7 reference by
  minimum Euclidean distance on standardized (INT, BE) profile means
- One-to-one assignment via the Hungarian algorithm
  (scipy.optimize.linear_sum_assignment)

## Output Files
- reference_K7.csv / reference_K7_profiles.csv — frozen primary solution
- covariance_sensitivity.csv — full/diag/spherical fits
- score_representation_sensitivity.csv — mean vs factor scores
- seed_sensitivity.csv — seeds 1..5
- gap_direction_sensitivity.csv — signed INT-BE per matched profile
- profile_matching_sensitivity.csv — matched profile + distance
- phase17_master.csv — summary across all specifications

## No Modifications
Previous phase outputs untouched. Dataset unchanged. No plots.
