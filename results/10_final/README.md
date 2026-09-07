# Final Frozen Results Package
## Intention–Behaviour Gap in Household Energy-Saving Behaviour

### Overview
This directory contains the verified outputs from all 10 computational phases.
All values are recalculated from `data.xls`; no historical results were hard-coded.

### Selected Model
- **K = 7** profiles (selected by minimum BIC, verified in Phase 05)
- Covariance: full
- Indicators: z_INT, z_BE (standardized across respondents)
- N = 1166 respondents
- Random seed = 42, n_init = 1000

### File Organization

#### data_diagnostics/
- `data_dimensions.csv` — N rows, N columns (Phase 01)
- `column_information.csv` — column dtype, unique counts, missing (Phase 01)
- `missing_values.csv` — per-column missing value counts (Phase 01)
- `duplicate_information.csv` — duplicate row counts (Phase 01)
- `item_ranges.csv` — per-item N, missing, unique, min, max, mean, SD (Phase 01)
- `dataset_metadata.json` — workbook metadata (Phase 01)

#### measurement_results/
- `item_statistics.csv` — per-item mean, SD, min, max, variance (Phase 02)
- `construct_statistics.csv` — construct mean, SD, Cronbach alpha (Phase 02)
- `cronbach_alpha.csv` — per-construct Cronbach alpha (Phase 02)
- `int_be_correlation.csv` — Pearson r, p, N, 95% CI (Phase 02)
- `construct_scores.csv` — respondent-level construct scores (Phase 02)
- `construct_correlation_matrix.csv` — all-pairs construct correlations (Phase 02)

#### gap_results/
- `gap_scores.npy` — respondent-level GAP = z_INT - z_BE (Phase 03)
- `standardized_scores.csv` — z_INT, z_BE, GAP per respondent (Phase 03)
- `gap_statistics.csv` — summary gap statistics (Phase 03)

#### lpa_model_comparison/
- `model_fit.csv` — AIC, BIC, entropy, convergence for K=2..6 (Phase 04)
- `K_7/` — selected-model artifacts:
  - `profile_parameters.csv` — N, proportion, means per profile (Phase 04)
  - `profile_sizes.csv` — class sizes (Phase 04)
  - `covariance_matrices.json` — full covariance per profile (Phase 04)
  - `posterior_probabilities.csv` — per-respondent posteriors (Phase 04)
  - `profile_means.csv` — mean vectors (Phase 04)
  - `random_start_diagnostics.csv` — convergence diagnostics (Phase 04)

#### selected_lpa/
- `model_comparison.csv` — full comparison K=2..6 with deltas (Phase 05)
- `stability.csv` — convergence across random seeds (Phase 05)
- `classification_uncertainty.csv` — max-posterior thresholds (Phase 05)
- `selection_diagnostics.csv` — decision diagnostics (Phase 05)
- `selected_model.csv` — selected K and basis (Phase 05)

#### profile_parameters/
- `profile_sizes.csv` — N and percentage per profile (Phase 06)
- `profile_means.csv` — mean and SD per construct per profile (Phase 06)
- `gap_profile_check.csv` — standardized gap and pattern per profile (Phase 06)

#### profile_comparisons/
- `profile_comparisons.csv` — Welch t-tests between profiles (Phase 06)
- `profile_effect_sizes.csv` — Cohen's d between profiles (Phase 06)

#### profile_predictors/
- `coefficients.csv` — beta, SE, z, p per coefficient (Phase 07)
- `odds_ratios.csv` — OR = exp(beta) (Phase 07)
- `confidence_intervals.csv` — 95% CI for OR (Phase 07)
- `p_values.csv` — adjusted and raw p-values (Phase 07)
- `fdr_results.csv` — Benjamini-Hochberg FDR-adjusted p-values (Phase 07)

#### robustness_results/
- `random_start_stability.csv` — n_init=1000 stability (Phase 08)
- `random_seed_sensitivity.csv` — seed [42,123,999,2024,7] (Phase 08)
- `covariance_sensitivity.csv` — full/diag/spherical comparison (Phase 08)
- `score_sensitivity.csv` — mean vs factor score correlation (Phase 08)
- `classification_sensitivity.csv` — max-posterior thresholds per K (Phase 08)
- `outlier_sensitivity.csv` — |z|>3 sensitivity (Phase 08)
- `profile_stability.csv` — cross-condition stability summary (Phase 08)

#### numerical_audit/
- `audit_checks.csv` — full audit of all checks (Phase 09)
- `audit_discrepancies.csv` — any failed checks (Phase 09)

### Key Relationships
- Construct scores in `measurement_results/construct_scores.csv` use respondent ID as index (consistent ordering).
- LPA indicators are standardized versions of INT and BE (see `gap_results/standardized_scores.csv`).
- Profile assignments in `lpa_model_comparison/K_7/posterior_probable.csv` use `assigned_class` column.
