# Phase 12 — Paper 1 Evidence Package

## Purpose
Extract verified numerical results from the completed analysis into a
clean machine-readable evidence package for Paper 1.

## Source Files Used
- results/10_final/data_diagnostics/
- results/10_final/measurement_results/
- results/10_final/gap_results/
- results/10_final/lpa_model_comparison/
- results/10_final/selected_lpa/
- results/10_final/profile_parameters/
- results/10_final/profile_comparisons/
- results/10_final/profile_predictors/
- results/10_final/robustness_results/
- results/10_final/numerical_audit/
- results/11_final_verification/

## Output Files
- dataset_summary.csv — N, variables, missing, duplicates
- measurement_summary.csv — per-construct alpha, mean, SD
- int_behavior_correlation.csv — INT-BE Pearson r, p, N, CI
- gap_summary.csv — GAP statistics
- lpa_model_comparison.csv — BIC, AIC, entropy, sizes for each estimated K
- all_profile_structure.csv — profile structure for each estimated K
- k6_profile_table.csv — selected-K primary profile table
  (K read from results/05_lpa_selection/selected_model.csv)
- k6_predictors.csv — multinomial logit (coefficients, OR, CI, FDR)
- robustness_summary.csv — all robustness checks
- k6_reproducibility.csv — selected-K reproduction verification
- factor_score_robustness.csv — factor vs mean correlations
- paper1_master_evidence.csv — master evidence table
