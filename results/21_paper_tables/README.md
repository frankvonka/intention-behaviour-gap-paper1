# Phase 21 — Final Paper Table Data

## Purpose
Clean numerical tables for Paper 1. Every number is copied from
existing frozen result files. No new analysis, no new selection,
no plots, no prose.

## Tables Created
- 01_sample_table.csv
- 02_measurement_table.csv
- 03_correlation_table.csv + 03_correlation_summary.csv
- 04_lpa_comparison_table.csv
- 05_k6_profile_table.csv
- 06_profile_gap_table.csv + 06_profile_gap_supplement.csv
- 07_classification_table.csv
- 08_stability_table.csv
- 09_predictor_table.csv + 09_predictor_per_predictor.csv
- 10_robustness_table.csv + 10_robustness_supplement.csv
- 11_master_paper_tables.csv (index)
- README.md

## Frozen values preserved
- N = 1166
- K = 7 (minimum BIC among non-degenerate fits, per
  results/05_lpa_selection/selected_model.csv)
- K=7 profile sizes = [281, 125, 108, 264, 245, 70, 73]
- INT-BE r = 0.6515
- K=7 BIC = {float(fit.loc[fit['K'] == K, 'BIC'].iloc[0]):.4f}
- K=7 AIC = {float(fit.loc[fit['K'] == K, 'AIC'].iloc[0]):.4f}
- K=7 normalized classification entropy E = -sum(p log p)/(n log K)
  (in [0,1]; higher = better separated)

## No Modifications
All values copied from existing result files. No rounding changes
that affect interpretation. No silent removals.
