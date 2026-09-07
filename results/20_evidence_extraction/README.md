# Phase 20 — Final Evidence Extraction

## Purpose
Extract already-frozen numerical evidence from Phases 1-19 into
clean machine-readable tables. No new analysis, no model refit,
no interpretation.

## Frozen values preserved (examples, K=7)
- N = 1166
- INT-BE Pearson r = 0.6515
- K = 7 (minimum BIC among non-degenerate fits, per results/05_lpa_selection/selected_model.csv)
- K=7 BIC = {float(fit.loc[fit['K'] == K, 'BIC'].iloc[0]):.4f} — from results/04_lpa_estimation/model_fit.csv
- K=7 AIC = {float(fit.loc[fit['K'] == K, 'AIC'].iloc[0]):.4f} — from results/04_lpa_estimation/model_fit.csv
- K=7 profile sizes = {[r['N'] for r in k6_rows]}

## Output Files
- 01_dataset_evidence.csv
- 02_measurement_evidence.csv
- 03_int_behavior_evidence.csv
- 04_gap_evidence.csv
- 05_lpa_model_evidence.csv
- 06_k6_profile_evidence.csv
- 07_profile_gap_evidence.csv
- 08_classification_evidence.csv
- 09_stability_evidence.csv
- 10_predictor_evidence.csv
- 11_robustness_evidence.csv
- 12_final_evidence_matrix.csv
- README.md

## No Modifications
All values copied from existing frozen result files. No new
model fit. No new analysis. No literature search. No plots.
