# Paper 1 — Final Evidence Collection

## Purpose
Organizes the frozen Phase 1-22 results for Paper 1 writing.
No new analysis. All files are copies of existing frozen outputs
(unchanged).

## Folder Structure

| Folder | Purpose |
|--------|---------|
| 01_data | Sample size, variables, missing, duplicates, item ranges |
| 02_measurement | 10 constructs, Cronbach alpha, means, SDs, correlation matrix |
| 03_intention_behavior_gap | INT/BE means, INT-BE correlation, GAP definition, gap validation |
| 04_lpa | K=2-6 model comparison, K=6 selected solution, reproducibility |
| 05_profiles | K=6 profiles, classification quality, stability, structure across K |
| 06_predictors | MNLogit results, FDR, VIF, leave-one-out, coefficient stability |
| 07_robustness | Covariance/score/seed sensitivity, bootstrap, split-sample |
| 08_audit | Phase 19 numerical audit, Phase 22 claim audit, manifests |
| 09_master | Master evidence index, master paper tables |

## Key Files for Writing

- **01_data/01_data/01_sample_table.csv** — N, variables, missing, duplicates
- **02_measurement/02_measurement_table.csv** — alpha, means, SDs per construct
- **03_intention_behavior_gap/06_profile_gap_table.csv** — gap variance decomposition
- **04_lpa/04_lpa_comparison_table.csv** — K=2-6 BIC/AIC/entropy (K=6 min BIC)
- **05_profiles/05_k6_profile_table.csv** — profile N, INT/BE means, directions
- **05_profiles/07_classification_table.csv** — posterior quality
- **05_profiles/08_stability_table.csv** — bootstrap, split-sample
- **06_predictors/09_predictor_table.csv** — MNLogit summary, N, R2, AIC, BIC
- **06_predictors/09_predictor_per_predictor.csv** — per-predictor FDR summary
- **07_robustness/10_robustness_table.csv** — alternative model sensitivity
- **08_audit/05_final_claim_audit.csv** — supported/unsupported claims
- **09_master/paper1_master_evidence.csv** — complete evidence index (93 entries)

## Important Notes

- **SUPPORTED claims**: 12 (use freely)
- **SUPPORTED_WITH_CAVEAT**: 2 (preserve caveat)
- **NOT_SUPPORTED**: 8 (do NOT use as findings)

See `08_audit/03_unsupported_claims.csv` and `08_audit/02_caveated_claims.csv`.

## No New Analysis
All values copied verbatim from Phase 1-19 result directories.
Files in this folder are organized copies — originals untouched.
