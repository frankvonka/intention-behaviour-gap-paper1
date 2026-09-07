# Phase 13 — K=7 Profile Validation

## Purpose
Produce numerical evidence for later evaluation of the K=7 profile solution
(K selected in Phase 05). No refitting. No literature. No interpretation. No plots.

## Input Files
- results/02_measurement/construct_scores.csv
- results/04_lpa_estimation/K_7/posterior_probabilities.csv
- results/03_gap_analysis/gap_scores.npy

## Calculations
1. **k6_profile_descriptives.csv** — per-profile N, %, mean/SD/median/IQR for all 10 constructs + INT, BE, INT-BE, GAP
2. **pairwise_profile_separation.csv** — Cohen's d for all profile pairs on INT, BE, GAP
3. **k6_int_be_configuration.csv** — signed INT-BE differences per profile
4. **k6_classification_quality.csv** — distribution of max posterior probabilities
5. **k6_profile_uncertainty.csv** — uncertainty statistics per profile
6. **k6_profile_sizes.csv** — simple N and % table
7. **k6_validation_master.csv** — master summary

## Methods
- Construct scores: arithmetic mean of items (from Phase 02)
- GAP: z_INT - z_BE (from Phase 03)
- Cohen's d: (M1-M2) / pooled SD with pooled SD = sqrt[((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2)]
- IQR: Q75 - Q25

## No Data Modifications
All calculations read existing files. No new fitting. No dataset changes.
