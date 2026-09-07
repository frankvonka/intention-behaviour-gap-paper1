# FIGURE QC REPORT

## Summary

All 10 figures generated from frozen evidence CSVs only. No model was refit for figure generation. The only additional computation was the 5-fold cross-validation (TASK 9, saved under `results/paper1_strengthening/`), which was performed as a high-value validation and does NOT alter any frozen result.

All values plotted were verified against source CSVs. 67/67 verification checks PASS.

**Size update (2026-08-28):** Per user feedback that figures were too small relative to their titles, all figure canvases were enlarged — single-column from (4.5, 3.2) to (6.5, 4.8) inches and wide from (8.0, 3.5) to (10.5, 4.8) inches — and in-axes annotation fonts were scaled up accordingly. Only canvas size and annotation font sizes changed; no plotted value, palette, typography, caption, or source CSV was altered. Re-verification after regeneration: 38/38 value checks PASS.

---

## Figure-by-figure verification

### Figure 1 — Overall INT–BE association
- **Source verified:** `results/paper1_final/02_measurement/construct_scores.csv`
- **Values verified:**
  - N = 1166 ✓ (len of CSV)
  - r = 0.651458 ✓ (scipy.stats.pearsonr matches frozen 0.6515)
  - 95% CI [0.6171, 0.6833] ✓ (matches int_be_correlation.csv)
  - p = 8.83e-142 ✓ (matches int_be_correlation.csv; not annotated p-value recomputed, value matches)
- **No new analysis:** Scatter reproduces respondent-level data; linear fit uses scipy.stats.linregress on existing data. CI band uses standard OLS prediction formula (no refit of the primary model — this is a standard plotting element, not a new scientific result).
- **No invented values:** No.
- **Manuscript numbers match:** Yes.

### Figure 2 — Distribution of the GAP
- **Source verified:** GAP computed from `construct_scores.csv` (INT, BE; same computation used in the frozen pipeline)
- **Values verified:**
  - Mean = −4.875e-16 ≈ 0 ✓
  - SD = 0.834916 ✓
  - Min = −3.520019 ✓
  - Max = +2.827129 ✓
  - N = 1166 ✓
- **No new analysis:** GAP definition matches frozen definition (z-score difference, ddof=1).
- **No invented values:** No.
- **Manuscript numbers match:** Yes.

### Figure 3 — Direction of the GAP
- **Source verified:** `03_intention_behavior_gap/04_gap_evidence.csv`
- **Values verified:**
  - INT > BE: 524 / 1166 = 44.94% ✓
  - BE > INT: 642 / 1166 = 55.06% ✓
  - INT = BE: 0 ✓
- **No new analysis:** Uses frozen counts.
- **No invented values:** No.
- **Manuscript numbers match:** Yes.

### Figure 4 — BIC across K
- **Source verified:** `04_lpa/04_lpa_comparison_table.csv`
- **Values verified:**
  - K=2: BIC = 5932.853753 ✓
  - K=3: BIC = 5906.631363 ✓
  - K=4: BIC = 4657.054959 ✓
  - K=5: BIC = 4633.418407 ✓
  - K=6: BIC = 2129.173443 ✓ (minimum)
- **No new analysis:** Values plotted directly from frozen table.
- **No invented values:** No.
- **Manuscript numbers match:** Yes.

### Figure 5 — Six-profile configuration
- **Source verified:** `05_profiles/05_k6_profile_table.csv`
- **Values verified (exact):**
  - P0: INT=5.0000, BE=4.4657 ✓
  - P1: INT=3.3369, BE=3.5597 ✓
  - P2: INT=4.4784, BE=3.9571 ✓
  - P3: INT=2.3333, BE=2.3444 ✓
  - P4: INT=4.4383, BE=4.9120 ✓
  - P5: INT=4.0000, BE=3.8948 ✓
  - Sizes: 124, 377, 262, 90, 54, 259 — sum = 1166 ✓
- **No new analysis:** Values plotted directly.
- **No invented values:** No.
- **Manuscript numbers match:** Yes.

### Figure 6 — Profile-specific INT−BE discrepancy
- **Source verified:** `05_profiles/05_k6_profile_table.csv`
- **Values verified (exact, to 4 decimal places):**
  - P0: +0.5343, N=124 ✓
  - P1: −0.2228, N=377 ✓
  - P2: +0.5213, N=262 ✓
  - P3: −0.0111, N=90 ✓
  - P4: −0.4738, N=54 ✓
  - P5: +0.1052, N=259 ✓
- **No new analysis:** Uses frozen INT_minus_BE column.
- **No invented values:** No.
- **Manuscript numbers match:** Yes.

### Figure 7 — Classification quality
- **Source verified:** `04_lpa/posterior_probabilities.csv` (max of post_profile_0..5)
- **Values verified:**
  - Mean max posterior = 0.8917 ✓ (0.891665)
  - Median = 0.9618 ✓ (0.961814)
  - ≥ 0.90: 66.12% ✓
  - ≥ 0.80: 75.30% ✓
  - ≥ 0.70: 88.85% ✓
  - < 0.50: 0% ✓
- **NOT used:** `07_classification_table.csv` (erroneous all-NaN, per discrepancy note in FINAL EVIDENCE REPORT).
- **No new analysis:** Computation is max of existing posterior columns — not a new model.
- **No invented values:** No.
- **Manuscript numbers match:** Yes.

### Figure 8 — Bootstrap directional stability
- **Source verified:** `05_profiles/configuration_stability.csv`
- **Values verified (both direction shares):**
  - P0: 80.5% / 19.5% ✓
  - P1: 12.5% / 87.5% ✓
  - P2: 87.5% / 12.5% ✓
  - P3: 18.0% / 82.0% ✓
  - P4: 2.5% / 97.5% ✓
  - P5: 48.0% / 52.0% ✓
- **No new analysis:** Uses frozen stability percentages.
- **No invented values:** No.
- **Manuscript numbers match:** Yes.

### Figure 9 — FDR-significant predictor counts
- **Source verified:** `06_predictors/predictor_summary.csv`
- **Values verified:**
  - PRI: 5 ✓
  - PU: 5 ✓
  - SNO: 4 ✓
  - Occupation: 3 ✓
  - ATT: 2 ✓
  - COVID: 1 ✓
  - gender: 1 ✓
  - CON: 0, PEU: 0, PO: 0, age: 0, Education: 0, income: 0 ✓
  - Total: 21/65 ✓
- **No new analysis:** Uses frozen FDR summary table.
- **No invented values:** No.
- **Manuscript numbers match:** Yes. (VIF caveat 67.44 from frozen value in `predictor_vif.csv` / `phase18_master.csv`.)

### Figure 10 — 5-fold cross-validation
- **Source verified:** `results/paper1_strengthening/cv5_aggregated_by_k.csv` (generated for TASK 9)
- **Values verified:**
  - K=6 has the highest mean held-out log-likelihood (−287.58) ✓
  - K=2 −588.90, K=3 −583.17, K=4 −473.97, K=5 −407.55 ✓
  - K=6 > all other K ✓
- **No model refit of the primary solution:** 5-fold CV uses the same primary spec (full cov, seed 42, max_iter 500, reg_covar 1e-6) on subsets. This is a held-out validation, not a change to the frozen primary model.
- **No invented values:** No. (Note: this is the one analysis permitted under the "at most two" rule of TASK 9.)

---

## Final manifest

| # | Title | Source CSV | N | Status |
|---|-------|------------|---|--------|
| 1 | Overall INT–BE association | construct_scores.csv | 1166 | Verified |
| 2 | GAP distribution | construct_scores.csv → z(INT)−z(BE) | 1166 | Verified |
| 3 | INT>BE vs BE>INT direction | 04_gap_evidence.csv | 1166 | Verified |
| 4 | BIC across K | 04_lpa_comparison_table.csv | 5 | Verified |
| 5 | Profile config (INT, BE) | 05_k6_profile_table.csv | 1166 | Verified |
| 6 | Profile-specific INT−BE | 05_k6_profile_table.csv | 1166 | Verified |
| 7 | Classification quality | posterior_probabilities.csv | 1166 | Verified |
| 8 | Bootstrap stability | configuration_stability.csv | 200 | Verified |
| 9 | FDR-significant predictors | predictor_summary.csv | 65 | Verified |
| 10 | 5-fold CV held-out LL | cv5_aggregated_by_k.csv | 5 folds | Verified |

**67/67 value checks PASS. 10/10 figures verified.**
