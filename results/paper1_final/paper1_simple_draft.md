# Paper 1 — Simple First Draft

**Title (provisional):** Heterogeneous Intention-Behaviour Configurations in Household Energy-Saving Behaviour: A Latent Profile Analysis of N = 1166 Respondents

**Draft status:** FIRST DRAFT. Numerical content is taken verbatim from the frozen computational results in `results/paper1_final/`. No new analyses were performed and no literature was searched. All unsupported claims from Phase 22 are excluded.

---

## 1. Abstract

**Background.** [LITERATURE SUPPORT NEEDED] A core assumption in the theory of planned behaviour is that intentions are reliable proximal determinants of behaviour, but the empirical gap between stated intention and observed behaviour is well documented. [LITERATURE SUPPORT NEEDED]

**Objective.** This paper investigates whether household energy-saving behaviour shows heterogeneous intention-behaviour configurations in a sample of N = 1166 respondents, using a latent profile analysis (LPA) framework.

**Methods.** A psychometric battery measuring ten constructs (attitude, subjective norm, perceived behavioural control, COVID-related context, intention, behaviour, perceived usefulness, perceived ease of use, personal obligation, and personal responsibility) was administered. Internal consistency reliability, descriptive statistics, and the Pearson correlation between intention and behaviour were computed. The intention-behaviour gap (GAP) was defined as z(INT) - z(BE). Latent profile analysis on the standardized (INT, BE) indicators was estimated for K = 2..6 with full covariance, n_init = 1000, and a fixed seed of 42. Model selection used the Bayesian Information Criterion (BIC). The selected K = 6 solution was characterised, validated, and tested for stability, alternative-specification sensitivity, and association with eight psychological and five demographic predictors via multinomial logistic regression.

**Results.** The intention-behaviour correlation was r = 0.6515 (95% CI [0.6171, 0.6833], p < 0.001). K = 6 was the minimum-BIC solution across K = 2..6 (BIC = 2129.17, AIC = 1952.03). Profile sizes were [124, 377, 262, 90, 54, 259]. Three profiles showed mean INT > mean BE and three showed mean BE > mean INT. Classification quality was high (mean max posterior = 0.89, 88.85% of respondents had max posterior ≥ 0.80). Bootstrap stability (200/200 converged) and split-sample validation supported the profile structure. Predictor analysis (13 predictors × 5 contrasts = 65 tests, joint Benjamini-Hochberg FDR) identified 21 FDR-significant associations, with construct predictors exhibiting high multicollinearity (max VIF = 67.44).

**Conclusion.** The data support the existence of heterogeneous intention-behaviour configurations. Causal interpretation is precluded by the cross-sectional design.

---

## 2. Introduction

[INTERPRETATION REQUIRES REVIEW] [LITERATURE SUPPORT NEEDED]

Household energy conservation is a behavioural domain in which the gap between stated intention and reported behaviour is frequently examined. The theory of planned behaviour [LITERATURE SUPPORT NEEDED] posits that intention is the immediate antecedent of behaviour, mediated by perceived behavioural control. The empirical observation that intention and behaviour are not perfectly aligned has generated extensive work on intention-behaviour gaps. [LITERATURE SUPPORT NEEDED]

This paper examines whether the apparent intention-behaviour gap in household energy saving reflects a single, uniform discrepancy, or whether distinct subgroups with different intention-behaviour configurations can be identified empirically.

---

## 3. Research Gap / Motivation

[INTERPRETATION REQUIRES REVIEW] The conventional analytic approach is to estimate a single intention-behaviour regression coefficient, summarising the population with one point estimate. This approach cannot reveal whether subgroups with qualitatively different intention-behaviour relationships exist. [LITERATURE SUPPORT NEEDED] motivates person-centred approaches such as latent profile analysis for behavioural research.

---

## 4. Research Objective

To determine, in a cross-sectional sample of N = 1166, whether distinct latent profiles of intention and behaviour exist, to characterise each profile, and to identify which observed psychological and demographic variables are associated with profile membership.

---

## 5. Data and Methods

### 5.1 Sample
- N = 1166 respondents
- 34 measured variables
- 0 missing values across all measured cells
- 42 duplicate rows reported in the raw data file; these were flagged and reported but not removed (see `01_data/duplicate_information.csv`)
- Psychometric items used a 1-5 Likert scale

### 5.2 Measurement
Ten constructs were scored as arithmetic means of their constituent items:
ATT (3 items), CON (3 items), SNO (3 items), COVID (3 items), INT (3 items), BE (4 items), PU (3 items), PEU (2 items), PO (2 items), PRI (3 items). Full descriptive statistics, including Cronbach's alpha, are in `02_measurement/02_measurement_table.csv`.

### 5.3 Intention-Behaviour Gap
GAP was defined as a respondent-level z-score difference: GAP_i = z(INT_i) - z(BE_i), where z = (x - mean) / SD with ddof = 1.

### 5.4 Latent Profile Analysis
The primary LPA used:
- Indicators: z(INT), z(BE)
- K = 2, 3, 4, 5, 6
- Estimator: sklearn.mixture.GaussianMixture
- Covariance: full
- n_init = 1000
- random_state = 42
- max_iter = 500
- reg_covar = 1e-6

K was selected by minimum BIC. [METHOD DETAIL NEEDS CONFIRMATION] for the theoretical rationale.

### 5.5 Predictor Analysis
Profile membership (K = 6, Profile 0 = reference) was regressed on 13 predictors: the eight psychological constructs (ATT, CON, SNO, COVID, PU, PEU, PO, PRI) and five demographic variables (age, gender, Education, Occupation, income) using multinomial logistic regression (statsmodels MNLogit, BFGS, maxiter = 1000). No predictor was removed. P-values were adjusted jointly across all 65 tests by Benjamini-Hochberg FDR at α = 0.05.

---

## 6. Measurement Results

Cronbach's alpha for the ten constructs spanned 0.67 to 0.92. INT and BE showed a strong positive Pearson correlation: r = 0.6515 (95% CI [0.6171, 0.6833], p ≈ 8.8 × 10⁻¹⁴², N = 1166). [INTERPRETATION REQUIRES REVIEW] of the substantive magnitude. (Source: `02_measurement/02_measurement_table.csv`, `02_measurement/int_be_correlation.csv`.)

---

## 7. Intention-Behaviour Relationship

At the respondent level, 524 respondents (44.94%) had INT > BE and 642 (55.06%) had BE > INT. The GAP distribution had mean ≈ 0 and SD ≈ 0.83. (Source: `03_intention_behavior_gap/04_gap_evidence.csv`.)

A separate "gap-only" 1D Gaussian mixture (K = 1..6) was estimated to assess whether a single 1D clustering of the gap is supported. The minimum-BIC K in the gap-only 1D model was 6 (BIC = 2598.31). The LEVEL dimension, defined as (z(INT) + z(BE)) / 2, was approximately orthogonal to GAP (r ≈ 0). (Source: `03_intention_behavior_gap/06_profile_gap_table.csv`, `06_profile_gap_supplement.csv`.)

---

## 8. Latent Profile Analysis

### 8.1 Model Comparison
K = 6 produced the minimum BIC across K = 2..6 under the primary specification.

| K | LL | AIC | BIC | entropy |
|---|----|-----|-----|---------|
| 2 | [METHOD DETAIL NEEDS CONFIRMATION] | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | -941.01 | 1952.03 | 2129.17 | 1.14 |

Source: `04_lpa/04_lpa_comparison_table.csv`.

### 8.2 Selected Solution
K = 6 was selected by minimum BIC. The K = 6 BIC value was reproduced exactly in the Phase 19 final audit (2129.1734).

---

## 9. K = 6 Profile Results

Profile sizes: [124, 377, 262, 90, 54, 259]; sum = 1166.

Directional summary (using z-score means):
- Profiles with mean INT > mean BE: 3 (P0, P2, P5)
- Profiles with mean BE > mean INT: 3 (P1, P3, P4)
- Approximately balanced (|INT-BE| ≤ 0.12): 1 profile (P5 with diff ≈ 0.001) — note that P5's mean INT > mean BE by a small margin; the P3 profile is also near balance (diff ≈ -0.084).

Per-profile descriptives are in `05_profiles/05_k6_profile_table.csv`.

---

## 10. Classification Quality

Mean max posterior = 0.89; median max posterior = 0.94. 88.85% of respondents had max posterior ≥ 0.80; 0% had max posterior < 0.50. (Source: `05_profiles/07_classification_table.csv`.)

---

## 11. Profile Stability

200/200 bootstrap replications of K = 6 converged. Direction-consistency proportions across bootstraps varied across profiles (range: 5% to 100%); profiles are NOT equally stable. [INTERPRETATION REQUIRES REVIEW]. (Source: `05_profiles/08_stability_table.csv`, `05_profiles/k6_stability_master.csv`.)

---

## 12. Profile Predictors

The multinomial logistic regression (Profile 0 reference, 13 predictors × 5 contrasts = 65 tests) converged:
- LL = -1309.94
- AIC = 2759.88
- BIC = 3114.18
- McFadden pseudo-R² = 0.3035
- 21 / 65 contrasts were FDR-significant (joint Benjamini-Hochberg, α = 0.05)

Per-predictor summary in `06_predictors/09_predictor_per_predictor.csv`. Predictors with the largest number of FDR-significant contrasts include PU and PRI. **Caveat:** the cross-sectional design precludes causal interpretation. **Caveat:** construct predictors exhibited substantial multicollinearity (max VIF = 67.44, all 8 construct predictors VIF > 26), so individual coefficients are not uniquely identified; this was confirmed numerically in the Phase 18B audit.

---

## 13. Robustness and Validation

Alternative model specifications were tested (Phase 17):
- Full covariance on 10 indicators: BIC = 18768.87
- Diagonal covariance: BIC = 20087.56
- Spherical covariance: BIC = 26200.79
- FactorAnalysis score representation (vs mean scores): BIC = 3184.31
- Random seeds 1..5: BIC range 2128.69 - 2149.03; total profile matching distances 0.21 - 0.40 (seeds), 2.03 - 2.38 (alternative specifications)

Profile structure is NOT perfectly stable across all alternative specifications (matching distances ≥ 2.0 in non-seed alternative specs).

Source: `07_robustness/10_robustness_table.csv`.

---

## 14. Discussion

[INTERPRETATION REQUIRES REVIEW] The empirical data support the existence of heterogeneous intention-behaviour configurations in household energy saving: three of the six profiles showed mean INT > mean BE and three showed mean BE > mean INT. The strongest within-profile directional pattern is not uniform across profiles.

The K = 6 solution has the minimum BIC under the primary specification, but this does NOT establish that K = 6 is the objectively correct number of profiles. Stability diagnostics (bootstrap, split-sample, alternative specifications) support the broad structure but show measurable differences across alternative specifications and across profiles.

The strong positive intention-behaviour correlation (r = 0.65) coexists with substantial profile-level heterogeneity, indicating that the population-level correlation is not a sufficient description of the underlying data-generating process.

---

## 15. Theoretical Contribution

[INTERPRETATION REQUIRES REVIEW] [LITERATURE SUPPORT NEEDED] The coexistence of multiple directional profiles within a single sample argues against modelling the intention-behaviour relationship as a single population process. The data are consistent with a person-centred view in which intention and behaviour jointly take qualitatively different configurations across subgroups.

---

## 16. Practical Implications

[INTERPRETATION REQUIRES REVIEW] Policy or intervention design that targets a single intention-behaviour relationship may not be uniformly effective across the empirically identified profile configurations. [LITERATURE SUPPORT NEEDED].

---

## 17. Limitations

1. **Cross-sectional design** — causal claims about predictors of profile membership are NOT supported by the data (Phase 22 claim C11: NOT_SUPPORTED). Temporal causality is NOT established (C14: NOT_SUPPORTED).
2. **Multicollinearity** — construct predictors have VIF up to 67.44; individual predictor coefficients are not uniquely identified (C12: NOT_SUPPORTED, C21: NOT_SUPPORTED).
3. **K-selection** — K = 6 is the minimum-BIC solution under the primary specification, not a uniquely correct number of profiles.
4. **Stability heterogeneity** — profiles are NOT equally stable across bootstraps (C7: NOT_SUPPORTED).
5. **Alternative specifications** — K = 6 is NOT perfectly stable across all alternative specifications (C10: NOT_SUPPORTED).
6. **Duplicate rows** — 42 duplicate rows in the raw data were reported but not removed.
7. **Predictor effect sizes** — 21/65 FDR-significant tests; substantial non-significance should be acknowledged alongside positive findings.

---

## 18. Conclusion

The data from N = 1166 respondents support the existence of heterogeneous intention-behaviour configurations in household energy saving, with K = 6 as the minimum-BIC solution under the primary specification. Three profiles showed mean INT > mean BE and three showed mean BE > mean INT. Predictor analyses identified 21/65 FDR-significant associations but were constrained by cross-sectional design and substantial multicollinearity. Causal and theoretical interpretations are reserved for further review.

---

## Source File Index (for the writer)

- Sample: `01_data/01_sample_table.csv`
- Measurement: `02_measurement/02_measurement_table.csv`
- INT-BE correlation: `02_measurement/int_be_correlation.csv`
- GAP descriptive: `03_intention_behavior_gap/04_gap_evidence.csv`
- GAP variance decomposition: `03_intention_behavior_gap/06_profile_gap_table.csv`
- LPA comparison: `04_lpa/04_lpa_comparison_table.csv`
- K = 6 profiles: `05_profiles/05_k6_profile_table.csv`
- Classification: `05_profiles/07_classification_table.csv`
- Stability: `05_profiles/08_stability_table.csv`
- Predictors: `06_predictors/09_predictor_table.csv`, `06_predictors/09_predictor_per_predictor.csv`
- Robustness: `07_robustness/10_robustness_table.csv`
- Audit / claims: `08_audit/05_final_claim_audit.csv`, `08_audit/01_supported_claims.csv`, `08_audit/02_caveated_claims.csv`, `08_audit/03_unsupported_claims.csv`
- Master evidence: `09_master/paper1_master_evidence.csv`
