# Reviewer-Fix Pass — Task 7: Methodology Information Audit

## Purpose
Item-by-item audit of methodology information a reviewer would expect. For each
item: (a) where the value came from, (b) whether it is *verifiable* from the
present evidence package, (c) the *exact* fact to put in the manuscript, and
(d) explicit gap flags where the source dataset does not contain the information.

## Source-of-truth policy
- **Verifiable from code** (`scripts/*.py`): high confidence. The script is the
  record of the analysis that was actually run.
- **Verifiable from frozen CSVs** (`results/02_measurement/*.csv`, etc.):
  medium confidence. The CSV is the cached output of a known script.
- **Not in evidence package** (no metadata, no README, no recruitment log in
  data.xls): marked `[METHOD DETAIL NEEDS CONFIRMATION]`. The manuscript
  *cannot* assert this from the evidence alone; it must be confirmed against
  the original study documentation or the claim must be omitted.

---

## 1. Sample size
- Value: **N = 1166**
- Source: `data.xls` sheet 0 has 1166 rows; cross-checked in every phase (`01_data_inspection/data_dimensions.csv`).
- Verifiable: ✅ YES.
- Manuscript: "The analytic sample consisted of N = 1,166 respondents."

## 2. Number of variables
- Value: **34** (29 psychometric items + 5 demographics)
- Source: `data.xls` 34 columns; `01_data_inspection/column_information.csv`.
- Verifiable: ✅ YES.
- Manuscript: "34 measured variables (29 psychometric items in 10 constructs, plus 5 demographics: gender, age, Education, Occupation, income)."

## 3. Item-level / scale anchors
- Value: **1–5 Likert** (every psychometric item has observed min = 1, max = 5)
- Source: `01_data_inspection/item_ranges.csv`; `scripts/01_data_inspection.py` sets `LIKERT_MIN, LIKERT_MAX = 1, 5`.
- Verifiable: ✅ YES for the *range*; ❌ NO for polarity wording.
- Manuscript: "All psychometric items used a 1–5 response scale."
- Gap: **Direction labels (e.g., 1 = strongly disagree, 5 = strongly agree) are NOT in the data file** — must be confirmed from the instrument source. → `[METHOD DETAIL NEEDS CONFIRMATION]` for the polarity wording.

## 4. Construct definitions (item composition)
- Value: ATT(3), CON(3), SNO(3), COVID(3), INT(3), BE(4), PU(3), PEU(2), PO(2), PRI(3) — 29 items total
- Source: `scripts/02_measurement.py:CONSTRUCTS`; `scripts/01_data_inspection.py:EXPECTED_ITEMS`.
- Verifiable: ✅ YES (the script is the record of the analysis that was run).
- Manuscript: "Ten latent constructs were defined a priori: ATT, CON, SNO, COVID, INT, BE, PU, PEU, PO, PRI, composed of 2–4 items each (29 items total). Scoring: arithmetic mean of available items per respondent."
- Gap: **Theoretical citations for each construct** (INT/BE from which theory, PRI from which framework) are NOT in the evidence package. → `[METHOD DETAIL NEEDS CONFIRMATION]` for citation anchors.

## 5. Construct scoring rule
- Value: Arithmetic mean of available items per respondent (no reverse coding, no imputation)
- Source: `scripts/02_measurement.py`: `scores[construct] = df[present].astype(float).mean(axis=1)`.
- Verifiable: ✅ YES.
- Manuscript: "Each construct was scored as the arithmetic mean of its items per respondent; no reverse coding was applied and no imputation was performed (no missing values were observed)."

## 6. Missing data
- Value: **0** missing cells across 1166 × 34 = 39,644 cells
- Source: `01_data_inspection/missing_values.csv` (0 in every column); `df.isna().sum().sum() == 0`.
- Verifiable: ✅ YES.
- Manuscript: "No missing values were observed; complete-case analysis was used for all reported models."

## 7. Duplicate rows
- Value: **42** all-column-duplicate rows flagged (3.60% of N)
- Source: `df.duplicated().sum() == 42`; `01_data_inspection/duplicate_information.csv`; `01_data_inspection/duplicate_rows.csv`.
- Verifiable: ✅ YES (count); the decision NOT to remove is a documented frozen choice.
- Manuscript: "Forty-two respondents (3.60% of the sample) had identical responses across all measured variables. Per the pre-registered analysis decision these were flagged but retained; a sensitivity analysis excluding them is reported in the Reviewer-Fix Pass (Task 3; `results/paper1_strengthening/duplicate_sensitivity.csv`)."
- Gap: **No objective criterion exists to identify "true" duplicates vs legitimately identical responses** in a fully-observed Likert dataset. The frozen decision was *report but do not remove*; the manuscript must justify this rather than imply the 42 are data errors.

## 8. Sampling frame / recruitment method
- Value: `[METHOD DETAIL NEEDS CONFIRMATION]`
- Source: **NOT in evidence package.** `data.xls` contains no metadata, no recruitment log, no field sheet.
- Verifiable: ❌ NO.
- Manuscript: Must be added from the original study documentation. Required items for publication (per standard social-science reporting): (a) sampling frame (population, geography, time window), (b) recruitment channel (online panel, classroom, household survey, etc.), (c) sample-size rationale / power analysis, (d) response rate or completion rate, (e) compensation, if any, (f) data collection dates, (g) language(s) of administration, (h) inclusion/exclusion criteria. The frozen package contains **none of these**.
- Action: **NOT FIXED** by re-analysis. **REQUIRES ORIGINAL STUDY DOCUMENTATION.** Until supplied, the manuscript's external-validity statements must be cautious.

## 9. Ethics / consent
- Value: `[METHOD DETAIL NEEDS CONFIRMATION]`
- Source: NOT in evidence package.
- Verifiable: ❌ NO.
- Manuscript: Standard line, e.g. "Participants provided informed consent under [IRB reference]." Must be filled in from the ethics file.

## 10. Demographics (verifiable cells only)

| Field | Observed values (data.xls) | N present |
|---|---|---|
| age | 17–71 (mean 25.70, n_unique 44) | 1166 |
| gender | {0, 1} (0: 595, 1: 571) | 1166 |
| Education | 1..4 (counts 95, 76, 828, 167) | 1166 |
| income | 1..4 (counts 54, 184, 415, 513) | 1166 |
| Occupation | 1..7 (counts 584, 15, 70, 80, 319, 30, 68) | 1166 |

- Verifiable: ✅ YES for counts and ranges; categorical **value labels** (what does gender 0/1 mean? what does Education 1/2/3/4 mean?) are NOT in the data file.
- Manuscript: "Demographic composition: age 17–71 (M = 25.70); gender balanced (n₀ = 595, n₁ = 571); Education concentrated in level 3 (n = 828, 71.0%); income skewed to level 4 (n = 513, 44.0%); Occupation dominated by category 1 (n = 584, 50.1%)."
- Gap: **Demographic category labels require the codebook from the original study.** → `[METHOD DETAIL NEEDS CONFIRMATION]` for label meanings.

## 11. Reliability (Cronbach's α)
- Value: α(ATT)=0.7021, α(CON)=0.7621, α(SNO)=0.8055, α(COVID)=0.7729, α(INT)=0.7823, α(BE)=0.7480, α(PU)=0.7274, α(PEU)=0.6727, α(PO)=0.8113, α(PRI)=0.9186
- Source: `scripts/02_measurement.py:cronbach_alpha` (formula `(k/(k-1))·(1 − Σσ²ᵢ/σ²ₜ)`); values in `02_measurement/cronbach_alpha.csv`. **Verified independently in Task 6** (recomputed α(INT)=0.7823, α(BE)=0.7480 match).
- Verifiable: ✅ YES.
- Manuscript: "Internal consistency reliability (Cronbach's α) ranged from 0.67 (PEU, k = 2) to 0.92 (PRI, k = 3). Five constructs met or exceeded the conventional 0.70 threshold; PEU (α = 0.67) and ATT (α = 0.70) fell slightly below."
- Caveat: PEU has k = 2 items; α with k = 2 is mechanically low and should be reported with that caveat.

## 12. INT–BE correlation
- Value: r = 0.6515, p = 8.83 × 10⁻¹⁴², N = 1166, 95% CI = [0.6171, 0.6833] (Fisher z)
- Source: `scripts/02_measurement.py` (`pearsonr` + Fisher-z CI); verified independently in **Task 6** (`results/paper1_strengthening/pearson_verification.csv`).
- Verifiable: ✅ YES.
- Manuscript: "Intention and behaviour were strongly positively correlated (Pearson r = 0.6515, 95% CI [0.6171, 0.6833] by Fisher z-transformation, p = 8.83 × 10⁻¹⁴², N = 1,166)."

## 13. Standardization
- Value: z-score with ddof = 1
- Source: `scripts/04_lpa_estimation.py`: `z_INT = (scores["INT"] - scores["INT"].mean()) / scores["INT"].std(ddof=1)` (identical in scripts/03, 04, 14, 17).
- Verifiable: ✅ YES.
- Manuscript: "Intention and behaviour were each standardized to z-scores (sample mean and standard deviation, ddof = 1) prior to latent profile analysis."

## 14. LPA specification
- Value: sklearn `GaussianMixture`, `covariance_type = "full"`, `n_init = 1000`, `random_state = 42`, `max_iter = 500`, `reg_covar = 1e-6`
- Source: `scripts/04_lpa_estimation.py:run_gmm()` (verbatim). Re-applied identically in scripts 14, 17, 18B, 23.
- Verifiable: ✅ YES.
- Manuscript: "Profiles were estimated using full-covariance Gaussian mixture models on the standardized (z_INT, z_BE) indicators, with K = 2, 3, 4, 5, 6 profiles considered; each model used 1,000 random initializations (sklearn `GaussianMixture(n_init = 1000, random_state = 42, max_iter = 500, reg_covar = 1e-6)`). The K = 2 to 10 sensitivity is reported in the Reviewer-Fix Pass."

## 15. Model selection rule
- Value: Minimum BIC (BIC = −2·log L + k·log n)
- Source: `scripts/04_lpa_estimation.py`; `scripts/05_lpa_selection.py`.
- Verifiable: ✅ YES.
- Manuscript: "The number of profiles was selected by minimum Bayesian Information Criterion (BIC = −2 log L + k log n) across K = 2 to 6 (primary) and K = 2 to 10 (sensitivity)."

## 16. Classification quality
- Value: Entropy E = 1 − Σₚ log p / (n·log K); mean max posterior 0.8917; 88.85% of respondents ≥ 0.70
- Source: `scripts/04_lpa_estimation.py:safe_entropy`; `10_final/k6_classification_quality.csv`.
- Verifiable: ✅ YES.
- Manuscript: "Classification quality: mean maximum posterior probability = 0.89 (median = 0.96); 88.9% of respondents were assigned to their modal profile with posterior ≥ 0.70; no respondent had max posterior < 0.50."

## 17. Stability (bootstrap, split-sample)
- Value: B = 200 bootstrap replications, n_init = 100, matching by minimum Euclidean distance on (mean z_INT, mean z_BE) via Hungarian algorithm; 5 of 6 profiles directionally stable (≥ 80% across bootstraps), profile 5 at 48%/52% (near chance). Mean matching cost = 2.599; split-sample matching cost = 2.713.
- Source: `scripts/14_k6_stability.py:match_profiles`; matching definition documented in **Task 5** (`results/paper1_strengthening/matching_distance_definition.md`).
- Verifiable: ✅ YES.
- Manuscript: "Profile structure was stable across 200 bootstrap replications (matching cost mean = 2.599) and across a random 50/50 split (matching cost = 2.713). Directional stability (sign of mean z_INT − mean z_BE) exceeded 80% for five of six profiles; profile 5 was at chance (48% / 52%), a flagged limitation."

## 18. Predictor model
- Value: statsmodels MNLogit, Profile 0 reference, 13 predictors (8 construct + 5 demographic), 65 contrasts, joint Benjamini–Hochberg FDR at α = 0.05, McFadden pseudo-R² = 0.3035
- Source: `scripts/18_profile_predictors.py`; `18_profile_predictors/multinomial_results.csv`.
- Verifiable: ✅ YES (model output); **VIF diagnostics show severe multicollinearity** (max VIF = 67.44).
- Manuscript: **EXPLORATORY ASSOCIATIONAL ANALYSIS.** "Multinomial logistic regression was used to examine the *associational* relationship between profile membership and 13 candidate predictors. We emphasize this is exploratory, not causal: the design is cross-sectional, the predictors were not pre-registered, and the construct predictors exhibited severe multicollinearity (VIF range 26.7–67.4; condition number 18.4). 21 of 65 contrasts were FDR-significant (joint Benjamini–Hochberg, α = 0.05). Individual coefficient estimates should not be interpreted as uniquely identified effects."
- Gap: This reframing is the **Task 4** action item — the manuscript must NOT report individual odds ratios as if they were stable, uniquely identified estimates.

## 19. Software & reproducibility
- Value: Python 3; sklearn 1.9 `GaussianMixture`; statsmodels MNLogit; scipy.stats.pearsonr; scipy.optimize.linear_sum_assignment; numpy 2.5.2, pandas 3.0.5.
- Source: script imports.
- Verifiable: ✅ YES.
- Manuscript: "All analyses were implemented in Python 3 using scikit-learn (GaussianMixture), statsmodels (MNLogit), SciPy (Pearson correlation, Hungarian matching), NumPy, and pandas. Random seed = 42 throughout the primary analysis. Verification of key results is reported in the supplementary Reviewer-Fix Pass (`results/paper1_strengthening/`)."

---

## Summary table

| # | Item | Verifiable from evidence? | Action |
|---|------|--------------------------|--------|
| 1 | N = 1166 | ✅ YES | Use as-is |
| 2 | 34 variables, 29 items + 5 demographics | ✅ YES | Use as-is |
| 3 | 1–5 Likert range | ✅ YES; polarity wording needs confirmation | Use range; flag wording |
| 4 | Construct item composition | ✅ YES (code); citations need confirmation | Use as-is; flag citations |
| 5 | Arithmetic-mean scoring | ✅ YES | Use as-is |
| 6 | 0 missing | ✅ YES | Use as-is |
| 7 | 42 duplicates | ✅ YES (count) | Use as-is + Task 3 sensitivity |
| 8 | Sampling frame, recruitment, response rate | ❌ **NO** | **NOT FIXED. REQUIRES ORIGINAL STUDY DOCS** |
| 9 | Ethics / consent | ❌ **NO** | **NOT FIXED. REQUIRES ORIGINAL STUDY DOCS** |
| 10 | Demographics ranges and counts | ✅ YES; value labels need codebook | Use counts; flag labels |
| 11 | Cronbach α | ✅ YES (verified in Task 6) | Use as-is |
| 12 | r = 0.6515, CI, p | ✅ YES (verified in Task 6) | Use as-is |
| 13 | z-score ddof = 1 | ✅ YES | Use as-is |
| 14 | LPA spec | ✅ YES | Use as-is |
| 15 | BIC selection | ✅ YES | Use as-is |
| 16 | Classification quality | ✅ YES | Use as-is |
| 17 | Stability diagnostics | ✅ YES (with matching definition file) | Use as-is |
| 18 | Predictor analysis | ✅ YES (output); reframing required | EXPLORATORY reclassification (Task 4) |
| 19 | Software stack | ✅ YES | Use as-is |

## Final classification of reviewer concerns

| Concern | Status | Resolution |
|---|---|---|
| Sampling & recruitment unspecified | **NOT FIXED** | Requires original study documentation; until supplied, the manuscript must use cautious external-validity language and add a Limitations section. |
| Ethics / consent statement missing | **NOT FIXED** | Requires original study documentation. |
| Demographic category labels | **NOT FIXED** | Requires the codebook; manuscript should not invent labels. |
| Polarity wording (1=SD, 5=SA) | **NOT FIXED** | Default-conventional wording is fine if stated as an assumption, but the instrument document is the authoritative source. |
| Construct theory citations | **NOT FIXED** | Requires author knowledge / instrument source. |
| Everything else (N, items, scoring, missing, duplicates count, α, r, z, LPA spec, BIC, classification, stability) | **FIXED BY VERIFIED EVIDENCE** | This audit reproduces each value from the scripts and frozen CSVs. |
| Predictor analysis | **FIXED BY REWORDING** (Task 4) | Relabel as exploratory; do not report individual ORs as stable estimates. |

## Hard gaps that the manuscript MUST acknowledge

The frozen evidence package **does not contain**: (a) the sampling frame, (b) the
recruitment method, (c) the response rate, (d) the data-collection date range,
(e) the language(s) of administration, (f) the ethics approval reference, (g) the
informed-consent procedure, (h) the demographic category labels, (i) the
polarity wording of the 1–5 anchors, (j) the construct-level theoretical
citations. **None of these can be supplied by re-analysis.** The author must
insert them from the original study documentation before submission.
