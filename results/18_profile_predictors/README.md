# Phase 18 — Profile Predictor Analysis

## Purpose
Analyze which observed psychological, contextual, and demographic
variables predict membership in the frozen PRIMARY profiles
(K = 7, minimum BIC among non-degenerate fits, per
results/05_lpa_selection/selected_model.csv).
Purely computational. No interpretation, no plots, no literature.

## Model Specification
- **Outcome:** frozen K=7 profile membership (from
  results/04_lpa_estimation/K_7/posterior_probabilities.csv,
  column assigned_class). The selected solution is NOT refitted.
- **Reference category:** Profile 0
- **Model:** Multinomial logistic regression
  (statsmodels MNLogit, method='bfgs', maxiter=1000)
- **Predictors (13 total):**
  - Psychometric: ATT, CON, SNO, COVID, PU, PEU, PO, PRI
    (construct mean scores from results/02_measurement/construct_scores.csv)
  - Demographic: age, gender, Education, Occupation, income
    (raw columns from data.xls, treated as numeric)
- **No variable removal.** All 13 predictors retained regardless
  of significance.

## Calculations
- For each predictor x non-reference profile (13 x 6 = 78 tests):
  - beta (coefficient)
  - SE (standard error)
  - z = beta / SE
  - p (raw two-sided p-value from normal approximation)
  - OR = exp(beta)
  - CI_lower = exp(beta - 1.96 * SE)
  - CI_upper = exp(beta + 1.96 * SE)
- **FDR correction:** Benjamini-Hochberg applied jointly across
  ALL 78 tests (not per-profile).
  - p_adj(i) = min_{j>=i} [p_(j) * m / j], monotonic, clipped [0,1]
  - significant_FDR = (p_FDR < 0.05)
- **McFadden pseudo-R2** = 1 - LL_full / LL_null
- **VIF:** variance_inflation_factor on the predictor matrix
  (without constant), one per predictor
- **Condition number:** np.linalg.cond of the full design matrix
  (with constant)

## Output Files
- multinomial_results.csv — 78 rows: profile, reference_profile,
  predictor, beta, SE, z, p, p_FDR, OR, CI_lower, CI_upper,
  significant_FDR
- predictor_summary.csv — per-predictor: n significant contrasts,
  min raw p, min FDR p, largest |beta|, largest OR, smallest OR
- model_diagnostics.csv — sample size, classes, LL, AIC, BIC,
  McFadden R2, convergence, max VIF, condition number, n FDR sig
- predictor_vif.csv — VIF per predictor
- profile_outcome_summary.csv — profile sizes and percentages
- phase18_master.csv — summary metrics
- README.md — this file

## No Modifications
Previous phase outputs untouched. Dataset unchanged. No plots.
No variable removal. No per-profile FDR.
