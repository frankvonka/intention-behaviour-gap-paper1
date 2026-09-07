# Phase 18B — Profile Predictor Multicollinearity Audit

## Purpose
Investigate the multicollinearity observed in the Phase 18
multinomial predictor model. Diagnostic and sensitivity only.
No model selection, no interpretation, no variable removal beyond
the one-at-a-time leave-out diagnostics specified.

## Tests Performed
1. **Primary model reproduction** — same specification as Phase 18
2. **VIF diagnostic** — variance_inflation_factor for all 13 predictors
3. **Predictor correlation matrix** — Pearson r (N = 1166)
4. **Condition diagnostics** — condition number, eigenvalues of
   predictor correlation matrix, smallest eigenvalue
5. **High-correlation pairs** — pairs with |r| >= 0.7
6. **Leave-one-predictor-out** — drop each of the 8 construct
   predictors one at a time (demographics always retained)
7. **Standardized-continuous sensitivity** — standardize all
   continuous predictors (gender NOT standardized)
8. **Coefficient stability table** — primary coefficients vs
   leave-one-out vs standardized

## Models
- Outcome: frozen K=7 profile membership (minimum BIC among non-degenerate
  fits, per results/05_lpa_selection/selected_model.csv)
- Reference: Profile 0
- Estimator: statsmodels MNLogit, BFGS, maxiter=1000
- Predictors (primary, 13): ['ATT', 'CON', 'SNO', 'COVID', 'PU', 'PEU', 'PO', 'PRI', 'age', 'gender', 'Education', 'Occupation', 'income']
- Predictors (leave-one-out, 12 each): drop one construct at a time

## FDR Procedure
- Benjamini-Hochberg within each model across ALL predictor/profile
  contrasts (NOT per-profile, NOT across models)
- alpha = 0.05

## Primary Model Reproduction
- LL = -1480.3265
- AIC = 3128.6531
- BIC = 3553.8052
- McFadden pseudo-R2 = 0.2982
- Converged = True
- max |beta diff vs Phase 18| = 0.000000
- max |OR diff vs Phase 18|   = 0.000000

## VIF (descending)
predictor       VIF
    COVID 67.442218
      SNO 64.934818
       PU 61.105732
      CON 59.037423
      PRI 51.246235
      ATT 43.214189
       PO 35.714168
      PEU 26.665426
      age 13.924741
Education 13.780571
   income 12.870950

## Condition Diagnostics
- condition number (corr) = 18.4020
- smallest eigenvalue     = 0.260886
- n pairs |r| >= 0.70     = 1

## Leave-One-Out Diagnostics
specification  log_likelihood         AIC         BIC  McFadden_pseudo_R2  converged  n_predictors  n_tests
    A_primary    -1480.326532 3128.653064 3553.805151            0.298248       True            13       78
     drop_ATT    -1485.867712 3127.735423 3522.519504            0.295621       True            12       72
     drop_CON    -1483.693592 3123.387185 3518.171265            0.296652       True            12       72
     drop_SNO    -1521.501233 3199.002466 3593.786547            0.278729       True            12       72
   drop_COVID    -1483.614621 3123.229243 3518.013323            0.296689       True            12       72
      drop_PU    -1533.736807 3223.473615 3618.257696            0.272929       True            12       72
     drop_PEU    -1494.779624 3145.559248 3540.343329            0.291396       True            12       72
      drop_PO    -1485.159133 3126.318266 3521.102346            0.295957       True            12       72
     drop_PRI    -1555.794685 3267.589369 3662.373450            0.262472       True            12       72

## High-Correlation Pairs (top 20)
predictor_1 predictor_2  pearson_r    abs_r
        SNO       COVID   0.714871 0.714871

## Coefficient Stability
- Comparison table in `coefficient_stability.csv`
- Numerical stability summary in `phase18B_master.csv`

## No Modifications
Phase 18 outputs untouched. Dataset unchanged. No plots.
