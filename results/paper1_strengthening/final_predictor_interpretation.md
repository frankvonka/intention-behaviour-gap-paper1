# Reviewer-Fix Pass — Task 4: Predictor Analysis Reframing

## Status: FIXED BY REWORDING

## What the frozen evidence shows

The predictor analysis (`scripts/18_profile_predictors.py`, output
`results/18_profile_predictors/multinomial_results.csv`) is a statsmodels
`MNLogit` with:
- Outcome: 6-profile membership (Profile 0 = reference)
- Predictors: 13 (8 psychometric constructs + 5 demographics)
- Contrasts: 13 × 5 = 65
- Correction: joint Benjamini–Hochberg FDR at α = 0.05
- McFadden pseudo-R² = 0.3035
- FDR-significant contrasts: **21 / 65**

## Why the original framing is not defensible

1. **Severe multicollinearity.** VIF diagnostics (`results/18B_predictor_multicollinearity/predictor_vif.csv`):
   - Max VIF = **67.44** (COVID); all 8 construct VIFs > 26.
   - Condition number (predictor correlation matrix) = **18.40**.
   - 1 pair |r| ≥ 0.70: SNO–COVID r = 0.7149.
   - Under these conditions, individual β coefficients are **not uniquely identified**;
     the leave-one-predictor-out sensitivity (`results/18B_predictor_multicollinearity/leave_one_predictor_out_diagnostics.csv`)
     shows the coefficient estimates shift substantially when any construct predictor is dropped.

2. **Cross-sectional design.** The data are a single-wave survey. No temporal ordering
   between predictors and profile membership can be established. Causal language
   ("predicts", "drives", "leads to") is not supported.

3. **Not pre-registered.** The 13-predictor set was assembled post hoc for an
   associational exploration; no a priori hypothesis was registered for any specific
   predictor–profile contrast.

4. **Profile 5 instability.** Profile 5 (the smallest directional-stable profile,
   48% / 52% across bootstraps) makes any predictor contrast involving Profile 5
   especially fragile.

## What the analysis *can* support

The analysis is defensible as an **exploratory associational analysis** that:
- Reports the *overall* model fit (McFadden R² = 0.3035) as a descriptive summary
  of how much of the profile-membership variation is linearly associated with the
  13 predictors jointly.
- Reports the *number* of FDR-significant contrasts (21/65) as a global signal that
  profile membership is not independent of the predictors.
- Identifies *which* contrasts survive FDR correction, **without interpreting any
  single coefficient as a uniquely identified effect**.

## Required manuscript changes

### 1. Re-label the section
- **From:** "Predictors of Profile Membership" / "Determinants of the Intention–Behaviour Gap"
- **To:** "Exploratory Analysis of Profile-Membership Associations"

### 2. Add an explicit exploratory disclaimer (first paragraph of the section)
> "We emphasize that this analysis is exploratory and associational, not causal.
> The design is cross-sectional, the predictor set was not pre-registered, and the
> construct predictors exhibit severe multicollinearity (VIF range 26.7–67.4;
> condition number 18.4). Individual coefficient estimates should therefore not be
> interpreted as uniquely identified effects. We report the joint model fit and the
> number of FDR-significant contrasts as descriptive summaries only."

### 3. Move detailed coefficient tables to the supplement
- **Main text:** report only (a) McFadden R² = 0.3035, (b) 21/65 FDR-significant,
  (c) the list of which predictors have ≥ 1 significant contrast, (d) the VIF
  caveat. A single summary sentence per predictor is sufficient.
- **Supplement:** the full 65-row coefficient table (β, SE, z, p, p_FDR, OR, 95% CI)
  and the VIF table, with the multicollinearity caveat repeated.

### 4. Remove / reword forbidden claims
Per the frozen Do-Not-Claim list and the Phase 22 claim audit:
- ❌ Do NOT say any single predictor "predicts" or "is associated with" a specific
  profile in a way that implies a uniquely identified effect.
- ❌ Do NOT report individual odds ratios as stable estimates.
- ❌ Do NOT use causal or temporal language.
- ✅ DO say "in this exploratory model, X of 65 contrasts were FDR-significant;
  the predictors that contributed most to the joint association were PU, PRI, and
  SNO (each with 4–5 significant contrasts)."

### 5. Add a Limitations sentence
> "Because the construct predictors are highly collinear, the individual
> coefficient estimates are unstable and should not be over-interpreted; the
> results are best read as indicating that profile membership is jointly
> associated with the predictor set, without identifying which specific
> predictor drives which profile."

## What does NOT need to change
- The model itself (MNLogit, Profile 0 reference, 13 predictors, joint BH-FDR) is
  a legitimate exploratory specification. No re-estimation is required.
- The 21/65 FDR-significant count is reproducible and defensible as a global signal.
- McFadden R² = 0.3035 is a legitimate descriptive fit index.

## Verification
- ✅ 65 contrasts confirmed in `results/18_profile_predictors/multinomial_results.csv`.
- ✅ 21/65 FDR-significant confirmed (p_FDR < 0.05).
- ✅ VIF table confirmed in `results/18B_predictor_multicollinearity/predictor_vif.csv`.
- ✅ Leave-one-out sensitivity confirmed in `results/18B_predictor_multicollinearity/leave_one_predictor_out_diagnostics.csv`.
- ✅ Cross-sectional design confirmed (single-wave `data.xls`, no time variable).
