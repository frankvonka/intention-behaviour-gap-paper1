"""
Phase 18 — Profile Predictor Analysis
=======================================
Analyzes which observed psychological, contextual, and demographic
variables predict membership in the frozen PRIMARY profiles
(K read dynamically from results/05_lpa_selection/selected_model.csv).

Outcome: frozen selected-K profile membership (Profile 0 = reference).
Predictors: ATT, CON, SNO, COVID, PU, PEU, PO, PRI (psychometric)
            age, gender, Education, Occupation, income (demographic)

Multinomial logistic regression via statsmodels MNLogit.
13 predictors x (K-1) non-reference profiles = 13 x (K-1) tests.
Benjamini-Hochberg FDR applied across ALL tests jointly.

No interpretation. No plots. No model refitting. No variable removal.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORES_PATH = "results/02_measurement/construct_scores.csv"
PHASE04_DIR = "results/04_lpa_estimation"
DATA_PATH = "data.xls"
RESULTS_DIR = "results/18_profile_predictors"

PHASE05_DIR = "results/05_lpa_selection"
K = int(pd.read_csv(os.path.join(PHASE05_DIR, "selected_model.csv"))["selected_K"].iloc[0])  # frozen primary solution (selected by Phase 05)
REFERENCE_PROFILE = 0

CONSTRUCT_PREDICTORS = ["ATT", "CON", "SNO", "COVID", "PU", "PEU", "PO", "PRI"]
DEMOGRAPHICS = ["age", "gender", "Education", "Occupation", "income"]
PREDICTORS = CONSTRUCT_PREDICTORS + DEMOGRAPHICS

ALPHA_FDR = 0.05


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def benjamini_hochberg(pvalues):
    """
    Benjamini-Hochberg FDR correction.
    p_adj(i) = min_{j>=i} [p_(j) * m / j], enforced monotonic.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adj_sorted = sorted_p * m / np.arange(1, m + 1)
    # Enforce monotonicity from largest rank down
    for i in range(m - 2, -1, -1):
        if adj_sorted[i] > adj_sorted[i + 1]:
            adj_sorted[i] = adj_sorted[i + 1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    result = np.empty(m)
    result[order] = adj_sorted
    return result


def main():
    ensure_dir(RESULTS_DIR)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    raw = pd.read_excel(DATA_PATH, sheet_name=0).copy()
    N = scores.shape[0]

    # Frozen selected-K classification
    kdir = os.path.join(PHASE04_DIR, f"K_{K}")
    post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
    labels = post_df["assigned_class"].values

    # ------------------------------------------------------------------
    # Profile outcome summary
    # ------------------------------------------------------------------
    outcome_rows = []
    for p in range(K):
        mask = labels == p
        outcome_rows.append({
            "profile": p,
            "N": int(mask.sum()),
            "percentage": float(mask.sum() / N * 100),
            "is_reference": bool(p == REFERENCE_PROFILE),
        })
    outcome_df = pd.DataFrame(outcome_rows)
    outcome_df.to_csv(os.path.join(RESULTS_DIR, "profile_outcome_summary.csv"), index=False)

    # ------------------------------------------------------------------
    # Predictor matrix
    # ------------------------------------------------------------------
    X = pd.concat([scores[CONSTRUCT_PREDICTORS], raw[DEMOGRAPHICS]], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    n_missing = int(X.isna().sum().sum())
    X_design = sm.add_constant(X)
    y = pd.Series(labels, index=scores.index)

    # ------------------------------------------------------------------
    # Fit multinomial logistic regression (Profile 0 = reference)
    # ------------------------------------------------------------------
    model = MNLogit(y, X_design)
    result = model.fit(method="bfgs", maxiter=1000, disp=False)

    params = result.params   # index = predictor names, columns = categories 1..K-1
    bse = result.bse
    z_vals = result.tvalues
    p_vals = result.pvalues

    # ------------------------------------------------------------------
    # Coefficient table: 13 predictors x (K-1) non-reference profiles = 13 x (K-1) tests
    # ------------------------------------------------------------------
    rows = []
    fdr_input = []
    categories = list(params.columns)
    # statsmodels MNLogit columns are positional 0..K-2, mapping to
    # non-reference profiles 1..K-1 in label order (Profile 0 is reference).
    for cat_pos in categories:
        target_profile = int(cat_pos) + 1
        for pred in params.index:
            if pred == "const":
                continue
            beta = float(params.loc[pred, cat_pos])
            se = float(bse.loc[pred, cat_pos])
            z = float(z_vals.loc[pred, cat_pos])
            p = float(p_vals.loc[pred, cat_pos])
            or_val = float(np.exp(beta))
            ci_lo = float(np.exp(beta - 1.96 * se))
            ci_hi = float(np.exp(beta + 1.96 * se))
            rows.append({
                "profile": target_profile,
                "reference_profile": REFERENCE_PROFILE,
                "predictor": pred,
                "beta": beta,
                "SE": se,
                "z": z,
                "p": p,
                "OR": or_val,
                "CI_lower": ci_lo,
                "CI_upper": ci_hi,
            })
            fdr_input.append(p)

    results_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # FDR correction across ALL 13 x (K-1) tests jointly
    # ------------------------------------------------------------------
    fdr_adjusted = benjamini_hochberg(fdr_input)
    results_df["p_FDR"] = fdr_adjusted
    results_df["significant_FDR"] = results_df["p_FDR"] < ALPHA_FDR

    # Reorder columns per specification
    results_df = results_df[[
        "profile", "reference_profile", "predictor", "beta", "SE", "z",
        "p", "p_FDR", "OR", "CI_lower", "CI_upper", "significant_FDR",
    ]]
    results_df.to_csv(os.path.join(RESULTS_DIR, "multinomial_results.csv"), index=False)

    # ------------------------------------------------------------------
    # Predictor summary
    # ------------------------------------------------------------------
    summary_rows = []
    for pred in PREDICTORS:
        sub = results_df[results_df["predictor"] == pred]
        summary_rows.append({
            "predictor": pred,
            "n_significant_contrasts": int(sub["significant_FDR"].sum()),
            "min_raw_p": float(sub["p"].min()),
            "min_FDR_p": float(sub["p_FDR"].min()),
            "largest_abs_beta": float(sub["beta"].abs().max()),
            "largest_OR": float(sub["OR"].max()),
            "smallest_OR": float(sub["OR"].min()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "predictor_summary.csv"), index=False)

    # ------------------------------------------------------------------
    # Model diagnostics
    # ------------------------------------------------------------------
    # VIF among predictors (design matrix without constant)
    X_vif = X.values.astype(float)
    vif_values = []
    for i in range(X_vif.shape[1]):
        vif_values.append(float(variance_inflation_factor(X_vif, i)))
    vif_df = pd.DataFrame({"predictor": PREDICTORS, "VIF": vif_values})

    # Condition number of the full design matrix (with constant, unscaled).
    # NOTE: this is NOT comparable to the correlation-matrix condition number
    # (condition_number_corr = 18.40 in Phase 18B). The design-matrix version
    # mixes raw scales (age ~ tens, Likert ~ ones) and a constant column, so a
    # large value here is expected and must not be quoted as multicollinearity
    # evidence. Quote condition_number_corr from Phase 18B instead.
    cond_number = float(np.linalg.cond(X_design.values.astype(float)))

    # McFadden pseudo-R2 = 1 - LL_full / LL_null
    llf = float(result.llf)
    llnull = float(result.llnull)
    mcfadden_r2 = float(1.0 - llf / llnull)

    converged = bool(result.mle_retvals.get("converged", False))

    diag_rows = [
        {"metric": "sample_size", "value": float(N)},
        {"metric": "n_outcome_classes", "value": float(K)},
        {"metric": "reference_profile", "value": float(REFERENCE_PROFILE)},
        {"metric": "n_predictors", "value": float(len(PREDICTORS))},
        {"metric": "n_tests", "value": float(len(fdr_input))},
        {"metric": "log_likelihood", "value": llf},
        {"metric": "log_likelihood_null", "value": llnull},
        {"metric": "AIC", "value": float(result.aic)},
        {"metric": "BIC", "value": float(result.bic)},
        {"metric": "McFadden_pseudo_R2", "value": mcfadden_r2},
        {"metric": "converged", "value": float(converged)},
        {"metric": "n_missing_predictor_values", "value": float(n_missing)},
        {"metric": "max_VIF", "value": float(max(vif_values))},
        {"metric": "condition_number_design_with_const_UNSCALED_DO_NOT_COMPARE", "value": cond_number},
        {"metric": "n_FDR_significant", "value": float(int(results_df["significant_FDR"].sum()))},
    ]
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(os.path.join(RESULTS_DIR, "model_diagnostics.csv"), index=False)
    vif_df.to_csv(os.path.join(RESULTS_DIR, "predictor_vif.csv"), index=False)

    # ------------------------------------------------------------------
    # Master results
    # ------------------------------------------------------------------
    master_rows = [
        {"item": "N", "value": float(N)},
        {"item": "K_frozen", "value": float(K)},
        {"item": "reference_profile", "value": float(REFERENCE_PROFILE)},
        {"item": "n_predictors", "value": float(len(PREDICTORS))},
        {"item": "n_tests", "value": float(len(fdr_input))},
        {"item": "log_likelihood", "value": llf},
        {"item": "AIC", "value": float(result.aic)},
        {"item": "BIC", "value": float(result.bic)},
        {"item": "McFadden_pseudo_R2", "value": mcfadden_r2},
        {"item": "converged", "value": float(converged)},
        {"item": "max_VIF", "value": float(max(vif_values))},
        {"item": "condition_number_design_with_const_UNSCALED_DO_NOT_COMPARE", "value": cond_number},
        {"item": "n_FDR_significant", "value": float(int(results_df["significant_FDR"].sum()))},
    ]
    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(os.path.join(RESULTS_DIR, "phase18_master.csv"), index=False)

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = f"""# Phase 18 — Profile Predictor Analysis

## Purpose
Analyze which observed psychological, contextual, and demographic
variables predict membership in the frozen PRIMARY profiles
(K = {K}, minimum BIC among non-degenerate fits, per
results/05_lpa_selection/selected_model.csv).
Purely computational. No interpretation, no plots, no literature.

## Model Specification
- **Outcome:** frozen K={K} profile membership (from
  results/04_lpa_estimation/K_{K}/posterior_probabilities.csv,
  column assigned_class). The selected solution is NOT refitted.
- **Reference category:** Profile {REFERENCE_PROFILE}
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
- For each predictor x non-reference profile (13 x {K - 1} = {len(fdr_input)} tests):
  - beta (coefficient)
  - SE (standard error)
  - z = beta / SE
  - p (raw two-sided p-value from normal approximation)
  - OR = exp(beta)
  - CI_lower = exp(beta - 1.96 * SE)
  - CI_upper = exp(beta + 1.96 * SE)
- **FDR correction:** Benjamini-Hochberg applied jointly across
  ALL {len(fdr_input)} tests (not per-profile).
  - p_adj(i) = min_{{j>=i}} [p_(j) * m / j], monotonic, clipped [0,1]
  - significant_FDR = (p_FDR < {ALPHA_FDR})
- **McFadden pseudo-R2** = 1 - LL_full / LL_null
- **VIF:** variance_inflation_factor on the predictor matrix
  (without constant), one per predictor
- **Condition number:** np.linalg.cond of the full design matrix
  (with constant)

## Output Files
- multinomial_results.csv — {len(fdr_input)} rows: profile, reference_profile,
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
"""
    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write(readme)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 18 — PROFILE PREDICTOR ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"N = {N}, K = {K} (frozen), reference profile = {REFERENCE_PROFILE}")
    print(f"Predictors ({len(PREDICTORS)}): {PREDICTORS}")
    print(f"Converged: {converged}")
    print(f"Log-likelihood: {llf:.4f}")
    print(f"AIC: {result.aic:.4f}")
    print(f"BIC: {result.bic:.4f}")
    print(f"McFadden pseudo-R2: {mcfadden_r2:.4f}")
    print(f"Max VIF: {max(vif_values):.4f}")
    print(f"Condition number: {cond_number:.4f}")
    print(f"Total tests: {len(fdr_input)}")
    print(f"FDR significant (p_FDR < {ALPHA_FDR}): {int(results_df['significant_FDR'].sum())}")
    print(f"\nPredictor summary:")
    print(summary_df.to_string(index=False))
    print(f"\nProfile outcome summary:")
    print(outcome_df.to_string(index=False))
    print(f"\nOutputs:")
    for fn in ["multinomial_results.csv", "predictor_summary.csv",
               "model_diagnostics.csv", "predictor_vif.csv",
               "profile_outcome_summary.csv", "phase18_master.csv", "README.md"]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
