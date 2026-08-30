"""
Phase 07 — Predictors of Profile Membership
==============================================
Project: Intention–Behaviour Gap in Household Energy-Saving Behaviour

Uses the selected LPA class assignment as the dependent variable.
Predictors: ATT, CON, SNO, COVID, PU, PEU, PO, PRI + demographics
(age, gender, Education, Occupation, income).

Multinomial logistic regression via statsmodels MNLogit.
For each coefficient: beta, SE, z, p, OR, 95% CI for OR.
FDR correction (Benjamini-Hochberg) applied to all tested p-values.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORES_PATH = "results/02_measurement/construct_scores.csv"
PHASE04_DIR = "results/04_lpa_estimation"
PHASE05_DIR = "results/05_lpa_selection"
DATA_PATH = "data.xls"
RESULTS_DIR = "results/07_profile_predictors"
SEED = 42

CONSTRUCT_PREDICTORS = ["ATT", "CON", "SNO", "COVID", "PU", "PEU", "PO", "PRI"]
DEMOGRAPHICS = ["age", "gender", "Education", "Occupation", "income"]


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
    ranked = np.empty(m)
    ranked[order] = np.arange(1, m + 1)
    adjusted = p * m / ranked
    # Enforce monotonicity from the largest rank down
    cummin = adjusted.copy()
    for i in range(m - 2, -1, -1):
        # rank i+1 corresponds to order[i]
        pass
    # Recompute properly: sort, adjust, enforce monotonic descending
    sorted_p = np.sort(p)
    adj_sorted = sorted_p * m / np.arange(1, m + 1)
    # Enforce monotonicity: from largest to smallest rank
    for i in range(m - 2, -1, -1):
        if adj_sorted[i] > adj_sorted[i + 1]:
            adj_sorted[i] = adj_sorted[i + 1]
    # Clip to [0,1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    # Map back to original order
    result = np.empty(m)
    result[order] = adj_sorted
    return result


def main():
    ensure_dir(RESULTS_DIR)
    np.random.seed(SEED)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    raw = pd.read_excel(DATA_PATH, sheet_name=0).copy()
    N = scores.shape[0]

    # Selected K
    selected = pd.read_csv(os.path.join(PHASE05_DIR, "selected_model.csv"))
    K = int(selected["selected_K"].values[0])

    # Class assignments
    kdir = os.path.join(PHASE04_DIR, f"K_{K}")
    post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
    labels = post_df["assigned_class"].values

    # Build predictor matrix
    pred_cols = CONSTRUCT_PREDICTORS + DEMOGRAPHICS
    X = pd.concat([scores[CONSTRUCT_PREDICTORS], raw[DEMOGRAPHICS]], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = sm.add_constant(X)
    y = pd.Series(labels, index=scores.index)

    # ------------------------------------------------------------------
    # Fit multinomial logistic regression
    # ------------------------------------------------------------------
    model = MNLogit(y, X)
    # Use a robust optimizer
    result = model.fit(method="bfgs", maxiter=1000, disp=False)

    # ------------------------------------------------------------------
    # Extract coefficients
    # ------------------------------------------------------------------
    # params: DataFrame (predictors x K-1 categories), baseline = 0
    params = result.params  # index = predictor names, columns = categories 1..K-1
    bse = result.bse
    z_vals = result.tvalues
    p_vals = result.pvalues

    coef_rows = []
    or_rows = []
    ci_rows = []
    p_rows = []
    fdr_input = []

    categories = [c for c in params.columns.tolist()]

    for cat in categories:
        for pred in params.index:
            beta = float(params.loc[pred, cat])
            se = float(bse.loc[pred, cat])
            z = float(z_vals.loc[pred, cat])
            p = float(p_vals.loc[pred, cat])
            or_val = float(np.exp(beta))
            ci_lo = float(np.exp(beta - 1.96 * se))
            ci_hi = float(np.exp(beta + 1.96 * se))

            coef_rows.append(
                {
                    "baseline_profile": 0,
                    "target_profile": int(cat),
                    "predictor": pred,
                    "beta": beta,
                    "SE": se,
                    "z": z,
                    "p_value": p,
                }
            )
            or_rows.append(
                {
                    "baseline_profile": 0,
                    "target_profile": int(cat),
                    "predictor": pred,
                    "OR": or_val,
                }
            )
            ci_rows.append(
                {
                    "baseline_profile": 0,
                    "target_profile": int(cat),
                    "predictor": pred,
                    "OR_95CI_lower": ci_lo,
                    "OR_95CI_upper": ci_hi,
                }
            )
            p_rows.append(
                {
                    "baseline_profile": 0,
                    "target_profile": int(cat),
                    "predictor": pred,
                    "p_value": p,
                }
            )
            fdr_input.append(p)

    coef_df = pd.DataFrame(coef_rows)
    or_df = pd.DataFrame(or_rows)
    ci_df = pd.DataFrame(ci_rows)
    p_df = pd.DataFrame(p_rows)

    # ------------------------------------------------------------------
    # FDR correction (Benjamini-Hochberg)
    # ------------------------------------------------------------------
    fdr_adjusted = benjamini_hochberg(fdr_input)
    fdr_df = coef_df[["baseline_profile", "target_profile", "predictor", "p_value"]].copy()
    fdr_df["FDR_adjusted_p"] = fdr_adjusted
    fdr_df["significant_FDR_05"] = fdr_df["FDR_adjusted_p"] < 0.05

    # Save
    coef_df.to_csv(os.path.join(RESULTS_DIR, "coefficients.csv"), index=False)
    or_df.to_csv(os.path.join(RESULTS_DIR, "odds_ratios.csv"), index=False)
    ci_df.to_csv(os.path.join(RESULTS_DIR, "confidence_intervals.csv"), index=False)
    p_df.to_csv(os.path.join(RESULTS_DIR, "p_values.csv"), index=False)
    fdr_df.to_csv(os.path.join(RESULTS_DIR, "fdr_results.csv"), index=False)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 07 — PROFILE PREDICTORS COMPLETE")
    print("=" * 60)
    print(f"Selected K = {K}, baseline profile = 0")
    print(f"Predictors: {pred_cols}")
    print(f"N = {N}")
    print(f"Model converged: {result.mle_retvals.get('converged', 'unknown')}")
    print(f"Log-likelihood: {result.llf:.4f}")
    print(f"AIC: {result.aic:.4f}")
    print(f"BIC: {result.bic:.4f}")

    n_tests = len(fdr_input)
    n_sig = int(fdr_df["significant_FDR_05"].sum())
    print(f"\nFDR (BH) tests: {n_tests}, significant at 0.05: {n_sig}")

    print("\nKey outputs:")
    for fn in [
        "coefficients.csv",
        "odds_ratios.csv",
        "confidence_intervals.csv",
        "p_values.csv",
        "fdr_results.csv",
    ]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
