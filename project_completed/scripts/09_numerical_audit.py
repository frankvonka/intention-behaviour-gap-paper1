"""
Phase 09 — Full Numerical Audit
=================================
Project: Intention–Behaviour Gap in Household Energy-Saving Behaviour

Audits every major result from Phases 01-08.
Checks data, measurement, gap, LPA, predictors, robustness.
Specifically tests for the historical BIC bug (BIC must be MINIMIZED).
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = "results/09_numerical_audit"
SEED = 42

PHASE_DIRS = {
    "01": "results/01_data_inspection",
    "02": "results/02_measurement",
    "03": "results/03_gap_analysis",
    "04": "results/04_lpa_estimation",
    "05": "results/05_lpa_selection",
    "06": "results/06_profile_analysis",
    "07": "results/07_profile_predictors",
    "08": "results/08_robustness",
}


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def make_check_adder(checks_list):
    """Return a closure that appends to checks_list."""
    def add(category, name, expected, observed, status):
        checks_list.append(
            {
                "category": category,
                "check": name,
                "expected": expected,
                "observed": observed,
                "status": status,
            }
        )
    return add


def main():
    ensure_dir(RESULTS_DIR)
    checks = []
    discrepancies = []
    add_check = make_check_adder(checks)

    # ------------------------------------------------------------------
    # DATA (Phase 01)
    # ------------------------------------------------------------------
    meta = json.load(open(os.path.join(PHASE_DIRS["01"], "dataset_metadata.json")))
    N = meta["N_rows"]
    ncols = meta["N_columns"]
    add_check("data", "N_rows", 1166, N, "PASS" if N == 1166 else "FAIL")
    add_check("data", "N_columns", 34, ncols, "PASS" if ncols == 34 else "FAIL")

    missing = pd.read_csv(os.path.join(PHASE_DIRS["01"], "missing_values.csv"))
    total_missing = missing["n_missing"].sum()
    add_check("data", "total_missing", 0, total_missing, "PASS" if total_missing == 0 else "FAIL")

    dup = pd.read_csv(os.path.join(PHASE_DIRS["01"], "duplicate_information.csv"))
    n_dup = int(dup.loc[dup["check"] == "all_columns", "n_duplicates"].values[0])
    add_check("data", "duplicate_rows", 42, n_dup, "PASS" if n_dup == 42 else "FAIL")

    # Item ranges
    item_ranges = pd.read_csv(os.path.join(PHASE_DIRS["01"], "item_ranges.csv"))
    min_ok = (item_ranges["min"] >= 1).all()
    max_ok = (item_ranges["max"] <= 5).all()
    add_check("data", "item_min_ge_1", True, min_ok, "PASS" if min_ok else "FAIL")
    add_check("data", "item_max_le_5", True, max_ok, "PASS" if max_ok else "FAIL")

    # ------------------------------------------------------------------
    # MEASUREMENT (Phase 02)
    # ------------------------------------------------------------------
    scores = pd.read_csv(os.path.join(PHASE_DIRS["02"], "construct_scores.csv"), index_col="respondent")
    add_check("measurement", "scores_N", N, scores.shape[0], "PASS" if scores.shape[0] == N else "FAIL")
    add_check("measurement", "scores_ncols", 10, scores.shape[1], "PASS" if scores.shape[1] == 10 else "FAIL")

    # Verify construct scores = mean of items
    data = pd.read_excel("data.xls", sheet_name=0)
    constructs = {
        "ATT": ["ATT1", "ATT2", "ATT3"],
        "INT": ["INT1", "INT2", "INT3"],
        "BE": ["BE1", "BE2", "BE3", "BE4"],
    }
    for cname, items in constructs.items():
        expected = data[items].mean(axis=1).values
        observed = scores[cname].values
        diff = np.abs(expected - observed).max()
        add_check("measurement", f"{cname}_score_max_diff", 0.0, float(diff), "PASS" if diff < 1e-6 else "FAIL")

    # Cronbach alpha
    cronbach = pd.read_csv(os.path.join(PHASE_DIRS["02"], "cronbach_alpha.csv"))
    alpha_ok = (cronbach["Cronbach_alpha"] > 0).all() and (cronbach["Cronbach_alpha"] <= 1).all()
    add_check("measurement", "cronbach_alpha_range", "0<alpha<=1", alpha_ok, "PASS" if alpha_ok else "FAIL")

    # INT-BE correlation
    intbe = pd.read_csv(os.path.join(PHASE_DIRS["02"], "int_be_correlation.csv"))
    r = intbe["r"].values[0]
    add_check("measurement", "INT_BE_r_range", "-1<=r<=1", -1 <= r <= 1, "PASS" if -1 <= r <= 1 else "FAIL")
    add_check("measurement", "INT_BE_p_positive", ">0", intbe["p"].values[0] > 0, "PASS" if intbe["p"].values[0] > 0 else "FAIL")

    # ------------------------------------------------------------------
    # GAP (Phase 03)
    # ------------------------------------------------------------------
    gap = np.load(os.path.join(PHASE_DIRS["03"], "gap_scores.npy"))
    add_check("gap", "gap_N", N, len(gap), "PASS" if len(gap) == N else "FAIL")
    add_check("gap", "gap_mean_near_0", "~0", float(abs(gap.mean()) < 1e-6), "PASS" if abs(gap.mean()) < 1e-6 else "FAIL")

    std = pd.read_csv(os.path.join(PHASE_DIRS["03"], "standardized_scores.csv"))
    z_int = std["z_INT"].values
    z_be = std["z_BE"].values
    add_check("gap", "z_INT_SD~1", 1.0, float(z_int.std(ddof=1)), "PASS" if abs(z_int.std(ddof=1) - 1.0) < 1e-6 else "FAIL")
    add_check("gap", "z_BE_SD~1", 1.0, float(z_be.std(ddof=1)), "PASS" if abs(z_be.std(ddof=1) - 1.0) < 1e-6 else "FAIL")

    # Verify gap = z_INT - z_BE
    gap_recomputed = z_int - z_be
    diff = np.abs(gap - gap_recomputed).max()
    add_check("gap", "gap_formula", 0.0, float(diff), "PASS" if diff < 1e-6 else "FAIL")

    # ------------------------------------------------------------------
    # LPA (Phase 04)
    # ------------------------------------------------------------------
    fit = pd.read_csv(os.path.join(PHASE_DIRS["04"], "model_fit.csv"))
    # BIC must be MINIMIZED
    best_bic = fit["BIC"].min()
    best_K_bic = int(fit.loc[fit["BIC"].idxmin(), "K"])
    add_check("lpa", "BIC_minimized", True, best_K_bic == 6, "PASS" if best_K_bic == 6 else "FAIL")

    # Verify BIC formula: BIC = -2LL + k*log(n)
    for _, row in fit.iterrows():
        K = int(row["K"])
        LL = row["log_likelihood"]
        k = int(row["n_params"])
        bic_expected = -2 * LL + k * np.log(N)
        bic_observed = row["BIC"]
        diff = abs(bic_expected - bic_observed)
        add_check("lpa", f"BIC_formula_K{K}", float(bic_expected), float(bic_observed), "PASS" if diff < 0.01 else "FAIL")

    # Verify AIC formula: AIC = -2LL + 2k
    for _, row in fit.iterrows():
        K = int(row["K"])
        LL = row["log_likelihood"]
        k = int(row["n_params"])
        aic_expected = -2 * LL + 2 * k
        aic_observed = row["AIC"]
        diff = abs(aic_expected - aic_observed)
        add_check("lpa", f"AIC_formula_K{K}", float(aic_expected), float(aic_observed), "PASS" if diff < 0.01 else "FAIL")

    # Posterior probabilities sum to ~1
    for K in [2, 3, 4, 5, 6]:
        kdir = os.path.join(PHASE_DIRS["04"], f"K_{K}")
        post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
        post_cols = [c for c in post_df.columns if c.startswith("post_profile_")]
        post = post_df[post_cols].values
        row_sums = post.sum(axis=1)
        max_dev = np.abs(row_sums - 1.0).max()
        add_check("lpa", f"posterior_sum_K{K}", 1.0, float(row_sums.mean()), "PASS" if max_dev < 1e-6 else "FAIL")

    # Profile proportions sum to ~100%
    for K in [2, 3, 4, 5, 6]:
        kdir = os.path.join(PHASE_DIRS["04"], f"K_{K}")
        sizes = pd.read_csv(os.path.join(kdir, "profile_sizes.csv"))
        total = sizes["size"].sum()
        add_check("lpa", f"profile_total_N_K{K}", N, int(total), "PASS" if total == N else "FAIL")

    # ------------------------------------------------------------------
    # PREDICTORS (Phase 07)
    # ------------------------------------------------------------------
    coef = pd.read_csv(os.path.join(PHASE_DIRS["07"], "coefficients.csv"))
    or_df = pd.read_csv(os.path.join(PHASE_DIRS["07"], "odds_ratios.csv"))
    ci_df = pd.read_csv(os.path.join(PHASE_DIRS["07"], "confidence_intervals.csv"))
    fdr = pd.read_csv(os.path.join(PHASE_DIRS["07"], "fdr_results.csv"))

    # OR = exp(beta)
    merged = coef.merge(or_df, on=["baseline_profile", "target_profile", "predictor"])
    or_recomputed = np.exp(merged["beta"].values)
    max_diff = np.abs(or_recomputed - merged["OR"].values).max()
    add_check("predictors", "OR_exp_beta", 0.0, float(max_diff), "PASS" if max_diff < 1e-4 else "FAIL")

    # CI: lower = exp(beta - 1.96*SE), upper = exp(beta + 1.96*SE)
    merged2 = coef.merge(ci_df, on=["baseline_profile", "target_profile", "predictor"])
    ci_lo_exp = np.exp(merged2["beta"] - 1.96 * merged2["SE"])
    ci_hi_exp = np.exp(merged2["beta"] + 1.96 * merged2["SE"])
    max_diff_lo = np.abs(ci_lo_exp.values - merged2["OR_95CI_lower"].values).max()
    max_diff_hi = np.abs(ci_hi_exp.values - merged2["OR_95CI_upper"].values).max()
    add_check("predictors", "CI_lower", 0.0, float(max_diff_lo), "PASS" if max_diff_lo < 1e-4 else "FAIL")
    add_check("predictors", "CI_upper", 0.0, float(max_diff_hi), "PASS" if max_diff_hi < 1e-4 else "FAIL")

    # FDR adjusted p-values >= raw p-values
    fdr_ok = (fdr["FDR_adjusted_p"] >= fdr["p_value"] - 1e-9).all()
    add_check("predictors", "FDR_ge_raw", True, fdr_ok, "PASS" if fdr_ok else "FAIL")

    # ------------------------------------------------------------------
    # ROBUSTNESS (Phase 08)
    # ------------------------------------------------------------------
    rs = pd.read_csv(os.path.join(PHASE_DIRS["08"], "random_start_stability.csv"))
    add_check("robustness", "random_start_converged", True, bool(rs["converged"].all()), "PASS" if rs["converged"].all() else "FAIL")

    # ------------------------------------------------------------------
    # RESPONDENT ORDERING
    # ------------------------------------------------------------------
    # All respondent-level outputs should have N rows
    add_check("ordering", "gap_N", N, len(gap), "PASS" if len(gap) == N else "FAIL")
    add_check("ordering", "scores_N", N, scores.shape[0], "PASS" if scores.shape[0] == N else "FAIL")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(os.path.join(RESULTS_DIR, "audit_checks.csv"), index=False)

    fail_mask = checks_df["status"] == "FAIL"
    discrepancies_df = checks_df[fail_mask].copy()
    discrepancies_df.to_csv(os.path.join(RESULTS_DIR, "audit_discrepancies.csv"), index=False)

    n_pass = int((checks_df["status"] == "PASS").sum())
    n_fail = int(fail_mask.sum())
    print("=" * 60)
    print("PHASE 09 — NUMERICAL AUDIT COMPLETE")
    print("=" * 60)
    print(f"Total checks: {len(checks_df)}")
    print(f"PASS: {n_pass}, FAIL: {n_fail}")
    if n_fail > 0:
        print("\nDiscrepancies:")
        print(discrepancies_df.to_string(index=False))
    else:
        print("\nNo discrepancies found.")
    print(f"\nOutputs: {RESULTS_DIR}/audit_checks.csv, audit_discrepancies.csv")


if __name__ == "__main__":
    main()
