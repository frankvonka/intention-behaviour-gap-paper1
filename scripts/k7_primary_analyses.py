#!/usr/bin/env python3
"""K=7-primary analyses — the three pieces that were only ever run for K=6.

  A. GAP variance decomposition for K=7 (validated on frozen K=6 first)
  B. Profile predictors (MNLogit) on K=7 membership (validated on frozen K=6 first)
  C. Duplicate-row sensitivity: K=7 GMM refit on the unique sample (N computed at runtime)

All inputs are frozen files; no frozen file is modified.
Outputs -> results/k7_primary/
"""
import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.optimize import linear_sum_assignment
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.stats.outliers_influence import variance_inflation_factor

SEED = 42
N_INIT = 1000
MAX_ITER = 500
REG_COVAR = 1e-6

PF = "results/paper1_final"
S = "results/paper1_strengthening"
LPA = "results/04_lpa_estimation"
OUT = "results/k7_primary"
ALPHA_FDR = 0.05

CONSTRUCT_PREDICTORS = ["ATT", "CON", "SNO", "COVID", "PU", "PEU", "PO", "PRI"]
DEMOGRAPHICS = ["age", "gender", "Education", "Occupation", "income"]
PREDICTORS = CONSTRUCT_PREDICTORS + DEMOGRAPHICS

CONSTRUCTS = {"INT": ["INT1", "INT2", "INT3"], "BE": ["BE1", "BE2", "BE3", "BE4"]}


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def safe_entropy(posterior):
    p = np.clip(posterior, 1e-15, 1.0)
    n, K = posterior.shape
    return float(-np.sum(p * np.log(p)) / (n * np.log(K)))


def count_params_full(K, n_features):
    return int(K * n_features + K * n_features * (n_features + 1) / 2.0 + (K - 1))


def benjamini_hochberg(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adj_sorted = sorted_p * m / np.arange(1, m + 1)
    for i in range(m - 2, -1, -1):
        if adj_sorted[i] > adj_sorted[i + 1]:
            adj_sorted[i] = adj_sorted[i + 1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    result = np.empty(m)
    result[order] = adj_sorted
    return result


# =====================================================================
# A. GAP variance decomposition
# =====================================================================
def decomposition_row(labels, K, GAP, N):
    profile_means, within_variances, weights = [], [], []
    for p in range(K):
        mask = labels == p
        gap_p = GAP[mask]
        profile_means.append(float(gap_p.mean()))
        within_variances.append(float(gap_p.var(ddof=1)))
        weights.append(int(mask.sum()) / N)
    weights_arr = np.array(weights)
    grand_mean = np.sum(weights_arr * np.array(profile_means))
    between_var = np.sum(weights_arr * (np.array(profile_means) - grand_mean) ** 2)
    within_var = np.sum(weights_arr * np.array(within_variances)) if K > 1 else 0.0
    total_var = float(GAP.var(ddof=1))
    return {
        "K": K,
        "between_profile_variance": float(between_var),
        "within_profile_variance": float(within_var),
        "total_variance": total_var,
        "between_over_total": float(between_var / total_var) if total_var > 0 else np.nan,
    }


def load_labels(K):
    """assigned_class if present, else argmax over post_profile_* columns."""
    path = os.path.join(LPA, f"K_{K}", "posterior_probabilities.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "assigned_class" in df.columns:
        return df["assigned_class"].values.astype(int)
    cols = [c for c in df.columns if c.startswith("post_profile_")]
    return df[cols].values.argmax(axis=1).astype(int)


def part_a(scores):
    print("=" * 70)
    print("A. GAP VARIANCE DECOMPOSITION (validate on K=6, compute K=2-7)")
    print("=" * 70)
    z_INT = (scores["INT"] - scores["INT"].mean()) / scores["INT"].std(ddof=1)
    z_BE = (scores["BE"] - scores["BE"].mean()) / scores["BE"].std(ddof=1)
    GAP = (z_INT - z_BE).values
    N = len(GAP)

    rows, validation = [], {}
    for K in range(2, 8):
        # K=7 is now the PRIMARY solution — Phase 04 writes K_7 directly,
        # so all K in 2..7 load labels from the Phase 04 per-K directory.
        labels = load_labels(K)
        if labels is None:
            print(f"  K={K}: posterior file missing, skipped")
            continue
        row = decomposition_row(labels, K, GAP, N)
        rows.append(row)
        if K == 6:
            frozen = pd.read_csv(f"{PF}/03_intention_behavior_gap/gap_variance_decomposition.csv")
            f6 = frozen[frozen.K == 6].iloc[0]
            ok = bool(np.isclose(row["between_over_total"], f6.between_over_total, atol=1e-8)
                      and np.isclose(row["between_profile_variance"], f6.between_profile_variance, atol=1e-8))
            validation["K6_decomposition"] = {
                "computed": row["between_over_total"], "frozen": float(f6.between_over_total), "match": ok}
    dec = pd.DataFrame(rows)
    dec.to_csv(os.path.join(OUT, "gap_variance_decomposition_K2_K7.csv"), index=False)

    # per-profile gap stats for K=7
    labels7 = load_labels(7)
    gap_rows = []
    for p in range(7):
        gap_p = GAP[labels7 == p]
        gap_rows.append({
            "profile": p, "N": int((labels7 == p).sum()),
            "mean_GAP": float(gap_p.mean()), "SD_GAP": float(gap_p.std(ddof=1)),
            "median_GAP": float(np.median(gap_p)),
            "pct_INT_gt_BE": float((gap_p > 0).mean() * 100),
        })
    pd.DataFrame(gap_rows).to_csv(os.path.join(OUT, "k7_per_profile_gap.csv"), index=False)

    # Full-sample K=7 profile table on the original 1-5 scale
    # (pure join of primary-solution posteriors x frozen scores; no model refit)
    labels7_full = load_labels(7)
    prof_rows = []
    for p in range(7):
        m = labels7_full == p
        prof_rows.append({
            "profile": p,
            "N": int(m.sum()),
            "percentage": float(m.sum() / len(labels7_full) * 100),
            "INT_mean": float(scores["INT"].values[m].mean()),
            "BE_mean": float(scores["BE"].values[m].mean()),
            "INT_minus_BE": float(scores["INT"].values[m].mean() - scores["BE"].values[m].mean()),
            "mean_z_INT": float(z_INT.values[m].mean()),
            "mean_z_BE": float(z_BE.values[m].mean()),
            "mean_GAP": float(GAP[m].mean()),
        })
    pd.DataFrame(prof_rows).to_csv(os.path.join(OUT, "k7_profile_table_full_sample.csv"), index=False)
    print("\nK=7 full-sample profile table (1-5 scale):")
    print(pd.DataFrame(prof_rows).to_string(index=False))

    print(dec.to_string(index=False))
    print(f"Validation K=6: {validation['K6_decomposition']}")
    return validation


# =====================================================================
# B. Profile predictors (MNLogit)
# =====================================================================
def run_predictors(labels, K, outdir):
    ensure_dir(outdir)
    scores = pd.read_csv(f"{PF}/02_measurement/construct_scores.csv", index_col="respondent")
    raw = pd.read_excel("data.xls", sheet_name=0).copy()
    N = scores.shape[0]

    X = pd.concat([scores[CONSTRUCT_PREDICTORS], raw[DEMOGRAPHICS]], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    n_missing = int(X.isna().sum().sum())
    X_design = sm.add_constant(X)
    y = pd.Series(labels, index=scores.index)

    model = MNLogit(y, X_design)
    result = model.fit(method="bfgs", maxiter=1000, disp=False)
    params, bse, z_vals, p_vals = result.params, result.bse, result.tvalues, result.pvalues

    rows, fdr_input = [], []
    for cat_pos in params.columns:
        target = int(cat_pos) + 1
        for pred in params.index:
            if pred == "const":
                continue
            beta = float(params.loc[pred, cat_pos])
            se = float(bse.loc[pred, cat_pos])
            p = float(p_vals.loc[pred, cat_pos])
            rows.append({
                "profile": target, "reference_profile": 0, "predictor": pred,
                "beta": beta, "SE": se, "z": float(z_vals.loc[pred, cat_pos]), "p": p,
                "OR": float(np.exp(beta)),
                "CI_lower": float(np.exp(beta - 1.96 * se)),
                "CI_upper": float(np.exp(beta + 1.96 * se)),
            })
            fdr_input.append(p)
    res = pd.DataFrame(rows)
    res["p_FDR"] = benjamini_hochberg(fdr_input)
    res["significant_FDR"] = res["p_FDR"] < ALPHA_FDR
    res = res[["profile", "reference_profile", "predictor", "beta", "SE", "z",
               "p", "p_FDR", "OR", "CI_lower", "CI_upper", "significant_FDR"]]
    res.to_csv(os.path.join(outdir, "multinomial_results.csv"), index=False)

    X_vif = X.values.astype(float)
    vif_values = [float(variance_inflation_factor(X_vif, i)) for i in range(X_vif.shape[1])]
    llf, llnull = float(result.llf), float(result.llnull)
    diag = pd.DataFrame([
        {"metric": "sample_size", "value": float(N)},
        {"metric": "n_outcome_classes", "value": float(K)},
        {"metric": "reference_profile", "value": 0.0},
        {"metric": "n_predictors", "value": float(len(PREDICTORS))},
        {"metric": "n_tests", "value": float(len(fdr_input))},
        {"metric": "log_likelihood", "value": llf},
        {"metric": "log_likelihood_null", "value": llnull},
        {"metric": "AIC", "value": float(result.aic)},
        {"metric": "BIC", "value": float(result.bic)},
        {"metric": "McFadden_pseudo_R2", "value": float(1.0 - llf / llnull)},
        {"metric": "converged", "value": float(bool(result.mle_retvals.get("converged", False)))},
        {"metric": "n_missing_predictor_values", "value": float(n_missing)},
        {"metric": "max_VIF", "value": float(max(vif_values))},
        {"metric": "condition_number_design_with_const_UNSCALED_DO_NOT_COMPARE", "value": float(np.linalg.cond(X_design.values.astype(float)))},
        {"metric": "n_FDR_significant", "value": float(int(res["significant_FDR"].sum()))},
    ])
    diag.to_csv(os.path.join(outdir, "model_diagnostics.csv"), index=False)

    summary_rows = []
    for pred in PREDICTORS:
        sub = res[res["predictor"] == pred]
        summary_rows.append({
            "predictor": pred,
            "n_significant_contrasts": int(sub["significant_FDR"].sum()),
            "min_raw_p": float(sub["p"].min()),
            "min_FDR_p": float(sub["p_FDR"].min()),
            "largest_abs_beta": float(sub["beta"].abs().max()),
            "largest_OR": float(sub["OR"].max()),
            "smallest_OR": float(sub["OR"].min()),
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(outdir, "predictor_summary.csv"), index=False)
    return res, diag, llf, llnull


def part_b(scores):
    print("=" * 70)
    print("B. PROFILE PREDICTORS (validate on K=6 vs frozen, compute K=7)")
    print("=" * 70)
    labels6 = load_labels(6)
    _, diag6, llf6, llnull6 = run_predictors(labels6, 6, os.path.join(OUT, "_validation_k6"))
    # Validate against the STATIC frozen K=6 copy (results/paper1_final was
    # frozen when K=6 was primary; results/18_profile_predictors now holds
    # the K=7 rerun after the primary-solution switch).
    frozen = pd.read_csv(f"{PF}/06_predictors/multinomial_results.csv")
    rerun = pd.read_csv(os.path.join(OUT, "_validation_k6", "multinomial_results.csv"))
    num = ["beta", "SE", "z", "p", "p_FDR", "OR", "CI_lower", "CI_upper"]
    ok = (len(frozen) == len(rerun)
          and (frozen[num].values.shape == rerun[num].values.shape)
          and bool(np.allclose(frozen[num].values, rerun[num].values, atol=1e-8))
          and bool((frozen["significant_FDR"] == rerun["significant_FDR"]).all()))
    validation = {"K6_predictors": {"n_rows_frozen": len(frozen), "n_rows_rerun": len(rerun), "match": ok}}

    labels7 = load_labels(7)
    res7, diag7, llf7, llnull7 = run_predictors(labels7, 7, os.path.join(OUT, "k7_predictors"))
    print(f"Validation K=6 predictors: {validation['K6_predictors']}")
    print(f"K=7: n_tests={len(res7)}, FDR significant={int(res7['significant_FDR'].sum())}, "
          f"McFadden R2={1 - llf7 / llnull7:.4f}, converged={bool(diag7.set_index('metric').loc['converged', 'value'])}")
    print(res7[res7.significant_FDR][["profile", "predictor", "beta", "p_FDR", "OR"]].to_string(index=False))
    return validation


# =====================================================================
# C. Duplicate-row sensitivity for K=7
# =====================================================================
def part_c():
    print("=" * 70)
    print("C. DUPLICATE-ROW SENSITIVITY — K=7 refit on the unique sample")
    print("=" * 70)
    df = pd.read_excel("data.xls", sheet_name=0).copy()
    dup_mask = df.duplicated(keep="first")
    df_u = df.loc[~dup_mask].copy()
    N_u = df_u.shape[0]
    print(f"N_full={len(df)}, duplicates removed={int(dup_mask.sum())}, N_unique={N_u}")

    int_u = df_u[CONSTRUCTS["INT"]].astype(float).mean(axis=1)
    be_u = df_u[CONSTRUCTS["BE"]].astype(float).mean(axis=1)
    r_u, p_u = stats.pearsonr(int_u, be_u)
    z_INT = (int_u - int_u.mean()) / int_u.std(ddof=1)
    z_BE = (be_u - be_u.mean()) / be_u.std(ddof=1)
    X = np.column_stack([z_INT.values, z_BE.values])

    gmm = GaussianMixture(n_components=7, covariance_type="full", n_init=N_INIT,
                          random_state=SEED, max_iter=MAX_ITER, reg_covar=REG_COVAR)
    gmm.fit(X)
    n = X.shape[0]
    k_params = count_params_full(7, X.shape[1])
    LL = float(gmm.score(X) * n)
    BIC = float(-2 * LL + k_params * np.log(n))
    AIC = float(-2 * LL + 2 * k_params)
    proba = gmm.predict_proba(X)
    entropy = safe_entropy(proba)
    labels = gmm.predict(X)
    sizes = np.bincount(labels, minlength=7)
    post_max = proba.max(axis=1)

    # Hungarian match vs primary K=7 profile means (full sample, Phase 04)
    frozen_means = pd.read_csv(os.path.join(LPA, "K_7", "profile_means.csv"))
    cost = np.linalg.norm(
        gmm.means_[:, None, :] - frozen_means[["z_INT", "z_BE"]].values[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    mean_dist = float(np.mean([cost[r, c] for r, c in zip(row_ind, col_ind)]))

    profile_rows = []
    for j in range(7):
        m = labels == j
        profile_rows.append({
            "profile": j, "matched_frozen_profile": mapping.get(j, np.nan),
            "N": int(m.sum()), "percentage": float(m.sum() / N_u * 100),
            "mean_INT": float(int_u[m].mean()), "mean_BE": float(be_u[m].mean()),
            "mean_z_INT": float(gmm.means_[j, 0]), "mean_z_BE": float(gmm.means_[j, 1]),
            "mean_GAP": float((z_INT - z_BE)[m].mean()),
        })
    prof = pd.DataFrame(profile_rows)
    prof.to_csv(os.path.join(OUT, "k7_duplicate_sensitivity_profiles.csv"), index=False)

    gap_u = (z_INT - z_BE).values
    pct_int_gt_u = float((gap_u > 0).mean() * 100)
    pct_be_gt_u = float((gap_u < 0).mean() * 100)

    # "Frozen" reference values are read from frozen result files, not
    # hardcoded — this script must contain no numeric literals for them.
    _dims = pd.read_csv("results/01_data_inspection/data_dimensions.csv")
    N_full_frozen = int(_dims["N_rows"].iloc[0])
    _dups = pd.read_csv("results/01_data_inspection/duplicate_information.csv")
    n_dup_frozen = int(_dups.loc[_dups["check"] == "all_columns", "n_duplicates"].iloc[0])
    _corr = pd.read_csv("results/02_measurement/int_be_correlation.csv")
    r_frozen = float(_corr["r"].iloc[0])
    _gap = pd.read_csv("results/03_gap_analysis/gap_statistics.csv").set_index("statistic")["value"]
    gap_sd_frozen = float(_gap["SD"])
    pct_int_gt_frozen = float(_gap["positive_gap_pct"])
    pct_be_gt_frozen = float(_gap["negative_gap_pct"])
    _fit = pd.read_csv(os.path.join(LPA, "model_fit.csv"))
    _fit7 = _fit[_fit["K"] == 7].iloc[0]
    LL_frozen = float(_fit7["log_likelihood"])
    BIC_frozen = float(_fit7["BIC"])
    AIC_frozen = float(_fit7["AIC"])
    entropy_frozen = float(_fit7["entropy"])
    _sizes7 = pd.read_csv(os.path.join(LPA, "K_7", "profile_sizes.csv"))
    sizes_full_frozen = str([int(v) for v in _sizes7["size"].tolist()])
    min_class_frozen = int(_sizes7["size"].min())
    _post7 = pd.read_csv(os.path.join(LPA, "K_7", "posterior_probabilities.csv"))
    _post_cols7 = [c for c in _post7.columns if c.startswith("post_profile_")]
    mean_max_post_frozen = float(_post7[_post_cols7].max(axis=1).mean())
    converged_frozen = bool(_fit7["converged"])

    out_rows = [
        {"metric": "N_full_sample", "frozen": N_full_frozen, "unique_sample": N_full_frozen, "note": "constant"},
        {"metric": "n_duplicates_removed", "frozen": n_dup_frozen, "unique_sample": int(dup_mask.sum()), "note": "constant"},
        {"metric": "N_unique_sample", "frozen": N_full_frozen - n_dup_frozen, "unique_sample": N_u, "note": "constant"},
        {"metric": "pearson_r_INT_BE", "frozen": r_frozen, "unique_sample": r_u, "note": "sensitivity check"},
        {"metric": "gap_SD", "frozen": gap_sd_frozen, "unique_sample": float((z_INT - z_BE).std(ddof=1)), "note": "sensitivity check"},
        {"metric": "pct_INT_gt_BE", "frozen": pct_int_gt_frozen, "unique_sample": pct_int_gt_u, "note": "sample-level, K-independent"},
        {"metric": "pct_BE_gt_INT", "frozen": pct_be_gt_frozen, "unique_sample": pct_be_gt_u, "note": "sample-level, K-independent"},
        {"metric": "K7_log_likelihood_full_sample", "frozen": LL_frozen, "unique_sample": np.nan, "note": "reference"},
        {"metric": "K7_log_likelihood", "frozen": LL_frozen, "unique_sample": LL,
         "note": f"refit on N={N_u}"},
        {"metric": "K7_BIC", "frozen": BIC_frozen, "unique_sample": BIC,
         "note": f"refit on N={N_u}"},
        {"metric": "K7_AIC", "frozen": AIC_frozen, "unique_sample": AIC,
         "note": f"refit on N={N_u}"},
        {"metric": "K7_entropy", "frozen": entropy_frozen, "unique_sample": entropy,
         "note": f"refit on N={N_u}"},
        {"metric": "K7_class_sizes_full_sample", "frozen": sizes_full_frozen,
         "unique_sample": str(sizes.tolist()), "note": f"refit on N={N_u}"},
        {"metric": "K7_min_class_N", "frozen": min_class_frozen, "unique_sample": int(sizes.min()),
         "note": f"refit on N={N_u}"},
        {"metric": "K7_mean_max_posterior", "frozen": mean_max_post_frozen, "unique_sample": float(post_max.mean()),
         "note": f"refit on N={N_u}"},
        {"metric": "K7_converged", "frozen": converged_frozen, "unique_sample": bool(gmm.converged_), "note": "refit"},
        {"metric": "K7_hungarian_mean_distance", "frozen": 0.0, "unique_sample": mean_dist,
         "note": "z-space distance, refit vs frozen K=7"},
    ]
    out = pd.DataFrame(out_rows)
    out.to_csv(os.path.join(OUT, "k7_duplicate_sensitivity.csv"), index=False)

    print(out.to_string(index=False))
    print("\nHungarian mapping (refit profile -> frozen profile):", mapping)
    print(prof.to_string(index=False))
    return out


def main():
    ensure_dir(OUT)
    np.random.seed(SEED)
    scores = pd.read_csv(f"{PF}/02_measurement/construct_scores.csv", index_col="respondent")
    val_a = part_a(scores)
    val_b = part_b(scores)
    out_c = part_c()

    report = {
        "validation": {**val_a, **val_b},
        "outputs": [
            "gap_variance_decomposition_K2_K7.csv", "k7_per_profile_gap.csv",
            "k7_predictors/multinomial_results.csv", "k7_predictors/model_diagnostics.csv",
            "k7_predictors/predictor_summary.csv", "k7_duplicate_sensitivity.csv",
            "k7_duplicate_sensitivity_profiles.csv",
        ],
    }
    import json
    with open(os.path.join(OUT, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nAll K=7-primary outputs written to results/k7_primary/")


if __name__ == "__main__":
    main()
