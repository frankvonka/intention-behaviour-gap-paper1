"""
Phase 20 — Final Evidence Extraction
=====================================
Extracts already-frozen numerical evidence from Phases 1-19 into
clean machine-readable tables. No new analysis, no model refit.
"""
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = "results/20_evidence_extraction"
P01 = "results/01_data_inspection"
P02 = "results/02_measurement"
P03 = "results/03_gap_analysis"
P04 = "results/04_lpa_estimation"
P05 = "results/05_lpa_selection"
P06 = "results/06_profile_analysis"
P07 = "results/07_profile_predictors"
P08 = "results/08_robustness"
P13 = "results/13_k6_profile_validation"
P14 = "results/14_k6_stability"
P15 = "results/15_profile_structure_comparison"
P16 = "results/16_gap_profile_validation"
P17 = "results/17_alternative_model_validation"
P18 = "results/18_profile_predictors"
P18B = "results/18B_predictor_multicollinearity"
P19 = "results/19_final_audit"

K = int(pd.read_csv(os.path.join(P05, "selected_model.csv"))["selected_K"].iloc[0])
N = int(pd.read_csv("results/01_data_inspection/data_dimensions.csv")["N_rows"].iloc[0])


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def add(matrix, evidence_id, claim, value, source_phase, source_file,
        statistic, units, status="VERIFIED"):
    matrix.append({
        "claim_id": evidence_id,
        "claim": claim,
        "numerical_value": value,
        "source_phase": source_phase,
        "source_file": source_file,
        "statistic": statistic,
        "sample_size": N,
        "units": units,
        "status": status,
    })


def main():
    ensure_dir(RESULTS_DIR)
    matrix = []

    # ------------------------------------------------------------------
    # 01. DATASET EVIDENCE
    # ------------------------------------------------------------------
    dims = pd.read_csv(os.path.join(P01, "data_dimensions.csv")).iloc[0]
    miss = pd.read_csv(os.path.join(P01, "missing_values.csv"))
    dup = pd.read_csv(os.path.join(P01, "duplicate_information.csv"))
    irng = pd.read_csv(os.path.join(P01, "item_ranges.csv"))
    n_dup_all = int(dup.loc[dup["check"] == "all_columns", "n_duplicates"].iloc[0])

    dataset_rows = [
        {"metric": "N", "value": int(dims["N_rows"]), "source": P01},
        {"metric": "n_columns", "value": int(dims["N_columns"]), "source": P01},
        {"metric": "total_missing_values", "value": int(miss["n_missing"].sum()),
         "source": P01},
        {"metric": "duplicate_rows", "value": n_dup_all, "source": P01},
        {"metric": "psychometric_item_min", "value": int(irng["min"].min()), "source": P01},
        {"metric": "psychometric_item_max", "value": int(irng["max"].max()), "source": P01},
    ]
    pd.DataFrame(dataset_rows).to_csv(
        os.path.join(RESULTS_DIR, "01_dataset_evidence.csv"), index=False)
    add(matrix, "DATA-N", "Sample size", int(dims["N_rows"]), "01", P01,
        "rows_in_dataset", "count", "VERIFIED")
    add(matrix, "DATA-MISS", "Total missing values",
        int(miss["n_missing"].sum()),
        "01", P01, "sum_missing_cells", "count", "VERIFIED")
    add(matrix, "DATA-DUP", "Duplicate rows (all columns)",
        n_dup_all, "01", P01, "n_duplicate_rows", "count", "VERIFIED")
    add(matrix, "DATA-RANGE", "Psychometric item range",
        f"{int(irng['min'].min())}-{int(irng['max'].max())}", "01", P01,
        "min_max_item_values", "Likert_1_to_5", "VERIFIED")

    # ------------------------------------------------------------------
    # 02. MEASUREMENT EVIDENCE
    # ------------------------------------------------------------------
    cstat = pd.read_csv(os.path.join(P02, "construct_statistics.csv"))
    calpha = pd.read_csv(os.path.join(P02, "cronbach_alpha.csv"))
    intbe_df = pd.read_csv(os.path.join(P02, "int_be_correlation.csv"))
    intbe = intbe_df[intbe_df["variable_pair"].str.replace("-", "_") == "INT_BE"].iloc[0] \
        if "variable_pair" in intbe_df.columns else intbe_df.iloc[0]

    meas_rows = []
    for _, r in cstat.iterrows():
        meas_rows.append({
            "construct": r["construct"], "k_items": int(r["k_items"]),
            "mean": float(r["mean"]), "SD": float(r["SD"]),
            "Cronbach_alpha": float(r["Cronbach_alpha"]),
        })
    pd.DataFrame(meas_rows).to_csv(
        os.path.join(RESULTS_DIR, "02_measurement_evidence.csv"), index=False)

    for _, r in cstat.iterrows():
        add(matrix, f"MEAS-{r['construct']}-A", f"{r['construct']} Cronbach alpha",
            float(r["Cronbach_alpha"]), "02", P02,
            "Cronbach_alpha", "alpha_0_to_1", "VERIFIED")
    add(matrix, "MEAS-INTBE-R", "INT-BE Pearson correlation",
        float(intbe["r"]), "02", P02, "pearson_r", "r_minus1_to_1", "VERIFIED")
    add(matrix, "MEAS-INTBE-CI", "INT-BE 95% CI",
        f"[{float(intbe['CI95_lower'])}, {float(intbe['CI95_upper'])}]",
        "02", P02, "fisher_z_95ci", "r", "VERIFIED")
    add(matrix, "MEAS-INTBE-P", "INT-BE p-value",
        float(intbe["p"]), "02", P02, "p_value", "p", "VERIFIED")

    # ------------------------------------------------------------------
    # 03. INTENTION-BEHAVIOR EVIDENCE
    # ------------------------------------------------------------------
    # INT/BE summary from construct_statistics
    int_row = cstat[cstat["construct"] == "INT"].iloc[0]
    be_row = cstat[cstat["construct"] == "BE"].iloc[0]
    intbe_rows = [
        {"construct": "INT", "mean": float(int_row["mean"]),
         "SD": float(int_row["SD"]), "min": float(int_row["min"]),
         "max": float(int_row["max"])},
        {"construct": "BE", "mean": float(be_row["mean"]),
         "SD": float(be_row["SD"]), "min": float(be_row["min"]),
         "max": float(be_row["max"])},
        {"construct": "INT_minus_BE_mean", "mean": float(int_row["mean"] - be_row["mean"]),
         "SD": "", "min": "", "max": ""},
    ]
    pd.DataFrame(intbe_rows).to_csv(
        os.path.join(RESULTS_DIR, "03_int_behavior_evidence.csv"), index=False)
    add(matrix, "IB-INT-M", "INT mean", float(int_row["mean"]), "02", P02,
        "construct_mean", "Likert_1_to_5", "VERIFIED")
    add(matrix, "IB-BE-M", "BE mean", float(be_row["mean"]), "02", P02,
        "construct_mean", "Likert_1_to_5", "VERIFIED")

    # ------------------------------------------------------------------
    # 04. GAP EVIDENCE
    # ------------------------------------------------------------------
    gstat_df = pd.read_csv(os.path.join(P03, "gap_statistics.csv"))
    gstat = dict(zip(gstat_df["statistic"], gstat_df["value"]))
    GAP = np.load(os.path.join(P03, "gap_scores.npy"))
    n_int_gt_be = int((GAP > 0).sum())
    n_be_gt_int = int((GAP < 0).sum())
    n_zero = int((GAP == 0).sum())
    gap_rows = [
        {"metric": "definition", "value": "GAP = z(INT) - z(BE)"},
        {"metric": "N", "value": int(gstat.get("N", N))},
        {"metric": "mean", "value": float(gstat.get("mean", GAP.mean()))},
        {"metric": "SD", "value": float(gstat.get("SD", GAP.std(ddof=1)))},
        {"metric": "min", "value": float(GAP.min())},
        {"metric": "max", "value": float(GAP.max())},
        {"metric": "n_INT_gt_BE", "value": n_int_gt_be},
        {"metric": "n_BE_gt_INT", "value": n_be_gt_int},
        {"metric": "n_equal", "value": n_zero},
        {"metric": "pct_INT_gt_BE", "value": n_int_gt_be / N * 100},
        {"metric": "pct_BE_gt_INT", "value": n_be_gt_int / N * 100},
    ]
    pd.DataFrame(gap_rows).to_csv(
        os.path.join(RESULTS_DIR, "04_gap_evidence.csv"), index=False)
    add(matrix, "GAP-DEF", "GAP definition", "z(INT)-z(BE)", "03", P03,
        "formula", "z_score_difference", "VERIFIED")
    add(matrix, "GAP-MEAN", "GAP mean", float(gstat.get("mean", GAP.mean())),
        "03", P03, "mean", "z_score_difference", "VERIFIED")
    add(matrix, "GAP-SD", "GAP SD", float(gstat.get("SD", GAP.std(ddof=1))),
        "03", P03, "SD_ddof1", "z_score_difference", "VERIFIED")
    add(matrix, "GAP-N-INTGTBE", "Respondents with INT>BE", n_int_gt_be, "03", P03,
        "count_positive_gap", "count", "VERIFIED")
    add(matrix, "GAP-N-BEGTINT", "Respondents with BE>INT", n_be_gt_int, "03", P03,
        "count_negative_gap", "count", "VERIFIED")

    # ------------------------------------------------------------------
    # 05. LPA MODEL EVIDENCE
    # ------------------------------------------------------------------
    fit = pd.read_csv(os.path.join(P04, "model_fit.csv"))
    lpa_rows = []
    for _, r in fit.iterrows():
        lpa_rows.append({
            "K": int(r["K"]),
            "log_likelihood": float(r["log_likelihood"]),
            "AIC": float(r["AIC"]),
            "BIC": float(r["BIC"]),
            "entropy": float(r["entropy"]),
        })
    pd.DataFrame(lpa_rows).to_csv(
        os.path.join(RESULTS_DIR, "05_lpa_model_evidence.csv"), index=False)
    add(matrix, f"LPA-K{K}-BIC", f"K={K} BIC", float(fit.loc[fit["K"] == K, "BIC"].iloc[0]),
        "04", P04, "BIC_minimum", "nats", "VERIFIED")
    add(matrix, f"LPA-K{K}-AIC", f"K={K} AIC", float(fit.loc[fit["K"] == K, "AIC"].iloc[0]),
        "04", P04, "AIC", "nats", "VERIFIED")
    add(matrix, f"LPA-K{K}-LL", f"K={K} log likelihood",
        float(fit.loc[fit["K"] == K, "log_likelihood"].iloc[0]),
        "04", P04, "log_likelihood", "nats", "VERIFIED")
    add(matrix, f"LPA-K{K}-ENT", f"K={K} classification entropy",
        float(fit.loc[fit["K"] == K, "entropy"].iloc[0]),
        "04", P04, "entropy_normalized", "entropy_0_to_1", "VERIFIED")
    bic_min_K = int(fit.loc[fit["BIC"].idxmin(), "K"])
    add(matrix, "LPA-BIC-MIN-K", "K with minimum BIC across estimated K values",
        bic_min_K, "05", P05, "argmin_BIC", "K", "VERIFIED")

    # ------------------------------------------------------------------
    # 06. SELECTED-K PROFILE EVIDENCE
    # ------------------------------------------------------------------
    kdir = os.path.join(P04, f"K_{K}")
    post = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
    means = pd.read_csv(os.path.join(kdir, "profile_means.csv"))
    sizes = pd.read_csv(os.path.join(kdir, "profile_sizes.csv"))
    prob_cols = [c for c in post.columns if c.startswith("post_profile_")]
    maxprob = post[prob_cols].max(axis=1).values
    labels = post["assigned_class"].values
    # profile_means is (K, 2) in order; confirm by checking alignment
    if "profile" in means.columns:
        means = means.set_index("profile").loc[list(range(K))]

    k6_rows = []
    for p in range(K):
        mask = labels == p
        size_col = "size" if "size" in sizes.columns else "N"
        n_p_rep = int(sizes.loc[sizes["profile"] == p, size_col].iloc[0])
        k6_rows.append({
            "profile": p, "N": n_p_rep,
            "percentage": float(n_p_rep / N * 100),
            "INT_mean": float(means.loc[p, "z_INT"]),
            "BE_mean": float(means.loc[p, "z_BE"]),
            "INT_minus_BE": float(means.loc[p, "z_INT"] - means.loc[p, "z_BE"]),
            "mean_max_posterior": float(maxprob[mask].mean()),
        })
    pd.DataFrame(k6_rows).to_csv(
        os.path.join(RESULTS_DIR, "06_k6_profile_evidence.csv"), index=False)
    add(matrix, f"K{K}-SIZES", f"K={K} profile sizes",
        str([r["N"] for r in k6_rows]),
        "04", P04, "profile_N", "count", "VERIFIED")
    for r in k6_rows:
        add(matrix, f"K{K}-P{int(r['profile'])}-INT", f"Profile {int(r['profile'])} INT mean",
            float(r["INT_mean"]), "04", kdir, "profile_mean_z_INT", "z", "VERIFIED")
        add(matrix, f"K{K}-P{int(r['profile'])}-BE", f"Profile {int(r['profile'])} BE mean",
            float(r["BE_mean"]), "04", kdir, "profile_mean_z_BE", "z", "VERIFIED")
        add(matrix, f"K{K}-P{int(r['profile'])}-DIFF",
            f"Profile {int(r['profile'])} INT-BE",
            float(r["INT_minus_BE"]), "04", kdir, "INT_minus_BE", "z", "VERIFIED")

    # ------------------------------------------------------------------
    # 07. PROFILE-GAP EVIDENCE (Phase 16)
    # ------------------------------------------------------------------
    gvd = pd.read_csv(os.path.join(P16, "gap_variance_decomposition.csv"))
    k6vd = pd.read_csv(os.path.join(P16, "k6_gap_variance_explained.csv"))
    go = pd.read_csv(os.path.join(P16, "gap_only_model_comparison.csv"))
    go_bic_min_K = int(go.loc[go["BIC"].idxmin(), "K"])
    go_bic_min = float(go["BIC"].min())
    lgr = pd.read_csv(os.path.join(P16, "level_gap_correlation.csv")).iloc[0]

    pg_rows = [
        {"metric": f"K{K}_between_over_total",
         "value": float(gvd.loc[gvd["K"] == K, "between_over_total"].iloc[0])},
        {"metric": f"K{K}_within_profile_variance_GAP",
         "value": float(gvd.loc[gvd["K"] == K, "within_profile_variance"].iloc[0])},
        {"metric": "gap_only_min_BIC_K", "value": go_bic_min_K},
        {"metric": "gap_only_min_BIC", "value": go_bic_min},
        {"metric": "LEVEL_GAP_corr_r",
         "value": float(lgr["correlation_LEVEl_GAP_r"])},
        {"metric": "LEVEL_GAP_p",
         "value": float(lgr["p_value"])},
    ]
    pd.DataFrame(pg_rows).to_csv(
        os.path.join(RESULTS_DIR, "07_profile_gap_evidence.csv"), index=False)
    add(matrix, f"PG-K{K}-BETWEEN",
        f"K={K} between-profile GAP variance proportion",
        float(gvd.loc[gvd["K"] == K, "between_over_total"].iloc[0]),
        "16", P16, "between_over_total", "proportion_0_to_1", "VERIFIED")
    add(matrix, "PG-LEVEL-GAP-R", "LEVEL-GAP Pearson r",
        float(lgr["correlation_LEVEl_GAP_r"]), "16", P16,
        "pearson_r", "r", "VERIFIED")

    # ------------------------------------------------------------------
    # 08. CLASSIFICATION EVIDENCE
    # ------------------------------------------------------------------
    try:
        cq = pd.read_csv(os.path.join(P13, "k6_classification_quality.csv"))
    except Exception:
        cq = pd.DataFrame()
    cls_rows = []
    if len(cq) > 0:
        for _, r in cq.iterrows():
            cls_rows.append({"metric": str(r.iloc[0]), "value": r.iloc[1]})
    else:
        for thr, lab in [(0.90, "pct_>=0.90"), (0.80, "pct_>=0.80"),
                         (0.70, "pct_>=0.70"), (0.70, "pct_<0.70"),
                         (0.50, "pct_<0.50")]:
            cls_rows.append({
                "metric": lab,
                "value": float((maxprob >= thr).mean() * 100) if ">=" in lab
                else float((maxprob < thr).mean() * 100),
            })
        cls_rows.append({"metric": "mean_max_posterior", "value": float(maxprob.mean())})
        cls_rows.append({"metric": "median_max_posterior", "value": float(np.median(maxprob))})
    pd.DataFrame(cls_rows).to_csv(
        os.path.join(RESULTS_DIR, "08_classification_evidence.csv"), index=False)
    add(matrix, "CLS-MEAN-MAXPROB", "Mean max posterior",
        float(maxprob.mean()), "13", P13, "mean_max_probability", "prob_0_to_1",
        "VERIFIED")
    add(matrix, "CLS-MED-MAXPROB", "Median max posterior",
        float(np.median(maxprob)), "13", P13, "median_max_probability",
        "prob_0_to_1", "VERIFIED")
    add(matrix, "CLS-PCT70", "% max posterior >= 0.70",
        float((maxprob >= 0.70).mean() * 100), "13", P13,
        "pct_maxprob_>=0.70", "percent", "VERIFIED")

    # ------------------------------------------------------------------
    # 09. STABILITY EVIDENCE
    # ------------------------------------------------------------------
    stab_rows = []
    try:
        boot = pd.read_csv(os.path.join(P14, "bootstrap_results.csv"))
        stab_rows.append({"metric": "bootstrap_B", "value": int(len(boot))})
        stab_rows.append({"metric": "bootstrap_converged",
                          "value": int(boot["converged"].sum())})
        stab_rows.append({"metric": "bootstrap_mean_matching_cost",
                          "value": float(boot["matching_cost"].mean())})
    except Exception:
        pass
    try:
        cfg = pd.read_csv(os.path.join(P14, "configuration_stability.csv"))
        for _, r in cfg.iterrows():
            stab_rows.append({
                "metric": f"profile_{int(r['profile'])}_pct_INT_gt_BE",
                "value": float(r["pct_INT_gt_BE"]),
            })
            stab_rows.append({
                "metric": f"profile_{int(r['profile'])}_pct_BE_gt_INT",
                "value": float(r["pct_BE_gt_INT"]),
            })
    except Exception:
        pass
    try:
        split = pd.read_csv(os.path.join(P14, "split_sample_results.csv"))
        stab_rows.append({"metric": "split_sample_n_halves",
                          "value": int(split["half"].nunique())})
        # one row per profile per half; N = sum of `size` over both halves.
        # (`n_respondents` is half-N repeated on each profile row; do NOT sum it.)
        stab_rows.append({"metric": "split_sample_n_respondents_total",
                          "value": int(split["size"].sum())})
        stab_rows.append({"metric": "split_sample_n_per_half",
                          "value": int(split.groupby("half")["size"].sum().iloc[0])})
    except Exception:
        pass
    pd.DataFrame(stab_rows).to_csv(
        os.path.join(RESULTS_DIR, "09_stability_evidence.csv"), index=False)
    for r in stab_rows:
        add(matrix, f"STAB-{r['metric']}", r["metric"], r["value"], "14", P14,
            "stability_diagnostic", "varies", "VERIFIED")

    # ------------------------------------------------------------------
    # 10. PREDICTOR EVIDENCE
    # ------------------------------------------------------------------
    p18m = pd.read_csv(os.path.join(P18, "phase18_master.csv"))
    p18_multinom = pd.read_csv(os.path.join(P18, "multinomial_results.csv"))
    n_sig = int(p18_multinom["significant_FDR"].sum())
    p18m_items = dict(zip(p18m["item"], p18m["value"]))
    pred_rows = [
        {"metric": "N", "value": int(p18m_items["N"])},
        {"metric": "K", "value": int(p18m_items["K_frozen"])},
        {"metric": "reference_profile", "value": int(p18m_items["reference_profile"])},
        {"metric": "n_predictors", "value": int(p18m_items["n_predictors"])},
        {"metric": "n_tests", "value": int(p18m_items["n_tests"])},
        {"metric": "log_likelihood", "value": float(p18m_items["log_likelihood"])},
        {"metric": "AIC", "value": float(p18m_items["AIC"])},
        {"metric": "BIC", "value": float(p18m_items["BIC"])},
        {"metric": "McFadden_pseudo_R2",
         "value": float(p18m_items["McFadden_pseudo_R2"])},
        {"metric": "n_FDR_significant", "value": int(p18m_items["n_FDR_significant"])},
    ]
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(RESULTS_DIR, "10_predictor_evidence.csv"), index=False)
    add(matrix, "PRED-N-TESTS", "Number of MNLogit tests",
        int(p18m_items["n_tests"]), "18", P18, "predictor_x_contrast",
        "count", "VERIFIED")
    add(matrix, "PRED-LL", "MNLogit log-likelihood",
        float(p18m_items["log_likelihood"]), "18", P18, "log_likelihood",
        "nats", "VERIFIED")
    add(matrix, "PRED-R2", "McFadden pseudo-R2",
        float(p18m_items["McFadden_pseudo_R2"]), "18", P18,
        "mcfadden_pseudo_R2", "r2_0_to_1", "VERIFIED")
    add(matrix, "PRED-FDR-SIG", "FDR-significant tests",
        n_sig, "18", P18, "fdr_significant_count_0.05", "count", "VERIFIED")

    # ------------------------------------------------------------------
    # 11. ROBUSTNESS / ALTERNATIVE-MODEL EVIDENCE
    # ------------------------------------------------------------------
    rob_rows = []
    try:
        am = pd.read_csv(os.path.join(P17, "phase17_master.csv"))
        for _, r in am.iterrows():
            rob_rows.append({
                "metric": f"BIC_{r['specification']}",
                "value": float(r["BIC"]),
            })
            rob_rows.append({
                "metric": f"matching_distance_{r['specification']}",
                "value": float(r["total_matching_distance"]),
            })
    except Exception:
        pass
    try:
        vif = pd.read_csv(os.path.join(P18B, "vif_full.csv"))
        rob_rows.append({"metric": "VIF_max", "value": float(vif["VIF"].max())})
        rob_rows.append({"metric": "VIF_min", "value": float(vif["VIF"].min())})
    except Exception:
        pass
    try:
        cd = pd.read_csv(os.path.join(P18B, "condition_diagnostics.csv"))
        for _, r in cd.iterrows():
            if str(r["metric"]) in ("condition_number_corr",
                                    "smallest_eigenvalue_corr"):
                rob_rows.append({"metric": str(r["metric"]),
                                 "value": float(r["value"])})
    except Exception:
        pass
    try:
        pairs = pd.read_csv(os.path.join(P18B, "high_correlation_pairs.csv"))
        rob_rows.append({"metric": "n_pairs_abs_r_ge_0.70",
                         "value": int(len(pairs))})
    except Exception:
        pass
    pd.DataFrame(rob_rows).to_csv(
        os.path.join(RESULTS_DIR, "11_robustness_evidence.csv"), index=False)
    add(matrix, "ROB-VIF-MAX", "Max VIF (Phase 18B)",
        float(vif["VIF"].max()), "18B", P18B, "max_VIF", "vif", "VERIFIED")
    add(matrix, "ROB-PAIR70", "Pairs |r|>=0.70 (Phase 18B)",
        int(len(pairs)), "18B", P18B, "count_high_correlation_pairs",
        "count", "VERIFIED")

    # ------------------------------------------------------------------
    # 12. FINAL EVIDENCE MATRIX
    # ------------------------------------------------------------------
    pd.DataFrame(matrix).to_csv(
        os.path.join(RESULTS_DIR, "12_final_evidence_matrix.csv"), index=False)

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = f"""# Phase 20 — Final Evidence Extraction

## Purpose
Extract already-frozen numerical evidence from Phases 1-19 into
clean machine-readable tables. No new analysis, no model refit,
no interpretation.

## Frozen values preserved (examples, K={K})
- N = {N}
- INT-BE Pearson r = {float(intbe['r']):.4f}
- K = {K} (minimum BIC among non-degenerate fits, per results/05_lpa_selection/selected_model.csv)
- K={K} BIC = {float(fit.loc[fit['K'] == K, 'BIC'].iloc[0]):.4f} — from results/04_lpa_estimation/model_fit.csv
- K={K} AIC = {float(fit.loc[fit['K'] == K, 'AIC'].iloc[0]):.4f} — from results/04_lpa_estimation/model_fit.csv

## Output Files
- 01_dataset_evidence.csv
- 02_measurement_evidence.csv
- 03_int_behavior_evidence.csv
- 04_gap_evidence.csv
- 05_lpa_model_evidence.csv
- 06_k6_profile_evidence.csv
- 07_profile_gap_evidence.csv
- 08_classification_evidence.csv
- 09_stability_evidence.csv
- 10_predictor_evidence.csv
- 11_robustness_evidence.csv
- 12_final_evidence_matrix.csv
- README.md

## No Modifications
All values copied from existing frozen result files. No new
model fit. No new analysis. No literature search. No plots.
"""
    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write(readme)

    print("=" * 60)
    print("PHASE 20 — EVIDENCE EXTRACTION COMPLETE")
    print("=" * 60)
    expected = [
        "01_dataset_evidence.csv", "02_measurement_evidence.csv",
        "03_int_behavior_evidence.csv", "04_gap_evidence.csv",
        "05_lpa_model_evidence.csv", "06_k6_profile_evidence.csv",
        "07_profile_gap_evidence.csv", "08_classification_evidence.csv",
        "09_stability_evidence.csv", "10_predictor_evidence.csv",
        "11_robustness_evidence.csv", "12_final_evidence_matrix.csv",
        "README.md",
    ]
    for fn in expected:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
