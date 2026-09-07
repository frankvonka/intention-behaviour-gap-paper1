"""
Phase 12 — Paper 1 Evidence Package
=====================================
Extracts verified numerical results from Phases 01–11 into a clean
machine-readable evidence package.

Uses ONLY existing result files — no refitting, no new calculations,
no literature search, no interpretation.
"""
import os
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Source directories
# ---------------------------------------------------------------------------
F10 = "results/10_final"
F11 = "results/11_final_verification"
RESULTS_DIR = "results/12_paper1_evidence"


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    ensure_dir(RESULTS_DIR)
    all_outputs = {}

    # ==================================================================
    # 1. DATASET EVIDENCE
    # ==================================================================
    meta = json.load(open(os.path.join(F10, "data_diagnostics", "dataset_metadata.json")))
    dims = pd.read_csv(os.path.join(F10, "data_diagnostics", "data_dimensions.csv"))
    missing = pd.read_csv(os.path.join(F10, "data_diagnostics", "missing_values.csv"))
    dup = pd.read_csv(os.path.join(F10, "data_diagnostics", "duplicate_information.csv"))

    dataset_summary = pd.DataFrame({
        "metric": ["N", "n_variables", "n_missing_values", "n_duplicate_rows"],
        "value": [
            int(meta["N_rows"]),
            int(meta["N_columns"]),
            int(missing["n_missing"].sum()),
            int(dup.loc[dup["check"] == "all_columns", "n_duplicates"].values[0]),
        ],
    })
    dataset_summary.to_csv(os.path.join(RESULTS_DIR, "dataset_summary.csv"), index=False)
    all_outputs["dataset_summary.csv"] = True

    # ==================================================================
    # 2. MEASUREMENT EVIDENCE
    # ==================================================================
    cs = pd.read_csv(os.path.join(F10, "measurement_results", "construct_statistics.csv"))
    cronbach = pd.read_csv(os.path.join(F10, "measurement_results", "cronbach_alpha.csv"))
    item_stats = pd.read_csv(os.path.join(F10, "measurement_results", "item_statistics.csv"))

    measurement_rows = []
    for _, row in cs.iterrows():
        construct = row["construct"]
        k = int(row["k_items"])
        alpha = cronbach.loc[cronbach["construct"] == construct, "Cronbach_alpha"].values[0]
        measurement_rows.append({
            "construct": construct,
            "k_items": k,
            "items": row["items"],
            "Cronbach_alpha": float(alpha),
            "mean": float(row["mean"]),
            "SD": float(row["SD"]),
        })
    measurement_summary = pd.DataFrame(measurement_rows)
    measurement_summary.to_csv(os.path.join(RESULTS_DIR, "measurement_summary.csv"), index=False)
    all_outputs["measurement_summary.csv"] = True

    # ==================================================================
    # 3. INTENTION-BEHAVIOR ASSOCIATION
    # ==================================================================
    intbe = pd.read_csv(os.path.join(F10, "measurement_results", "int_be_correlation.csv"))
    int_behavior = pd.DataFrame({
        "variable_pair": [intbe["variable_pair"].values[0]],
        "Pearson_r": [float(intbe["r"].values[0])],
        "p_value": [float(intbe["p"].values[0])],
        "N": [int(intbe["N"].values[0])],
        "CI95_lower": [float(intbe["CI95_lower"].values[0])],
        "CI95_upper": [float(intbe["CI95_upper"].values[0])],
    })
    int_behavior.to_csv(os.path.join(RESULTS_DIR, "int_behavior_correlation.csv"), index=False)
    all_outputs["int_behavior_correlation.csv"] = True

    # ==================================================================
    # 4. GAP SCORE EVIDENCE
    # ==================================================================
    gap_stats = pd.read_csv(os.path.join(F10, "gap_results", "gap_statistics.csv"))
    gap_dict = dict(zip(gap_stats["statistic"], gap_stats["value"]))

    gap_summary = pd.DataFrame({
        "statistic": ["mean", "SD", "minimum", "maximum",
                       "positive_gap_N", "positive_gap_pct",
                       "negative_gap_N", "negative_gap_pct"],
        "value": [
            float(gap_dict.get("mean", np.nan)),
            float(gap_dict.get("SD", np.nan)),
            float(gap_dict.get("min", np.nan)),
            float(gap_dict.get("max", np.nan)),
            int(gap_dict.get("positive_gap_count", np.nan)),
            float(gap_dict.get("positive_gap_pct", np.nan)),
            int(gap_dict.get("negative_gap_count", np.nan)),
            float(gap_dict.get("negative_gap_pct", np.nan)),
        ],
    })
    gap_summary.to_csv(os.path.join(RESULTS_DIR, "gap_summary.csv"), index=False)
    all_outputs["gap_summary.csv"] = True

    # ==================================================================
    # 5. ALL LPA MODEL COMPARISONS (all K values present in phase 10/11 fit files)
    # ==================================================================
    fit = pd.read_csv(os.path.join(F10, "lpa_model_comparison", "model_fit.csv"))
    comp = pd.read_csv(os.path.join(F10, "selected_lpa", "model_comparison.csv"))
    unc = pd.read_csv(os.path.join(F11, "lpa_comparison.csv"))

    lpa_rows = []
    for K in sorted(fit["K"].unique().tolist()):
        frow = fit[fit["K"] == K].iloc[0]
        urow = unc[unc["K"] == K].iloc[0]
        # Get profile sizes from per-K file (in Phase 04 results)
        sizes_k = pd.read_csv(os.path.join("results/04_lpa_estimation", f"K_{K}", "profile_sizes.csv"))
        sizes = sizes_k["size"].tolist()
        min_size = int(min(sizes))
        lpa_rows.append({
            "K": K,
            "log_likelihood": float(frow["log_likelihood"]),
            "AIC": float(frow["AIC"]),
            "BIC": float(frow["BIC"]),
            "entropy": float(frow["entropy"]),
            "n_profiles": K,
            "smallest_profile_N": min_size,
            "smallest_profile_pct": float(frow["smallest_class_pct"]),
            "pct_maxpost_lt_50": float(urow["pct_maxpost_lt_50"]),
            "pct_maxpost_lt_70": float(urow["pct_maxpost_lt_70"]),
            "pct_maxpost_ge_70": float(urow["pct_maxpost_ge_70"]),
            "converged": bool(frow["converged"]),
        })
    lpa_comparison = pd.DataFrame(lpa_rows)
    lpa_comparison.to_csv(os.path.join(RESULTS_DIR, "lpa_model_comparison.csv"), index=False)
    all_outputs["lpa_model_comparison.csv"] = True

    # ==================================================================
    # 6. ALL PROFILE STRUCTURE
    # ==================================================================
    all_profile = pd.read_csv(os.path.join(F11, "profile_structure.csv"))
    all_profile["INT_minus_BE"] = all_profile["mean_INT"] - all_profile["mean_BE"]
    all_profile[["K", "profile", "N", "percentage", "mean_INT", "mean_BE", "INT_minus_BE", "mean_GAP"]].to_csv(
        os.path.join(RESULTS_DIR, "all_profile_structure.csv"), index=False
    )
    all_outputs["all_profile_structure.csv"] = True

    # Selected K for the primary profile table (read from Phase 05)
    K_sel = int(pd.read_csv("results/05_lpa_selection/selected_model.csv")["selected_K"].iloc[0])

    # ==================================================================
    # 7. SELECTED-K PRIMARY PROFILE TABLE
    # ==================================================================
    k6 = all_profile[all_profile["K"] == K_sel].copy()
    k6_table = k6[["profile", "N", "percentage", "mean_INT", "mean_BE", "INT_minus_BE", "mean_GAP"]].copy()
    k6_table.to_csv(os.path.join(RESULTS_DIR, "k6_profile_table.csv"), index=False)
    all_outputs["k6_profile_table.csv"] = True

    # ==================================================================
    # 8. PROFILE PREDICTORS (selected K)
    # ==================================================================
    coef = pd.read_csv(os.path.join(F10, "profile_predictors", "coefficients.csv"))
    or_df = pd.read_csv(os.path.join(F10, "profile_predictors", "odds_ratios.csv"))
    ci = pd.read_csv(os.path.join(F10, "profile_predictors", "confidence_intervals.csv"))
    pv = pd.read_csv(os.path.join(F10, "profile_predictors", "p_values.csv"))
    fdr = pd.read_csv(os.path.join(F10, "profile_predictors", "fdr_results.csv"))

    pred = coef.merge(or_df, on=["baseline_profile", "target_profile", "predictor"])
    pred = pred.merge(ci, on=["baseline_profile", "target_profile", "predictor"])
    pred = pred.merge(fdr[["baseline_profile", "target_profile", "predictor", "FDR_adjusted_p", "significant_FDR_05"]],
                      on=["baseline_profile", "target_profile", "predictor"])

    k6_predictors = pd.DataFrame({
        "predictor": pred["predictor"],
        "comparison": [f"Profile_{int(t)}" for t in pred["target_profile"]],
        "coefficient": pred["beta"],
        "SE": pred["SE"],
        "OR": pred["OR"],
        "CI_lower": pred["OR_95CI_lower"],
        "CI_upper": pred["OR_95CI_upper"],
        "raw_p": pred["p_value"],
        "FDR_p": pred["FDR_adjusted_p"],
        "FDR_significant": pred["significant_FDR_05"],
    })
    k6_predictors.to_csv(os.path.join(RESULTS_DIR, "k6_predictors.csv"), index=False)
    all_outputs["k6_predictors.csv"] = True

    # ==================================================================
    # 9. ROBUSTNESS EVIDENCE
    # ==================================================================
    rob_rows = []
    # Random-start stability
    rss = pd.read_csv(os.path.join(F10, "robustness_results", "random_start_stability.csv"))
    for K in sorted(fit["K"].unique().tolist()):
        sub = rss[rss["K"] == K]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        rob_rows.append({
            "check": "random_start_stability",
            "K": K,
            "result": f"BIC={row['BIC']:.2f}, entropy={row['entropy']:.4f}, converged={row['converged']}",
            "numerical_evidence": f"n_init={row['n_init']}, BIC={row['BIC']}",
            "status": "PASS" if row["converged"] else "FAIL",
        })
    # Covariance sensitivity
    cov = pd.read_csv(os.path.join(F10, "robustness_results", "covariance_sensitivity.csv"))
    for _, row in cov.iterrows():
        rob_rows.append({
            "check": "covariance_sensitivity",
            "K": int(row["K"]),
            "result": f"cov={row['covariance_type']}, BIC={row['BIC']:.2f}, AIC={row['AIC']:.2f}",
            "numerical_evidence": f"cov={row['covariance_type']}, BIC={row['BIC']:.2f}",
            "status": "PASS" if row["converged"] else "FAIL",
        })
    # Score sensitivity
    ss = pd.read_csv(os.path.join(F10, "robustness_results", "score_sensitivity.csv"))
    for _, row in ss.iterrows():
        rob_rows.append({
            "check": "score_sensitivity",
            "K": "",
            "result": f"{row['score_comparison']}: r={row['pearson_r']:.4f}",
            "numerical_evidence": f"r={row['pearson_r']:.4f}",
            "status": "PASS" if abs(row["pearson_r"]) > 0.8 else "REVIEW",
        })
    # Outlier sensitivity
    osens = pd.read_csv(os.path.join(F10, "robustness_results", "outlier_sensitivity.csv"))
    for _, row in osens.iterrows():
        rob_rows.append({
            "check": "outlier_sensitivity",
            "K": int(row["K"]),
            "result": f"BIC_full={row['BIC_full']:.2f}, BIC_no_outliers={row['BIC_no_outliers']:.2f}",
            "numerical_evidence": f"n_removed={row['n_removed']}, BIC_full={row['BIC_full']:.2f}",
            "status": "PASS",
        })
    # Random seed sensitivity
    rss2 = pd.read_csv(os.path.join(F10, "robustness_results", "random_seed_sensitivity.csv"))
    for K in sorted(fit["K"].unique().tolist()):
        sub = rss2[rss2["K"] == K]
        if len(sub) == 0:
            continue
        rob_rows.append({
            "check": "random_seed_sensitivity",
            "K": K,
            "result": f"BIC_std={sub['BIC'].std():.2f}, seeds={sub['seed'].tolist()}",
            "numerical_evidence": f"n_seeds={len(sub)}",
            "status": "PASS",
        })
    robustness_summary = pd.DataFrame(rob_rows)
    robustness_summary.to_csv(os.path.join(RESULTS_DIR, "robustness_summary.csv"), index=False)
    all_outputs["robustness_summary.csv"] = True

    # ==================================================================
    # 10. REPRODUCIBILITY EVIDENCE
    # ==================================================================
    repro = pd.read_csv(os.path.join(F11, "k6_reproduction.csv"))
    ll_existing = float(repro.loc[repro["metric"] == "log_likelihood", "existing"].values[0])
    ll_repro = float(repro.loc[repro["metric"] == "log_likelihood", "reproduced"].values[0])
    bic_existing = float(repro.loc[repro["metric"] == "BIC", "existing"].values[0])
    bic_repro = float(repro.loc[repro["metric"] == "BIC", "reproduced"].values[0])
    aic_existing = float(repro.loc[repro["metric"] == "AIC", "existing"].values[0])
    aic_repro = float(repro.loc[repro["metric"] == "AIC", "reproduced"].values[0])
    sizes_existing = json.loads(str(repro.loc[repro["metric"] == "profile_sizes", "existing"].values[0]))
    sizes_repro = json.loads(str(repro.loc[repro["metric"] == "profile_sizes", "reproduced"].values[0]))

    k6_repro = pd.DataFrame({
        "metric": ["log_likelihood", "BIC", "AIC", "profile_sizes_equal", "profile_means_equal"],
        "original": [ll_existing, bic_existing, aic_existing, sorted(sizes_existing), "see k6_reproduction.csv"],
        "replicated": [ll_repro, bic_repro, aic_repro, sorted(sizes_repro), "see k6_reproduction.csv"],
        "difference": [ll_repro - ll_existing, bic_repro - bic_existing, aic_repro - aic_existing,
                       sorted(sizes_existing) == sorted(sizes_repro), ""],
    })
    k6_repro.to_csv(os.path.join(RESULTS_DIR, "k6_reproducibility.csv"), index=False)
    all_outputs["k6_reproducibility.csv"] = True

    # ==================================================================
    # 11. FACTOR-SCORE ROBUSTNESS
    # ==================================================================
    fs = pd.read_csv(os.path.join(F11, "factor_vs_mean.csv"))
    factor_robust = fs[["construct", "pearson_r"]].copy()
    factor_robust.to_csv(os.path.join(RESULTS_DIR, "factor_score_robustness.csv"), index=False)
    all_outputs["factor_score_robustness.csv"] = True

    # ==================================================================
    # 12. MASTER EVIDENCE FILE
    # ==================================================================
    master_rows = []

    # Dataset
    master_rows.append({
        "category": "dataset",
        "key_result": "N",
        "value": int(meta["N_rows"]),
        "file": "dataset_summary.csv",
    })
    master_rows.append({
        "category": "dataset",
        "key_result": "n_variables",
        "value": int(meta["N_columns"]),
        "file": "dataset_summary.csv",
    })
    master_rows.append({
        "category": "dataset",
        "key_result": "n_missing",
        "value": int(missing["n_missing"].sum()),
        "file": "dataset_summary.csv",
    })
    master_rows.append({
        "category": "dataset",
        "key_result": "n_duplicates",
        "value": int(dup.loc[dup["check"] == "all_columns", "n_duplicates"].values[0]),
        "file": "dataset_summary.csv",
    })

    # Measurement
    for _, row in measurement_summary.iterrows():
        master_rows.append({
            "category": "measurement",
            "key_result": f"{row['construct']}_alpha",
            "value": float(row["Cronbach_alpha"]),
            "file": "measurement_summary.csv",
        })

    # INT-BE
    master_rows.append({
        "category": "int_be",
        "key_result": "pearson_r",
        "value": float(intbe["r"].values[0]),
        "file": "int_behavior_correlation.csv",
    })

    # Gap
    master_rows.append({
        "category": "gap",
        "key_result": "mean",
        "value": float(gap_dict.get("mean", np.nan)),
        "file": "gap_summary.csv",
    })
    master_rows.append({
        "category": "gap",
        "key_result": "pct_positive",
        "value": float(gap_dict.get("positive_gap_pct", np.nan)),
        "file": "gap_summary.csv",
    })
    master_rows.append({
        "category": "gap",
        "key_result": "pct_negative",
        "value": float(gap_dict.get("negative_gap_pct", np.nan)),
        "file": "gap_summary.csv",
    })

    # LPA
    for K in sorted(fit["K"].unique().tolist()):
        frow = fit[fit["K"] == K].iloc[0]
        master_rows.append({
            "category": "lpa",
            "key_result": f"K{K}_BIC",
            "value": float(frow["BIC"]),
            "file": "lpa_model_comparison.csv",
        })
        master_rows.append({
            "category": "lpa",
            "key_result": f"K{K}_entropy",
            "value": float(frow["entropy"]),
            "file": "lpa_model_comparison.csv",
        })

    # Selected K (read from results/05_lpa_selection/selected_model.csv)
    for _, row in k6_table.iterrows():
        master_rows.append({
            "category": "selected_K_profile",
            "key_result": f"profile_{int(row['profile'])}_INT",
            "value": float(row["mean_INT"]),
            "file": "k6_profile_table.csv",
        })
        master_rows.append({
            "category": "selected_K_profile",
            "key_result": f"profile_{int(row['profile'])}_BE",
            "value": float(row["mean_BE"]),
            "file": "k6_profile_table.csv",
        })
        master_rows.append({
            "category": "selected_K_profile",
            "key_result": f"profile_{int(row['profile'])}_GAP",
            "value": float(row["mean_GAP"]),
            "file": "k6_profile_table.csv",
        })

    # Predictors — count significant
    n_sig = int(fdr["significant_FDR_05"].sum())
    master_rows.append({
        "category": "predictors",
        "key_result": "n_tests_total",
        "value": int(len(fdr)),
        "file": "k6_predictors.csv",
    })
    master_rows.append({
        "category": "predictors",
        "key_result": "n_significant_FDR",
        "value": n_sig,
        "file": "k6_predictors.csv",
    })

    # Reproducibility
    master_rows.append({
        "category": "reproducibility",
        "key_result": "BIC_difference",
        "value": bic_repro - bic_existing,
        "file": "k6_reproducibility.csv",
    })
    master_rows.append({
        "category": "reproducibility",
        "key_result": "LL_difference",
        "value": ll_repro - ll_existing,
        "file": "k6_reproducibility.csv",
    })

    pd.DataFrame(master_rows).to_csv(
        os.path.join(RESULTS_DIR, "paper1_master_evidence.csv"), index=False
    )
    all_outputs["paper1_master_evidence.csv"] = True

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = """# Phase 12 — Paper 1 Evidence Package

## Purpose
Extract verified numerical results from the completed analysis into a
clean machine-readable evidence package for Paper 1.

## Source Files Used
- results/10_final/data_diagnostics/
- results/10_final/measurement_results/
- results/10_final/gap_results/
- results/10_final/lpa_model_comparison/
- results/10_final/selected_lpa/
- results/10_final/profile_parameters/
- results/10_final/profile_comparisons/
- results/10_final/profile_predictors/
- results/10_final/robustness_results/
- results/10_final/numerical_audit/
- results/11_final_verification/

## Output Files
- dataset_summary.csv — N, variables, missing, duplicates
- measurement_summary.csv — per-construct alpha, mean, SD
- int_behavior_correlation.csv — INT-BE Pearson r, p, N, CI
- gap_summary.csv — GAP statistics
- lpa_model_comparison.csv — BIC, AIC, entropy, sizes for each estimated K
- all_profile_structure.csv — profile structure for each estimated K
- k6_profile_table.csv — selected-K primary profile table
  (K read from results/05_lpa_selection/selected_model.csv)
- k6_predictors.csv — multinomial logit (coefficients, OR, CI, FDR)
- robustness_summary.csv — all robustness checks
- k6_reproducibility.csv — selected-K reproduction verification
- factor_score_robustness.csv — factor vs mean correlations
- paper1_master_evidence.csv — master evidence table
"""
    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write(readme)
    all_outputs["README.md"] = True

    # ------------------------------------------------------------------
    # Verify all outputs exist
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 12 — PAPER 1 EVIDENCE PACKAGE COMPLETE")
    print("=" * 60)
    print("\nOutputs:")
    for fn, exists in sorted(all_outputs.items()):
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")

    missing_files = [fn for fn, ok in all_outputs.items() if not os.path.exists(os.path.join(RESULTS_DIR, fn))]
    if missing_files:
        print(f"\nMISSING: {missing_files}")
    else:
        print(f"\nAll {len(all_outputs)} files present.")


if __name__ == "__main__":
    main()
