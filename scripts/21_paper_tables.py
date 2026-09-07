"""
Phase 21 — Final Paper Table Data
==================================
Creates clean numerical tables for Paper 1. No prose, no plots,
no model selection. All numbers come from existing verified
result files (Phases 1-19).
"""
import os
import json
import numpy as np
import pandas as pd

RESULTS_DIR = "results/21_paper_tables"
P01 = "results/01_data_inspection"
P02 = "results/02_measurement"
P03 = "results/03_gap_analysis"
P04 = "results/04_lpa_estimation"
P13 = "results/13_k6_profile_validation"
P14 = "results/14_k6_stability"
P15 = "results/15_profile_structure_comparison"
P16 = "results/16_gap_profile_validation"
P17 = "results/17_alternative_model_validation"
P18 = "results/18_profile_predictors"
P18B = "results/18B_predictor_multicollinearity"
P19 = "results/19_final_audit"

K = int(pd.read_csv(os.path.join("results/05_lpa_selection", "selected_model.csv"))["selected_K"].iloc[0])
N = int(pd.read_csv(os.path.join("results/01_data_inspection", "data_dimensions.csv"))["N_rows"].iloc[0])


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    ensure_dir(RESULTS_DIR)
    master_rows = []

    # ------------------------------------------------------------------
    # 01. SAMPLE TABLE
    # ------------------------------------------------------------------
    dims = pd.read_csv(os.path.join(P01, "data_dimensions.csv")).iloc[0]
    miss = pd.read_csv(os.path.join(P01, "missing_values.csv"))
    dup = pd.read_csv(os.path.join(P01, "duplicate_information.csv"))
    irng = pd.read_csv(os.path.join(P01, "item_ranges.csv"))
    n_dup = int(dup.loc[dup["check"] == "all_columns", "n_duplicates"].iloc[0])
    sample_rows = [
        {"field": "N_respondents", "value": int(dims["N_rows"])},
        {"field": "N_variables", "value": int(dims["N_columns"])},
        {"field": "total_missing_values", "value": int(miss["n_missing"].sum())},
        {"field": "duplicate_rows", "value": n_dup},
        {"field": "psychometric_item_min", "value": int(irng["min"].min())},
        {"field": "psychometric_item_max", "value": int(irng["max"].max())},
        {"field": "instrument_scale", "value": "1-5 Likert"},
    ]
    pd.DataFrame(sample_rows).to_csv(
        os.path.join(RESULTS_DIR, "01_sample_table.csv"), index=False)
    master_rows.append({"table": "01_sample", "N": N, "n_rows": len(sample_rows)})

    # ------------------------------------------------------------------
    # 02. MEASUREMENT TABLE
    # ------------------------------------------------------------------
    cstat = pd.read_csv(os.path.join(P02, "construct_statistics.csv"))
    meas_rows = []
    for _, r in cstat.iterrows():
        meas_rows.append({
            "construct": r["construct"],
            "k_items": int(r["k_items"]),
            "mean": float(r["mean"]),
            "SD": float(r["SD"]),
            "min": float(r["min"]),
            "max": float(r["max"]),
            "Cronbach_alpha": float(r["Cronbach_alpha"]),
        })
    pd.DataFrame(meas_rows).to_csv(
        os.path.join(RESULTS_DIR, "02_measurement_table.csv"), index=False)
    master_rows.append({"table": "02_measurement", "N": N, "n_rows": len(meas_rows)})

    # ------------------------------------------------------------------
    # 03. CORRELATION TABLE
    # ------------------------------------------------------------------
    cmat = pd.read_csv(os.path.join(P02, "construct_correlation_matrix.csv"),
                       index_col=0)
    cor_rows = []
    for r in cmat.index:
        for c in cmat.columns:
            cor_rows.append({"row": r, "col": c, "r": float(cmat.loc[r, c])})
    pd.DataFrame(cor_rows).to_csv(
        os.path.join(RESULTS_DIR, "03_correlation_table.csv"), index=False)
    intbe = pd.read_csv(os.path.join(P02, "int_be_correlation.csv")).iloc[0]
    cor_summary_rows = [
        {"pair": "INT-BE", "r": float(intbe["r"]), "p": float(intbe["p"]),
         "N": int(intbe["N"]), "CI95_lower": float(intbe["CI95_lower"]),
         "CI95_upper": float(intbe["CI95_upper"])},
    ]
    pd.DataFrame(cor_summary_rows).to_csv(
        os.path.join(RESULTS_DIR, "03_correlation_summary.csv"), index=False)
    master_rows.append({"table": "03_correlation", "N": N, "n_rows": len(cor_rows)})

    # ------------------------------------------------------------------
    # 04. LPA COMPARISON TABLE
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
        os.path.join(RESULTS_DIR, "04_lpa_comparison_table.csv"), index=False)
    master_rows.append({"table": "04_lpa_comparison", "N": N, "n_rows": len(lpa_rows)})

    # ------------------------------------------------------------------
    # 05. SELECTED-K PROFILE TABLE
    # ------------------------------------------------------------------
    kdir = os.path.join(P04, f"K_{K}")
    post = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
    means = pd.read_csv(os.path.join(kdir, "profile_means.csv"))
    sizes = pd.read_csv(os.path.join(kdir, "profile_sizes.csv"))
    prob_cols = [c for c in post.columns if c.startswith("post_profile_")]
    maxprob = post[prob_cols].max(axis=1).values
    labels = post["assigned_class"].values
    if "profile" in means.columns:
        means = means.set_index("profile").loc[list(range(K))]
    score_path = "results/02_measurement/construct_scores.csv"
    scores = pd.read_csv(score_path, index_col="respondent")
    INT = scores["INT"].values
    BE = scores["BE"].values
    size_col = "size" if "size" in sizes.columns else "N"
    k6_rows = []
    for p in range(K):
        mask = labels == p
        n_p = int(sizes.loc[sizes["profile"] == p, size_col].iloc[0])
        k6_rows.append({
            "profile": p,
            "N": n_p,
            "percentage": float(n_p / N * 100),
            "INT_mean": float(INT[mask].mean()),
            "BE_mean": float(BE[mask].mean()),
            "INT_minus_BE": float(INT[mask].mean() - BE[mask].mean()),
            "mean_z_INT": float(means.loc[p, "z_INT"]),
            "mean_z_BE": float(means.loc[p, "z_BE"]),
            "mean_max_posterior": float(maxprob[mask].mean()),
            "median_max_posterior": float(np.median(maxprob[mask])),
        })
    pd.DataFrame(k6_rows).to_csv(
        os.path.join(RESULTS_DIR, "05_k6_profile_table.csv"), index=False)
    master_rows.append({"table": "05_k6_profile", "N": N, "n_rows": len(k6_rows)})

    # ------------------------------------------------------------------
    # 06. PROFILE-GAP TABLE
    # ------------------------------------------------------------------
    gv = pd.read_csv(os.path.join(P16, "gap_variance_decomposition.csv"))
    pg_rows = []
    for _, r in gv.iterrows():
        pg_rows.append({
            "K": int(r["K"]),
            "between_profile_variance": float(r["between_profile_variance"]),
            "within_profile_variance": float(r["within_profile_variance"]),
            "total_variance": float(r["total_variance"]),
            "between_over_total": float(r["between_over_total"]),
        })
    pd.DataFrame(pg_rows).to_csv(
        os.path.join(RESULTS_DIR, "06_profile_gap_table.csv"), index=False)
    go = pd.read_csv(os.path.join(P16, "gap_only_model_comparison.csv"))
    lgr = pd.read_csv(os.path.join(P16, "level_gap_correlation.csv")).iloc[0]
    go_bic_min = go.loc[go["BIC"].idxmin()]
    gap_extra = [
        {"K": "gap_only_min_BIC_K", "value": int(go_bic_min["K"])},
        {"K": "gap_only_min_BIC", "value": float(go_bic_min["BIC"])},
        {"K": "LEVEL_GAP_corr_r", "value": float(lgr["correlation_LEVEl_GAP_r"])},
        {"K": "LEVEL_GAP_p", "value": float(lgr["p_value"])},
    ]
    pd.DataFrame(gap_extra).to_csv(
        os.path.join(RESULTS_DIR, "06_profile_gap_supplement.csv"), index=False)
    master_rows.append({"table": "06_profile_gap", "N": N, "n_rows": len(pg_rows)})

    # ------------------------------------------------------------------
    # 07. CLASSIFICATION TABLE
    # ------------------------------------------------------------------
    cls_rows = [
        {"metric": "mean_max_posterior", "value": float(maxprob.mean())},
        {"metric": "median_max_posterior", "value": float(np.median(maxprob))},
    ]
    for thr in [0.90, 0.80, 0.70, 0.50]:
        cls_rows.append({"metric": f"pct_>={thr:.2f}",
                         "value": float((maxprob >= thr).mean() * 100)})
        cls_rows.append({"metric": f"pct_<{thr:.2f}",
                         "value": float((maxprob < thr).mean() * 100)})
    pd.DataFrame(cls_rows).to_csv(
        os.path.join(RESULTS_DIR, "07_classification_table.csv"), index=False)
    master_rows.append({"table": "07_classification", "N": N,
                        "n_rows": len(cls_rows)})

    # ------------------------------------------------------------------
    # 08. STABILITY TABLE
    # ------------------------------------------------------------------
    boot = pd.read_csv(os.path.join(P14, "bootstrap_results.csv"))
    cfg = pd.read_csv(os.path.join(P14, "configuration_stability.csv"))
    split = pd.read_csv(os.path.join(P14, "split_sample_results.csv"))
    stab_rows = [
        {"metric": "bootstrap_B", "value": int(len(boot)),
         "source": "results/14_k6_stability/bootstrap_results.csv"},
        {"metric": "bootstrap_converged",
         "value": int(boot["converged"].sum()),
         "source": "results/14_k6_stability/bootstrap_results.csv"},
        {"metric": "bootstrap_mean_matching_cost",
         "value": float(boot["matching_cost"].mean()),
         "source": "results/14_k6_stability/bootstrap_results.csv"},
    ]
    for _, r in cfg.iterrows():
        stab_rows.append({
            "metric": f"profile_{int(r['profile'])}_pct_INT_gt_BE",
            "value": float(r["pct_INT_gt_BE"]),
            "source": "results/14_k6_stability/configuration_stability.csv",
        })
        stab_rows.append({
            "metric": f"profile_{int(r['profile'])}_pct_BE_gt_INT",
            "value": float(r["pct_BE_gt_INT"]),
            "source": "results/14_k6_stability/configuration_stability.csv",
        })
    # split_sample_results.csv has one ROW PER PROFILE per half. `n_respondents`
    # is the half-N repeated on every profile row of that half, and `size` is
    # the profile's headcount within the half. Correct totals: N = sum of
    # `size` over both halves (=1166); per-half N = `n_respondents` of one row.
    _n_halves = int(split["half"].nunique())
    _n_total = int(split["size"].sum())
    _n_per_half = int(split.groupby("half")["size"].sum().iloc[0])
    stab_rows.append({
        "metric": "split_sample_n_halves",
        "value": _n_halves,
        "source": "results/14_k6_stability/split_sample_results.csv",
    })
    stab_rows.append({
        "metric": "split_sample_n_respondents",
        "value": _n_total,
        "source": "results/14_k6_stability/split_sample_results.csv",
    })
    stab_rows.append({
        "metric": "split_sample_n_per_half",
        "value": _n_per_half,
        "source": "results/14_k6_stability/split_sample_results.csv",
    })
    stab_rows.append({
        "metric": "split_sample_matching_cost",
        "value": float(pd.read_csv(
            os.path.join(P14, "split_sample_matching.csv"))
            ["matching_cost_total"].iloc[0]),
        "source": "results/14_k6_stability/split_sample_matching.csv",
    })
    pd.DataFrame(stab_rows).to_csv(
        os.path.join(RESULTS_DIR, "08_stability_table.csv"), index=False)
    master_rows.append({"table": "08_stability", "N": N, "n_rows": len(stab_rows)})

    # ------------------------------------------------------------------
    # 09. PREDICTOR TABLE
    # ------------------------------------------------------------------
    p18m = pd.read_csv(os.path.join(P18, "phase18_master.csv"))
    p18m_items = dict(zip(p18m["item"], p18m["value"]))
    p18_multinom = pd.read_csv(os.path.join(P18, "multinomial_results.csv"))
    p18_summary = pd.read_csv(os.path.join(P18, "predictor_summary.csv"))
    n_sig = int(p18_multinom["significant_FDR"].sum())
    pred_summary = [
        {"metric": "N", "value": int(p18m_items["N"])},
        {"metric": "K", "value": int(p18m_items["K_frozen"])},
        {"metric": "reference_profile",
         "value": int(p18m_items["reference_profile"])},
        {"metric": "n_predictors",
         "value": int(p18m_items["n_predictors"])},
        {"metric": "n_tests", "value": int(p18m_items["n_tests"])},
        {"metric": "log_likelihood",
         "value": float(p18m_items["log_likelihood"])},
        {"metric": "AIC", "value": float(p18m_items["AIC"])},
        {"metric": "BIC", "value": float(p18m_items["BIC"])},
        {"metric": "McFadden_pseudo_R2",
         "value": float(p18m_items["McFadden_pseudo_R2"])},
        {"metric": "n_FDR_significant", "value": n_sig},
    ]
    pd.DataFrame(pred_summary).to_csv(
        os.path.join(RESULTS_DIR, "09_predictor_table.csv"), index=False)
    p18_summary.to_csv(
        os.path.join(RESULTS_DIR, "09_predictor_per_predictor.csv"), index=False)
    master_rows.append({"table": "09_predictor", "N": N,
                        "n_rows": len(pred_summary)})

    # ------------------------------------------------------------------
    # 10. ROBUSTNESS TABLE
    # ------------------------------------------------------------------
    am = pd.read_csv(os.path.join(P17, "phase17_master.csv"))
    vif = pd.read_csv(os.path.join(P18B, "vif_full.csv"))
    cd = pd.read_csv(os.path.join(P18B, "condition_diagnostics.csv"))
    pairs = pd.read_csv(os.path.join(P18B, "high_correlation_pairs.csv"))
    rob_rows = []
    for _, r in am.iterrows():
        rob_rows.append({
            "specification": str(r["specification"]),
            "BIC": float(r["BIC"]),
            "AIC": float(r["AIC"]),
            "entropy": float(r["entropy"]),
            "total_matching_distance": float(r["total_matching_distance"]),
        })
    pd.DataFrame(rob_rows).to_csv(
        os.path.join(RESULTS_DIR, "10_robustness_table.csv"), index=False)
    cd_extra = cd.set_index("metric")["value"].to_dict()
    rob_extra = [
        {"metric": "VIF_min", "value": float(vif["VIF"].min()),
         "source": "results/18B_predictor_multicollinearity/vif_full.csv"},
        {"metric": "VIF_max", "value": float(vif["VIF"].max()),
         "source": "results/18B_predictor_multicollinearity/vif_full.csv"},
        {"metric": "VIF_predictor_max",
         "value": str(vif.iloc[0]["predictor"]),
         "source": "results/18B_predictor_multicollinearity/vif_full.csv"},
        {"metric": "condition_number_corr",
         "value": float(cd_extra.get("condition_number_corr", np.nan)),
         "source": "results/18B_predictor_multicollinearity/condition_diagnostics.csv"},
        {"metric": "smallest_eigenvalue_corr",
         "value": float(cd_extra.get("smallest_eigenvalue_corr", np.nan)),
         "source": "results/18B_predictor_multicollinearity/condition_diagnostics.csv"},
        {"metric": "n_pairs_abs_r_ge_0.70",
         "value": int(len(pairs)),
         "source": "results/18B_predictor_multicollinearity/high_correlation_pairs.csv"},
    ]
    pd.DataFrame(rob_extra).to_csv(
        os.path.join(RESULTS_DIR, "10_robustness_supplement.csv"), index=False)
    master_rows.append({"table": "10_robustness", "N": N, "n_rows": len(rob_rows)})

    # ------------------------------------------------------------------
    # 11. MASTER PAPER TABLES
    # ------------------------------------------------------------------
    pd.DataFrame(master_rows).to_csv(
        os.path.join(RESULTS_DIR, "11_master_paper_tables.csv"), index=False)

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = f"""# Phase 21 — Final Paper Table Data

## Purpose
Clean numerical tables for Paper 1. Every number is copied from
existing frozen result files. No new analysis, no new selection,
no plots, no prose.

## Tables Created
- 01_sample_table.csv
- 02_measurement_table.csv
- 03_correlation_table.csv + 03_correlation_summary.csv
- 04_lpa_comparison_table.csv
- 05_k6_profile_table.csv
- 06_profile_gap_table.csv + 06_profile_gap_supplement.csv
- 07_classification_table.csv
- 08_stability_table.csv
- 09_predictor_table.csv + 09_predictor_per_predictor.csv
- 10_robustness_table.csv + 10_robustness_supplement.csv
- 11_master_paper_tables.csv (index)
- README.md

## Frozen values preserved
- N = {N}
- K = {K} (minimum BIC among non-degenerate fits, per
  results/05_lpa_selection/selected_model.csv)
- K={K} profile sizes = {[r['N'] for r in k6_rows]}
- INT-BE r = {float(intbe['r']):.4f}
- K={K} BIC = {float(fit.loc[fit['K'] == K, 'BIC'].iloc[0]):.4f}
- K={K} AIC = {float(fit.loc[fit['K'] == K, 'AIC'].iloc[0]):.4f}
- K={K} normalized classification entropy E = -sum(p log p)/(n log K)
  (in [0,1]; higher = better separated)

## No Modifications
All values copied from existing result files. No rounding changes
that affect interpretation. No silent removals.
"""
    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write(readme)

    print("=" * 60)
    print("PHASE 21 — PAPER TABLE DATA COMPLETE")
    print("=" * 60)
    expected = [
        "01_sample_table.csv", "02_measurement_table.csv",
        "03_correlation_table.csv", "03_correlation_summary.csv",
        "04_lpa_comparison_table.csv", "05_k6_profile_table.csv",
        "06_profile_gap_table.csv", "06_profile_gap_supplement.csv",
        "07_classification_table.csv", "08_stability_table.csv",
        "09_predictor_table.csv", "09_predictor_per_predictor.csv",
        "10_robustness_table.csv", "10_robustness_supplement.csv",
        "11_master_paper_tables.csv", "README.md",
    ]
    for fn in expected:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
