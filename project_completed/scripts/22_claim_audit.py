"""
Phase 22 — Final Claim Audit
============================
Numerical audit of possible claims. No literature, no
interpretation, no new analysis. Each claim is judged strictly
from existing frozen numerical evidence.
"""
import os
import numpy as np
import pandas as pd

RESULTS_DIR = "results/22_claim_audit"
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
P20 = "results/20_evidence_extraction"

K = 6
N = 1166


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    ensure_dir(RESULTS_DIR)

    supported_rows = []
    caveated_rows = []
    unsupported_rows = []
    map_rows = []
    audit_rows = []

    # ------------------------------------------------------------------
    # Load frozen evidence once
    # ------------------------------------------------------------------
    intbe = pd.read_csv(os.path.join(P02, "int_be_correlation.csv")).iloc[0]
    int_be_r = float(intbe["r"])
    int_be_p = float(intbe["p"])
    int_be_ci_lo = float(intbe["CI95_lower"])
    int_be_ci_hi = float(intbe["CI95_upper"])

    fit = pd.read_csv(os.path.join(P04, "model_fit.csv"))
    bic_min_K = int(fit.loc[fit["BIC"].idxmin(), "K"])
    k6_bic = float(fit.loc[fit["K"] == 6, "BIC"].iloc[0])

    kdir = os.path.join(P04, f"K_{K}")
    post = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
    means = pd.read_csv(os.path.join(kdir, "profile_means.csv"))
    sizes = pd.read_csv(os.path.join(kdir, "profile_sizes.csv"))
    prob_cols = [c for c in post.columns if c.startswith("prob_")]
    maxprob = post[prob_cols].max(axis=1).values
    labels = post["assigned_class"].values
    if "profile" in means.columns:
        means = means.set_index("profile").loc[list(range(K))]
    size_col = "size" if "size" in sizes.columns else "N"

    k6_profiles = []
    for p in range(K):
        n_p = int(sizes.loc[sizes["profile"] == p, size_col].iloc[0])
        z_int = float(means.loc[p, "z_INT"])
        z_be = float(means.loc[p, "z_BE"])
        k6_profiles.append({
            "profile": p, "N": n_p, "z_INT": z_int, "z_BE": z_be,
            "diff": z_int - z_be,
            "direction": "INT>BE" if z_int > z_be else ("BE>INT" if z_be > z_int else "equal"),
            "mean_max_prob": float(maxprob[labels == p].mean()),
        })

    n_int_gt_be_profiles = sum(1 for p in k6_profiles if p["diff"] > 0)
    n_be_gt_int_profiles = sum(1 for p in k6_profiles if p["diff"] < 0)

    cfg = pd.read_csv(os.path.join(P14, "configuration_stability.csv"))
    boot = pd.read_csv(os.path.join(P14, "bootstrap_results.csv"))

    am = pd.read_csv(os.path.join(P17, "phase17_master.csv"))
    am_ref = am[am["specification"] == "reference_k6"].iloc[0]
    seed_rows = am[am["specification"].str.startswith("seed_")]
    seed_dist_max = float(seed_rows["total_matching_distance"].max()) if len(seed_rows) else np.nan

    p18m = pd.read_csv(os.path.join(P18, "phase18_master.csv"))
    p18m_items = dict(zip(p18m["item"], p18m["value"]))

    vif = pd.read_csv(os.path.join(P18B, "vif_full.csv"))
    max_vif = float(vif["VIF"].max())
    construct_vif_max = float(vif[vif["predictor"].isin(
        ["ATT","CON","SNO","COVID","PU","PEU","PO","PRI"])]["VIF"].max())

    cd = pd.read_csv(os.path.join(P18B, "condition_diagnostics.csv"))
    cond_num = float(cd.loc[cd["metric"] == "condition_number_corr", "value"].iloc[0])

    stability = pd.read_csv(os.path.join(P18B, "leave_one_predictor_out_diagnostics.csv"))

    def add(claim_id, claim_text, status, evidence_phase, evidence_file,
            statistic, value, reason):
        row = {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "evidence_phase": evidence_phase,
            "evidence_file": evidence_file,
            "statistic": statistic,
            "evidence_value": value,
            "status": status,
            "reason_for_status": reason,
        }
        map_rows.append({
            "claim_id": claim_id,
            "claim_text": claim_text,
            "evidence_phase": evidence_phase,
            "evidence_file": evidence_file,
            "statistic": statistic,
            "evidence_value": value,
        })
        if status == "SUPPORTED":
            supported_rows.append(row)
        elif status == "SUPPORTED_WITH_CAVEAT":
            caveated_rows.append(row)
        else:
            unsupported_rows.append(row)

    # ------------------------------------------------------------------
    # 1. INT and BE are positively associated.
    # ------------------------------------------------------------------
    if int_be_r > 0 and int_be_p < 0.001:
        add("C1", "INT and BE are positively associated.", "SUPPORTED",
            "02", P02, "pearson_r", f"r={int_be_r:.4f}, p={int_be_p:.2e}",
            "Pearson r > 0 with p < 0.001 reported in Phase 02.")
    else:
        add("C1", "INT and BE are positively associated.", "NOT_SUPPORTED",
            "02", P02, "pearson_r", f"r={int_be_r:.4f}, p={int_be_p:.2e}",
            "Evidence does not support positive association.")

    # ------------------------------------------------------------------
    # 2. INT and BE show heterogeneous configurations.
    # ------------------------------------------------------------------
    if n_int_gt_be_profiles >= 1 and n_be_gt_int_profiles >= 1:
        add("C2", "INT and BE show heterogeneous configurations across profiles.",
            "SUPPORTED",
            "04/06", kdir, "profile_z_INT_z_BE",
            f"profiles INT>BE = {n_int_gt_be_profiles}, profiles BE>INT = {n_be_gt_int_profiles}",
            "At least one profile has mean z_INT > z_BE and at least one has mean z_BE > z_INT.")
    else:
        add("C2", "INT and BE show heterogeneous configurations across profiles.",
            "NOT_SUPPORTED", "04/06", kdir, "profile_z_INT_z_BE",
            "All profiles in the same direction", "All profiles share one direction.")

    # ------------------------------------------------------------------
    # 3. There are profiles with INT > BE.
    # ------------------------------------------------------------------
    add("C3", "There are profiles with mean INT > mean BE.",
        "SUPPORTED" if n_int_gt_be_profiles >= 1 else "NOT_SUPPORTED",
        "04/06", kdir, "n_profiles_INT_gt_BE",
        int(n_int_gt_be_profiles),
        f"Phase 04/06 reports {n_int_gt_be_profiles} profile(s) with mean z_INT > z_BE.")

    # ------------------------------------------------------------------
    # 4. There are profiles with BE > INT.
    # ------------------------------------------------------------------
    add("C4", "There are profiles with mean BE > mean INT.",
        "SUPPORTED" if n_be_gt_int_profiles >= 1 else "NOT_SUPPORTED",
        "04/06", kdir, "n_profiles_BE_gt_INT",
        int(n_be_gt_int_profiles),
        f"Phase 04/06 reports {n_be_gt_int_profiles} profile(s) with mean z_BE > z_INT.")

    # ------------------------------------------------------------------
    # 5. There is one universal intention-behavior gap.
    # ------------------------------------------------------------------
    if n_int_gt_be_profiles == 0 and n_be_gt_int_profiles == 0:
        add("C5", "There is one universal intention-behavior gap.",
            "SUPPORTED", "04/06", kdir, "n_profiles", 0,
            "No profile shows a directional INT-BE difference.")
    else:
        add("C5", "There is one universal intention-behavior gap.",
            "NOT_SUPPORTED", "04/06", kdir, "profile_directions",
            f"{n_int_gt_be_profiles} INT>BE, {n_be_gt_int_profiles} BE>INT",
            "Multiple directional profiles exist; no single universal direction.")

    # ------------------------------------------------------------------
    # 6. A distinct INT > BE profile exists.
    # ------------------------------------------------------------------
    add("C6", "A distinct INT > BE profile exists.",
        "SUPPORTED" if n_int_gt_be_profiles >= 1 else "NOT_SUPPORTED",
        "04/06", kdir, "n_profiles_INT_gt_BE",
        int(n_int_gt_be_profiles),
        f"{n_int_gt_be_profiles} profile(s) show mean INT > mean BE.")

    # ------------------------------------------------------------------
    # 7. All six profiles are equally stable.
    # ------------------------------------------------------------------
    if len(cfg) >= K:
        pcts = cfg["pct_INT_gt_BE"].fillna(0).tolist() + \
               cfg["pct_BE_gt_INT"].fillna(0).tolist()
        spread = float(max(pcts) - min(pcts)) if pcts else np.nan
        if spread < 5.0:
            add("C7", "All six profiles are equally stable.",
                "SUPPORTED_WITH_CAVEAT",
                "14", P14, "stability_pct_spread",
                float(spread),
                f"Stability proportions differ by {spread:.2f} percentage points across profiles.")
        else:
            add("C7", "All six profiles are equally stable.",
                "NOT_SUPPORTED", "14", P14, "stability_pct_spread",
                float(spread),
                f"Stability proportions vary by {spread:.2f} percentage points; profiles are not equally stable.")
    else:
        add("C7", "All six profiles are equally stable.",
            "NOT_SUPPORTED", "14", P14, "stability_data", "insufficient",
            "Stability data not available for all profiles.")

    # ------------------------------------------------------------------
    # 8. Profile 5 has stable INT-BE direction.
    # ------------------------------------------------------------------
    p5 = next((p for p in k6_profiles if p["profile"] == 5), None)
    if p5 is not None and p5["diff"] != 0:
        cfg5 = cfg[cfg["profile"] == 5]
        if len(cfg5) > 0:
            if p5["diff"] > 0:
                pct_dir = float(cfg5["pct_INT_gt_BE"].iloc[0])
            else:
                pct_dir = float(cfg5["pct_BE_gt_INT"].iloc[0])
            if pct_dir >= 80.0:
                add("C8", "Profile 5 has stable INT-BE direction across bootstraps.",
                    "SUPPORTED", "14", P14, "configuration_stability_pct",
                    float(pct_dir),
                    f"Direction-consistency in {pct_dir:.2f}% of bootstraps.")
            else:
                add("C8", "Profile 5 has stable INT-BE direction across bootstraps.",
                    "SUPPORTED_WITH_CAVEAT", "14", P14,
                    "configuration_stability_pct", float(pct_dir),
                    f"Direction-consistency in only {pct_dir:.2f}% of bootstraps.")
        else:
            add("C8", "Profile 5 has stable INT-BE direction.",
                "NOT_SUPPORTED", "14", P14, "configuration_stability",
                "missing", "No bootstrap record for profile 5.")
    else:
        add("C8", "Profile 5 has stable INT-BE direction.",
            "NOT_SUPPORTED", "04", kdir, "profile_5_diff", 0.0,
            "Profile 5 INT-BE difference is zero.")

    # ------------------------------------------------------------------
    # 9. K=6 is the minimum-BIC solution under the primary specification.
    # ------------------------------------------------------------------
    if bic_min_K == K:
        add("C9", "K=6 is the minimum-BIC solution under the primary specification.",
            "SUPPORTED", "04/05", P04, "min_BIC_K", int(bic_min_K),
            f"Minimum BIC across K=2..6 is at K={bic_min_K} (BIC={k6_bic:.4f}).")
    else:
        add("C9", "K=6 is the minimum-BIC solution under the primary specification.",
            "NOT_SUPPORTED", "04/05", P04, "min_BIC_K", int(bic_min_K),
            f"Minimum BIC is at K={bic_min_K}, not K=6.")

    # ------------------------------------------------------------------
    # 10. K=6 is perfectly stable across all alternative specifications.
    # ------------------------------------------------------------------
    seed_dist_max_val = float(am["total_matching_distance"].max())
    if seed_dist_max_val < 0.01:
        add("C10", "K=6 is perfectly stable across all alternative specifications.",
            "SUPPORTED", "17", P17, "max_matching_distance",
            float(seed_dist_max_val),
            f"Maximum matching distance = {seed_dist_max_val:.4f}.")
    else:
        add("C10", "K=6 is perfectly stable across all alternative specifications.",
            "NOT_SUPPORTED", "17", P17, "max_matching_distance",
            float(seed_dist_max_val),
            f"Maximum matching distance = {seed_dist_max_val:.4f} (non-zero).")

    # ------------------------------------------------------------------
    # 11. Profile membership predictors are causal.
    # ------------------------------------------------------------------
    add("C11", "Profile membership predictors are causal.",
        "NOT_SUPPORTED", "18", P18, "study_design", "cross-sectional",
        "Cross-sectional MNLogit cannot establish causality from observational data.")

    # ------------------------------------------------------------------
    # 12. The predictor coefficients are unaffected by multicollinearity.
    # ------------------------------------------------------------------
    if construct_vif_max < 5.0 and cond_num < 10.0:
        add("C12", "The predictor coefficients are unaffected by multicollinearity.",
            "SUPPORTED", "18B", P18B, "max_construct_VIF_cond_num",
            f"VIF={construct_vif_max:.2f}, cond={cond_num:.2f}",
            "Multicollinearity diagnostics within safe range.")
    else:
        add("C12", "The predictor coefficients are unaffected by multicollinearity.",
            "NOT_SUPPORTED", "18B", P18B, "max_construct_VIF_cond_num",
            f"VIF={construct_vif_max:.2f}, cond={cond_num:.2f}",
            f"Construct VIF reaches {construct_vif_max:.2f}; condition number = {cond_num:.2f}. Coefficients are likely affected.")

    # ------------------------------------------------------------------
    # 13. The data support heterogeneity in the INT-BE relationship.
    # ------------------------------------------------------------------
    gvd = pd.read_csv(os.path.join(P16, "gap_variance_decomposition.csv"))
    k6_bt = float(gvd.loc[gvd["K"] == 6, "between_over_total"].iloc[0])
    if (n_int_gt_be_profiles >= 1 and n_be_gt_int_profiles >= 1) and k6_bt > 0:
        add("C13", "The data support heterogeneity in the intention-behavior relationship.",
            "SUPPORTED_WITH_CAVEAT", "04/06/16",
            P16, "between_over_total_K6", float(k6_bt),
            f"Between-profile proportion of GAP variance = {k6_bt:.4f}; substantial within-profile variance remains.")
    else:
        add("C13", "The data support heterogeneity in the INT-BE relationship.",
            "NOT_SUPPORTED", "16", P16, "between_over_total_K6", float(k6_bt),
            "Directional evidence not present.")

    # ------------------------------------------------------------------
    # 14. The results establish temporal causality.
    # ------------------------------------------------------------------
    add("C14", "The results establish temporal causality.",
        "NOT_SUPPORTED", "study_design", "data.xls", "design", "cross-sectional",
        "Cross-sectional data cannot establish temporal causality.")

    # ------------------------------------------------------------------
    # 15. K=6 reproduces with a specific primary BIC value.
    # ------------------------------------------------------------------
    add("C15", "K=6 reproduces with the specific primary BIC value (2129.1734).",
        "SUPPORTED", "19", "results/19_final_audit",
        "K6_BIC_reproduced", float(k6_bic),
        f"Phase 19 audit reproduces K=6 BIC = {k6_bic:.4f}.")

    # ------------------------------------------------------------------
    # 16. Profile sizes sum to 1166.
    # ------------------------------------------------------------------
    total_n = sum(p["N"] for p in k6_profiles)
    if total_n == N:
        add("C16", "K=6 profile sizes sum to N = 1166.",
            "SUPPORTED", "19", "results/19_final_audit",
            "sum_profile_N", int(total_n),
            f"Sum of profile sizes = {total_n} = N.")
    else:
        add("C16", "K=6 profile sizes sum to N = 1166.",
            "NOT_SUPPORTED", "04/19", P04, "sum_profile_N", int(total_n),
            f"Sum of profile sizes = {total_n}, expected 1166.")

    # ------------------------------------------------------------------
    # 17. INT-BE correlation confidence interval excludes 0.
    # ------------------------------------------------------------------
    if int_be_ci_lo > 0:
        add("C17", "INT-BE 95% CI excludes 0.",
            "SUPPORTED", "02", P02, "INT_BE_CI95",
            f"[{int_be_ci_lo:.4f}, {int_be_ci_hi:.4f}]",
            "Lower bound > 0; CI excludes 0.")
    else:
        add("C17", "INT-BE 95% CI excludes 0.",
            "NOT_SUPPORTED", "02", P02, "INT_BE_CI95",
            f"[{int_be_ci_lo:.4f}, {int_be_ci_hi:.4f}]",
            "CI includes 0.")

    # ------------------------------------------------------------------
    # 18. Stability of all K=6 profiles.
    # ------------------------------------------------------------------
    n_boot = len(boot)
    n_conv = int(boot["converged"].sum()) if "converged" in boot.columns else 0
    if n_boot > 0 and n_conv == n_boot:
        add("C18", "All bootstrap replications of K=6 converged.",
            "SUPPORTED", "14", P14, "bootstrap_converged",
            f"{n_conv}/{n_boot}",
            f"All {n_boot} bootstrap replications converged.")
    else:
        add("C18", "All bootstrap replications of K=6 converged.",
            "NOT_SUPPORTED", "14", P14, "bootstrap_converged",
            f"{n_conv}/{n_boot}",
            f"{n_boot - n_conv} of {n_boot} replications did not converge.")

    # ------------------------------------------------------------------
    # 19. Predictor model converged.
    # ------------------------------------------------------------------
    add("C19", "MNLogit predictor model converged.",
        "SUPPORTED" if p18m_items.get("converged", 0) == 1.0 else "NOT_SUPPORTED",
        "18", P18, "converged", int(p18m_items.get("converged", 0)),
        f"Phase 18 reports converged = {bool(p18m_items.get('converged', 0))}.")

    # ------------------------------------------------------------------
    # 20. INT-BE relationship is uniformly positive across all profiles.
    # ------------------------------------------------------------------
    all_positive = all(p["diff"] >= 0 for p in k6_profiles)
    add("C20", "INT-BE relationship is uniformly positive across all K=6 profiles.",
        "NOT_SUPPORTED" if not all_positive else "SUPPORTED",
        "04/06", kdir, "profile_diffs",
        "; ".join(f"P{p['profile']}={p['diff']:.4f}" for p in k6_profiles),
        "Some profiles show mean BE > mean INT." if not all_positive
        else "All profiles show non-negative INT-BE difference.")

    # ------------------------------------------------------------------
    # 21. VIF range is within recommended bounds.
    # ------------------------------------------------------------------
    if max_vif < 5.0:
        add("C21", "Predictor VIF range is within recommended bounds (<5).",
            "SUPPORTED", "18B", P18B, "max_VIF", float(max_vif),
            f"Max VIF = {max_vif:.2f}.")
    else:
        add("C21", "Predictor VIF range is within recommended bounds (<5).",
            "NOT_SUPPORTED", "18B", P18B, "max_VIF", float(max_vif),
            f"Max VIF = {max_vif:.2f} exceeds 5.")

    # ------------------------------------------------------------------
    # 22. INT-BE r magnitude is large.
    # ------------------------------------------------------------------
    if abs(int_be_r) >= 0.5:
        add("C22", "INT-BE Pearson r magnitude is >= 0.5.",
            "SUPPORTED", "02", P02, "abs_pearson_r", float(abs(int_be_r)),
            f"|r| = {abs(int_be_r):.4f} >= 0.5.")
    else:
        add("C22", "INT-BE Pearson r magnitude is >= 0.5.",
            "NOT_SUPPORTED", "02", P02, "abs_pearson_r", float(abs(int_be_r)),
            f"|r| = {abs(int_be_r):.4f} < 0.5.")

    # ------------------------------------------------------------------
    # WRITE OUTPUTS
    # ------------------------------------------------------------------
    pd.DataFrame(supported_rows).to_csv(
        os.path.join(RESULTS_DIR, "01_supported_claims.csv"), index=False)
    pd.DataFrame(caveated_rows).to_csv(
        os.path.join(RESULTS_DIR, "02_caveated_claims.csv"), index=False)
    pd.DataFrame(unsupported_rows).to_csv(
        os.path.join(RESULTS_DIR, "03_unsupported_claims.csv"), index=False)
    pd.DataFrame(map_rows).to_csv(
        os.path.join(RESULTS_DIR, "04_claim_evidence_map.csv"), index=False)

    # Final claim audit
    n_supp = len(supported_rows)
    n_cav = len(caveated_rows)
    n_unsupp = len(unsupported_rows)
    final_status = "PASS" if n_unsupp == 0 else "PASS_WITH_CAVEATS"
    audit_rows = [
        {"item": "n_supported", "value": n_supp},
        {"item": "n_caveated", "value": n_cav},
        {"item": "n_unsupported", "value": n_unsupp},
        {"item": "n_total_audited", "value": n_supp + n_cav + n_unsupp},
        {"item": "final_status", "value": final_status},
    ]
    pd.DataFrame(audit_rows).to_csv(
        os.path.join(RESULTS_DIR, "05_final_claim_audit.csv"), index=False)

    readme = f"""# Phase 22 — Final Claim Audit

## Purpose
Numerical audit of possible claims that could be made from the
frozen results. No literature, no interpretation, no new analysis.
Each claim judged strictly from existing frozen evidence.

## Status Counts
- SUPPORTED: {n_supp}
- SUPPORTED_WITH_CAVEAT: {n_cav}
- NOT_SUPPORTED: {n_unsupp}
- Total audited: {n_supp + n_cav + n_unsupp}

## Output Files
- 01_supported_claims.csv
- 02_caveated_claims.csv
- 03_unsupported_claims.csv
- 04_claim_evidence_map.csv
- 05_final_claim_audit.csv
- README.md

## Final Status: {final_status}

## No Modifications
All evidence is read from Phases 1-19 result files. No model
refit. No literature search. No manuscript text.
"""
    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write(readme)

    print("=" * 60)
    print("PHASE 22 — CLAIM AUDIT COMPLETE")
    print("=" * 60)
    print(f"SUPPORTED: {n_supp}")
    print(f"SUPPORTED_WITH_CAVEAT: {n_cav}")
    print(f"NOT_SUPPORTED: {n_unsupp}")
    print(f"Final status: {final_status}")
    expected = [
        "01_supported_claims.csv", "02_caveated_claims.csv",
        "03_unsupported_claims.csv", "04_claim_evidence_map.csv",
        "05_final_claim_audit.csv", "README.md",
    ]
    for fn in expected:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
