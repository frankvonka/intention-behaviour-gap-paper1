"""Repack results/paper1_final/ from corrected phase outputs (copies only)."""
import os, shutil

def _cp(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  packed {dst}")
    else:
        print(f"  [MISSING] {src}")

PF = "results/paper1_final"

for f in ["04_lpa_comparison_table.csv"]:
    for d in ["results/21_paper_tables", "results/05_lpa_selection", "results/04_lpa_estimation"]:
        p = os.path.join(d, f)
        if os.path.exists(p):
            _cp(p, f"{PF}/04_lpa/{f}")
            break
for f in ["model_fit.csv"]:
    _cp(os.path.join("results/04_lpa_estimation", f), f"{PF}/04_lpa/{f}")
for f in ["05_lpa_model_evidence.csv"]:
    _cp(os.path.join("results/20_evidence_extraction", f), f"{PF}/04_lpa/{f}")
for f in ["lpa_model_audit.csv"]:
    _cp(os.path.join("results/19_final_audit", f), f"{PF}/04_lpa/{f}")
for f in ["selected_model.csv"]:
    _cp(os.path.join("results/05_lpa_selection", f), f"{PF}/04_lpa/{f}")

for f in ["05_k6_profile_table.csv", "07_classification_table.csv",
          "08_stability_table.csv"]:
    for d in ["results/21_paper_tables", "results/13_k6_profile_validation",
              "results/06_profile_analysis"]:
        p = os.path.join(d, f)
        if os.path.exists(p):
            _cp(p, f"{PF}/05_profiles/{f}")
            break

for f in ["10_robustness_table.csv", "10_robustness_supplement.csv"]:
    for d in ["results/21_paper_tables", "results/17_alternative_model_validation",
              "results/08_robustness"]:
        p = os.path.join(d, f)
        if os.path.exists(p):
            _cp(p, f"{PF}/07_robustness/{f}")
            break
for f in ["covariance_sensitivity.csv", "random_seed_sensitivity.csv",
          "random_start_stability.csv"]:
    _cp(os.path.join("results/08_robustness", f), f"{PF}/07_robustness/{f}")
import pandas as _pd
_k_sel = int(_pd.read_csv(os.path.join("results/05_lpa_selection", "selected_model.csv"))["selected_K"].iloc[0])
for f in [f"reference_K{_k_sel}.csv"]:
    for d in ["results/17_alternative_model_validation", "results/08_robustness"]:
        p = os.path.join(d, f)
        if os.path.exists(p):
            _cp(p, f"{PF}/07_robustness/{f}")
            break
for f in ["phase17_master.csv"]:
    _cp(os.path.join("results/17_alternative_model_validation", f), f"{PF}/07_robustness/{f}")

for f in ["05_final_claim_audit.csv", "01_supported_claims.csv",
          "02_caveated_claims.csv", "03_unsupported_claims.csv",
          "04_claim_evidence_map.csv"]:
    for d in ["results/22_claim_audit", "results/19_final_audit"]:
        p = os.path.join(d, f)
        if os.path.exists(p):
            _cp(p, f"{PF}/08_audit/{f}")
            break
for f in ["final_evidence.csv"]:
    _cp(os.path.join("results/19_final_audit", f), f"{PF}/08_audit/{f}")

for f in ["12_final_evidence_matrix.csv"]:
    _cp(os.path.join("results/20_evidence_extraction", f), f"{PF}/09_master/{f}")

# fig4 data: refresh from corrected phase 04 model_fit
import pandas as pd
fit = pd.read_csv("results/04_lpa_estimation/model_fit.csv")
fig4 = fit[["K", "log_likelihood", "AIC", "BIC", "entropy"]].sort_values("K")
fig4.to_csv(f"{PF}/figures/fig4/fig4_data.csv", index=False)
print("  refreshed figures/fig4/fig4_data.csv")

# phase15 master copy if present
for f in ["phase15_master.csv"]:
    for d in ["results/15_profile_structure_comparison", f"{PF}/05_profiles"]:
        p = os.path.join(d, f)
        if os.path.exists(p) and d != f"{PF}/05_profiles":
            _cp(p, f"{PF}/05_profiles/{f}")
            break
print("DONE")
