"""Build paper1_master_evidence.csv from frozen evidence."""
import pandas as pd

matrix = pd.read_csv("results/paper1_final/09_master/12_final_evidence_matrix.csv")
rows = []
for _, r in matrix.iterrows():
    rows.append({
        "section": "evidence_matrix",
        "result_name": str(r["claim_id"]) + ": " + str(r["claim"]),
        "value": r["numerical_value"],
        "unit_or_scale": r["units"],
        "source_phase": r["source_phase"],
        "source_file": r["source_file"],
        "status": r["status"],
        "notes": str(r["statistic"]),
    })

def add(section, name, value, unit, phase, file, notes=""):
    rows.append({"section": section, "result_name": name, "value": value,
                 "unit_or_scale": unit, "source_phase": phase,
                 "source_file": file, "status": "VERIFIED", "notes": notes})

add("sample", "N", 1166, "count", "01", "01_data/data_dimensions.csv")
add("sample", "N_variables", 34, "count", "01", "01_data/data_dimensions.csv")
add("sample", "missing_values", 0, "count", "01", "01_data/missing_values.csv")
add("sample", "duplicate_rows", 42, "count", "01", "01_data/duplicate_information.csv")
add("measurement", "INT-BE_pearson_r", 0.6515, "r", "02", "02_measurement/int_be_correlation.csv")
add("measurement", "INT-BE_95CI", "[0.6171, 0.6833]", "r", "02", "02_measurement/int_be_correlation.csv")
add("lpa", "selected_K", 6, "K", "04/05", "04_lpa/model_fit.csv")
add("lpa", "K6_BIC", 2129.1734, "nats", "04", "04_lpa/model_fit.csv")
add("lpa", "K6_AIC", 1952.0267, "nats", "04", "04_lpa/model_fit.csv")
add("lpa", "K6_entropy", 1.1418, "entropy", "04", "04_lpa/model_fit.csv")
add("profile", "profile_sizes", "[124,377,262,90,54,259]", "count", "04", "04_lpa/K_6/profile_sizes.csv")
add("classification", "mean_max_posterior", 0.8948, "probability", "13", "05_profiles/k6_classification_quality.csv")
add("classification", "pct_maxprob_>=0.80", 88.85, "percent", "13", "05_profiles/k6_classification_quality.csv")
add("predictor", "n_tests", 65, "tests", "18", "06_predictors/multinomial_results.csv")
add("predictor", "n_FDR_significant", 21, "tests", "18", "06_predictors/phase18_master.csv")
add("predictor", "McFadden_R2", 0.3035, "R2", "18", "06_predictors/phase18_master.csv")
add("predictor", "max_VIF", 67.44, "VIF", "18B", "06_predictors/vif_full.csv")

pd.DataFrame(rows).to_csv("results/paper1_final/09_master/paper1_master_evidence.csv", index=False)
print("master:", len(rows), "rows")
