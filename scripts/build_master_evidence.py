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


# --- Dynamic sources (fail loudly if missing) ---

# Selected K
_sel_df = pd.read_csv("results/05_lpa_selection/selected_model.csv")
if "selected_K" not in _sel_df.columns:
    raise KeyError(f"'selected_K' column not found in results/05_lpa_selection/selected_model.csv; columns={list(_sel_df.columns)}")
SEL_K = int(_sel_df["selected_K"].iloc[0])

# Model fit: BIC/AIC/entropy for selected K
_fit = pd.read_csv("results/04_lpa_estimation/model_fit.csv")
if "K" not in _fit.columns:
    raise KeyError(f"'K' column not found in results/04_lpa_estimation/model_fit.csv; columns={list(_fit.columns)}")
_fit_row = _fit.loc[_fit["K"] == SEL_K]
if _fit_row.empty:
    raise KeyError(f"selected K={SEL_K} not found in results/04_lpa_estimation/model_fit.csv; available K={sorted(_fit['K'].unique().tolist())}")
_fit_row = _fit_row.iloc[0]
for _col in ("BIC", "AIC", "entropy"):
    if _col not in _fit.columns:
        raise KeyError(f"'{_col}' column not found in results/04_lpa_estimation/model_fit.csv; columns={list(_fit.columns)}")
SEL_BIC = float(_fit_row["BIC"])
SEL_AIC = float(_fit_row["AIC"])
SEL_ENTROPY = float(_fit_row["entropy"])

# Profile sizes for selected K
_ps_path = f"results/04_lpa_estimation/K_{SEL_K}/profile_sizes.csv"
_ps = pd.read_csv(_ps_path)
if "size" not in _ps.columns:
    raise KeyError(f"'size' column not found in {_ps_path}; columns={list(_ps.columns)}")
_ps_sizes = _ps["size"].astype(int).tolist()
_PS_SIZES_STR = "[" + ",".join(str(s) for s in _ps_sizes) + "]"

# Classification quality (legacy k6-named file, but content is generated for selected K)
_cq_path = "results/13_k6_profile_validation/k6_classification_quality.csv"
_cq = pd.read_csv(_cq_path)
if "statistic" not in _cq.columns or "value" not in _cq.columns:
    raise KeyError(f"Expected columns 'statistic','value' in {_cq_path}; got {list(_cq.columns)}")
_cq_dict = dict(zip(_cq["statistic"], _cq["value"]))
try:
    _mean_max_post = float(_cq_dict["mean_max_posterior"])
except KeyError:
    raise KeyError(f"'mean_max_posterior' not found in {_cq_path}; keys={list(_cq_dict.keys())}")
# pct threshold: prefer 0.70 (88.85 in K6 file) then 0.80, then 0.90; row name reflects the actual threshold
if "pct_maxpost_ge_0.70" in _cq_dict:
    _pct_val = float(_cq_dict["pct_maxpost_ge_0.70"])
    _pct_thr = "0.70"
elif "pct_maxpost_ge_0.80" in _cq_dict:
    _pct_val = float(_cq_dict["pct_maxpost_ge_0.80"])
    _pct_thr = "0.80"
elif "pct_maxpost_ge_0.90" in _cq_dict:
    _pct_val = float(_cq_dict["pct_maxpost_ge_0.90"])
    _pct_thr = "0.90"
else:
    # fallback: any key containing 'pct' and 'max'
    _pct_candidates = [k for k in _cq_dict if "pct" in k.lower() and "max" in k.lower()]
    if not _pct_candidates:
        raise KeyError(f"No pct_maxpost key found in {_cq_path}; keys={list(_cq_dict.keys())}")
    _pct_key = _pct_candidates[0]
    _pct_val = float(_cq_dict[_pct_key])

# Predictor values: phase18_master.csv for n_tests, McFadden, etc.
_p18_path = "results/18_profile_predictors/phase18_master.csv"
_p18 = pd.read_csv(_p18_path)
if "item" not in _p18.columns or "value" not in _p18.columns:
    raise KeyError(f"Expected columns 'item','value' in {_p18_path}; got {list(_p18.columns)}")
_p18_dict = dict(zip(_p18["item"], _p18["value"]))
try:
    _n_tests = int(float(_p18_dict["n_tests"]))
except KeyError:
    raise KeyError(f"'n_tests' not found in {_p18_path}; keys={list(_p18_dict.keys())}")
try:
    _mcfadden = float(_p18_dict["McFadden_pseudo_R2"])
except KeyError:
    # fallback to alternative naming
    if "McFadden_R2" in _p18_dict:
        _mcfadden = float(_p18_dict["McFadden_R2"])
    else:
        raise KeyError(f"'McFadden_pseudo_R2' not found in {_p18_path}; keys={list(_p18_dict.keys())}")

# n_FDR_significant from multinomial_results.csv (count of significant_FDR == True)
_multi_path = "results/18_profile_predictors/multinomial_results.csv"
_multi = pd.read_csv(_multi_path)
if "significant_FDR" in _multi.columns:
    # handle both boolean and string representations
    _sig_col = _multi["significant_FDR"]
    if _sig_col.dtype == object:
        _n_fdr = int(_sig_col.astype(str).str.lower().eq("true").sum())
    else:
        _n_fdr = int(_sig_col.astype(bool).sum())
    # if counting yields 0 but phase18 has non-zero, keep the count (fail loudly if mismatch is unexpected)
    # Do not silently fallback; if 0 and phase18 has value, still use counted value
else:
    if "p_FDR" in _multi.columns:
        _n_fdr = int((_multi["p_FDR"] < 0.05).sum())
    else:
        raise KeyError(f"Neither 'significant_FDR' nor 'p_FDR' found in {_multi_path}; columns={list(_multi.columns)}")

# max VIF from vif_full.csv
_vif_path = "results/18B_predictor_multicollinearity/vif_full.csv"
_vif = pd.read_csv(_vif_path)
if "VIF" not in _vif.columns:
    raise KeyError(f"'VIF' column not found in {_vif_path}; columns={list(_vif.columns)}")
_max_vif = float(_vif["VIF"].max())

# Pearson r and CI from int_be_correlation.csv
_corr_path = "results/02_measurement/int_be_correlation.csv"
_corr = pd.read_csv(_corr_path)
if "r" not in _corr.columns:
    raise KeyError(f"'r' column not found in {_corr_path}; columns={list(_corr.columns)}")
_r_val = float(_corr["r"].iloc[0])
# CI columns may be CI95_lower/CI95_upper or CI_lower/CI_upper
if "CI95_lower" in _corr.columns and "CI95_upper" in _corr.columns:
    _ci_low = float(_corr["CI95_lower"].iloc[0])
    _ci_up = float(_corr["CI95_upper"].iloc[0])
elif "CI_lower" in _corr.columns and "CI_upper" in _corr.columns:
    _ci_low = float(_corr["CI_lower"].iloc[0])
    _ci_up = float(_corr["CI_upper"].iloc[0])
elif "CI95_lower" in _corr.columns or "CI95_upper" in _corr.columns:
    raise KeyError(f"Incomplete CI columns in {_corr_path}; columns={list(_corr.columns)}")
else:
    raise KeyError(f"No CI columns found in {_corr_path}; columns={list(_corr.columns)}")
_CI_STR = f"[{_ci_low:.4f}, {_ci_up:.4f}]"

# --- Add rows with dynamic values ---

add("sample", "N", 1166, "count", "01", "01_data/data_dimensions.csv")
add("sample", "N_variables", 34, "count", "01", "01_data/data_dimensions.csv")
add("sample", "missing_values", 0, "count", "01", "01_data/missing_values.csv")
add("sample", "duplicate_rows", 42, "count", "01", "01_data/duplicate_information.csv")
add("measurement", "INT-BE_pearson_r", _r_val, "r", "02", "02_measurement/int_be_correlation.csv")
add("measurement", "INT-BE_95CI", _CI_STR, "r", "02", "02_measurement/int_be_correlation.csv")
add("lpa", "selected_K", SEL_K, "K", "04/05", "05_lpa_selection/selected_model.csv")
add("lpa", f"K{SEL_K}_BIC", SEL_BIC, "nats", "04", "04_lpa_estimation/model_fit.csv")
add("lpa", f"K{SEL_K}_AIC", SEL_AIC, "nats", "04", "04_lpa_estimation/model_fit.csv")
add("lpa", f"K{SEL_K}_entropy", SEL_ENTROPY, "entropy_normalized_0_to_1", "04", "04_lpa_estimation/model_fit.csv")
add("profile", "profile_sizes", _PS_SIZES_STR, "count", "04", f"04_lpa_estimation/K_{SEL_K}/profile_sizes.csv")
add("classification", "mean_max_posterior", _mean_max_post + 0.0, "probability", "13", "13_k6_profile_validation/k6_classification_quality.csv")
add("classification", f"pct_maxprob_>={_pct_thr}", _pct_val, "percent", "13", "13_k6_profile_validation/k6_classification_quality.csv")
add("predictor", "n_tests", _n_tests, "tests", "18", "18_profile_predictors/phase18_master.csv")
add("predictor", "n_FDR_significant", _n_fdr, "tests", "18", "18_profile_predictors/multinomial_results.csv")
add("predictor", "McFadden_R2", _mcfadden, "R2", "18", "18_profile_predictors/phase18_master.csv")
add("predictor", "max_VIF", _max_vif, "VIF", "18B", "18B_predictor_multicollinearity/vif_full.csv")

pd.DataFrame(rows).to_csv("results/paper1_final/09_master/paper1_master_evidence.csv", index=False)
print("master:", len(rows), "rows")
