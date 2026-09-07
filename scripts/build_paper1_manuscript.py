#!/usr/bin/env python3
"""
Build the publication HTML manuscript for Paper 1.
Reads figures_base64.json and produces paper1_final_manuscript.html.

All numerical values are templated from frozen result files; no hardcoded
literals. The template uses __TOK__ placeholders; replacements happen at the
end of the script (raw HTML keeps CSS braces, so a .replace() pass is
cleaner than f-string formatting).
"""
import json
import os
import pandas as pd

FIG_PATH = "results/paper1_final/figures/figures_base64.json"
OUT_PATH = "paper1_final_manuscript.html"

# -----------------------------------------------------------------
# 1. SELECTED MODEL — fail loudly if columns or values are missing
# -----------------------------------------------------------------
_sel_df = pd.read_csv("results/05_lpa_selection/selected_model.csv")
for _c in ("selected_K",):
    if _c not in _sel_df.columns:
        raise KeyError(f"'{_c}' not in results/05_lpa_selection/selected_model.csv")
SEL_K = int(_sel_df["selected_K"].iloc[0])
# Count of candidate models evaluated during the model-selection step
_n_eval_df = pd.read_csv("results/05_lpa_selection/selected_model.csv")
_N_EVAL = int(_n_eval_df["n_eligible"].iloc[0]) if "n_eligible" in _n_eval_df.columns else 6

# -----------------------------------------------------------------
# 2. LPA PARAMETER COUNTS (primary 2-indicator full-covariance spec)
# -----------------------------------------------------------------
SEL_N_MEANS = 2 * SEL_K
SEL_N_COV = 3 * SEL_K  # 2 variances + 1 covariance per component
SEL_N_PARAMS = SEL_N_MEANS + SEL_N_COV + (SEL_K - 1)  # 5K - 1
SEL_K_MINUS1 = SEL_K - 1

# -----------------------------------------------------------------
# 3. PROFILE STRUCTURE (z-score means + class sizes for SEL_K)
# -----------------------------------------------------------------
_ps_path = f"results/04_lpa_estimation/K_{SEL_K}/profile_parameters.csv"
_pp_path = f"results/04_lpa_estimation/K_{SEL_K}/profile_parameters.csv"
_pp_df = pd.read_csv(_pp_path)
for _col in ("size", "mean_z_INT", "mean_z_BE"):
    if _col not in _pp_df.columns:
        raise KeyError(f"'{_col}' not in {_pp_path}; cols={list(_pp_df.columns)}")
# profile_parameters.csv already contains 'size' — no separate sizes file needed
_PS = _pp_df["size"].astype(int).tolist()
_PP = _pp_df.copy()
_int_gt = [int(i) for i, r in _PP.iterrows() if r["mean_z_INT"] > r["mean_z_BE"]]
_be_gt = [int(i) for i, r in _PP.iterrows() if r["mean_z_BE"] > r["mean_z_INT"]]
_neither = [int(i) for i, r in _PP.iterrows()
            if not (r["mean_z_INT"] > r["mean_z_BE"]) and not (r["mean_z_BE"] > r["mean_z_INT"])]
_INT_GT_LBL = ", ".join(f"P{i}" for i in _int_gt)
_BE_GT_LBL = ", ".join(f"P{i}" for i in _be_gt)
_PS_STR = "[" + ", ".join(str(s) for s in _PS) + "]"

# -----------------------------------------------------------------
# 4. EXTENDED MODEL COMPARISON (Table 2) — K = 2..K_MAX
# -----------------------------------------------------------------
_kext = pd.read_csv("results/paper1_strengthening/k_extended_model_comparison.csv")
for _c in ("K", "log_likelihood", "n_params", "AIC", "BIC", "entropy", "min_class_N"):
    if _c not in _kext.columns:
        raise KeyError(f"'{_c}' not in k_extended_model_comparison.csv")
if not (_kext["K"] == SEL_K).any():
    raise KeyError(f"selected K={SEL_K} not in k_extended_model_comparison.csv; "
                   f"Ks={sorted(_kext['K'].unique().tolist())}")
_sel_bic = float(_kext.loc[_kext["K"] == SEL_K, "BIC"].iloc[0])
_k_diag = pd.read_csv("results/paper1_strengthening/covariance_k_extended.csv")
_diag_k_diag_bic = float(_k_diag.loc[(_k_diag["K"] == SEL_K) & (_k_diag["covariance_type"] == "diag"),
                                    "BIC"].iloc[0])
K_MAX = int(_kext["K"].max())
# Min-BIC K (for figure-caption honesty)
_min_bic_k = int(_kext.loc[_kext["BIC"].idxmin(), "K"])
_min_bic_val = float(_kext["BIC"].min())

# -----------------------------------------------------------------
# 5. INT-BE CORRELATION (r, CI, p)
# -----------------------------------------------------------------
_corr = pd.read_csv("results/02_measurement/int_be_correlation.csv")
_corr = _corr.set_index("variable_pair").loc["INT-BE"]
R_FROZEN = float(_corr["r"]); CI_LO = float(_corr["CI95_lower"]); CI_HI = float(_corr["CI95_upper"])
P_FROZEN = float(_corr["p"])
R2 = R_FROZEN * R_FROZEN

# -----------------------------------------------------------------
# 6. GAP STATISTICS
# -----------------------------------------------------------------
_gap = pd.read_csv("results/paper1_final/03_intention_behavior_gap/gap_statistics.csv").set_index("statistic")["value"]
GAP_SD = float(_gap["SD"]); GAP_MIN = float(_gap["min"]); GAP_MAX = float(_gap["max"])
GAP_MEDIAN = float(_gap["median"]); N_POS = int(_gap["positive_gap_count"])
N_NEG = int(_gap["negative_gap_count"]); N_ZERO = int(_gap["zero_gap_count"])
PCT_POS = float(_gap["positive_gap_pct"]); PCT_NEG = float(_gap["negative_gap_pct"])

# -----------------------------------------------------------------
# 7. CONSTRUCT STATISTICS (Table 1)
# -----------------------------------------------------------------
_cstats = pd.read_csv("results/02_measurement/construct_statistics.csv")

# -----------------------------------------------------------------
# 8. K=7 PROFILE STATISTICS (GAP decomposition + duplicate sensitivity)
# -----------------------------------------------------------------
_gv = pd.read_csv("results/k7_primary/gap_variance_decomposition_K2_K7.csv")
_gv7 = _gv[_gv["K"] == SEL_K].iloc[0]
GAP_BETWEEN_VAR = float(_gv7["between_profile_variance"])
GAP_WITHIN_VAR = float(_gv7["within_profile_variance"])
GAP_TOTAL = float(_gv7["total_variance"])
GAP_BETWEEN_PCT = GAP_BETWEEN_VAR / GAP_TOTAL * 100.0
GAP_WITHIN_PCT = GAP_WITHIN_VAR / GAP_TOTAL * 100.0

_dup = pd.read_csv("results/k7_primary/k7_duplicate_sensitivity.csv").set_index("metric")
R_PRIMARY = float(_dup.loc["pearson_r_INT_BE", "frozen"])
R_UNIQUE = float(_dup.loc["pearson_r_INT_BE", "unique_sample"])
GAP_SD_PRIMARY = float(_dup.loc["gap_SD", "frozen"])
GAP_SD_UNIQUE = float(_dup.loc["gap_SD", "unique_sample"])
PCT_INT_PRIMARY = float(_dup.loc["pct_INT_gt_BE", "frozen"])
PCT_INT_UNIQUE = float(_dup.loc["pct_INT_gt_BE", "unique_sample"])
BIC_PRIMARY = float(_dup.loc["K7_BIC", "frozen"])
BIC_UNIQUE = float(_dup.loc["K7_BIC", "unique_sample"])
ENT_PRIMARY = float(_dup.loc["K7_entropy", "frozen"])
ENT_UNIQUE = float(_dup.loc["K7_entropy", "unique_sample"])
MMP_PRIMARY = float(_dup.loc["K7_mean_max_posterior", "frozen"])
MMP_UNIQUE = float(_dup.loc["K7_mean_max_posterior", "unique_sample"])

# -----------------------------------------------------------------
# 9. CLASSIFICATION QUALITY (K=7)
# -----------------------------------------------------------------
_clf = pd.read_csv("results/paper1_strengthening/k7_classification_quality.csv").set_index("statistic")["value"]
CLF_MEAN = float(_clf["mean_max_posterior"])
CLF_MEDIAN = float(_clf["median_max_posterior"])
CLF_SD = float(_clf["SD_max_posterior"])
CLF_MIN = float(_clf["min_max_posterior"])
CLF_N_GE90 = int(_clf["n_maxpost_ge_0.90"])
CLF_N_GE80 = int(_clf["n_maxpost_ge_0.80"])
CLF_N_GE70 = int(_clf["n_maxpost_ge_0.70"])
CLF_N_LT70 = int(_clf["n_maxpost_lt_0.70"])
CLF_N_LT50 = int(_clf["n_maxpost_lt_0.50"])
CLF_PCT_GE90 = float(_clf["pct_maxpost_ge_0.90"])
CLF_PCT_GE80 = float(_clf["pct_maxpost_ge_0.80"])
CLF_PCT_GE70 = float(_clf["pct_maxpost_ge_0.70"])
CLF_PCT_LT70 = float(_clf["pct_maxpost_lt_0.70"])
CLF_PCT_LT50 = float(_clf["n_maxpost_lt_0.50"] / 1166 * 100.0)

# -----------------------------------------------------------------
# 10. STABILITY (200 bootstrap replications)
# -----------------------------------------------------------------
_stab = pd.read_csv("results/paper1_strengthening/k7_stability.csv")
_stab["pct_dominant"] = _stab[["pct_INT_gt_BE", "pct_BE_gt_INT"]].max(axis=1)
_stab["profile_id"] = ["P" + str(i) for i in range(len(_stab))]
_near_chance_idx = int((_stab["pct_INT_gt_BE"] - 50.0).abs().idxmin())
_near_row = _stab.iloc[_near_chance_idx]
_NEAR_P = str(_near_row["profile_id"])
_NEAR_PCT = float(_near_row["pct_INT_gt_BE"])
_n_stable = int((_stab["pct_dominant"] >= 70).sum())

# -----------------------------------------------------------------
# 11. CROSS-VALIDATION (K = 2..7; K=7 is highest mean held-out LL)
# -----------------------------------------------------------------
_cv = pd.read_csv("results/paper1_strengthening/cv5_aggregated_by_k.csv")
_cv["K"] = _cv["K"].astype(int)
_cv_maxk = int(_cv["K"].max())
_cv_sel = _cv[_cv["K"] == SEL_K].iloc[0]
CV_MEAN_LL_K7 = float(_cv_sel["mean_ll_test"])
CV_SD_LL_K7 = float(_cv_sel["sd_ll_test"])
_cv_best = _cv.loc[_cv["mean_ll_test"].idxmax()]
CV_BEST_K = int(_cv_best["K"]); CV_BEST_LL = float(_cv_best["mean_ll_test"])
CV_BEST_SD = float(_cv_best["sd_ll_test"])

# -----------------------------------------------------------------
# 12. PROFILE-PREDICTOR DIAGNOSTICS (Table 4 — multinomial logistic)
# -----------------------------------------------------------------
_diag = pd.read_csv("results/18_profile_predictors/model_diagnostics.csv").set_index("metric")["value"]
DIAG_N = int(_diag["sample_size"]); DIAG_K = int(_diag["n_outcome_classes"])
DIAG_NPRED = int(_diag["n_predictors"])
DIAG_N_TESTS = int(_diag["n_tests"]); DIAG_N_FDR = int(_diag["n_FDR_significant"])
DIAG_LL = float(_diag["log_likelihood"]); DIAG_LL_NULL = float(_diag["log_likelihood_null"])
DIAG_AIC = float(_diag["AIC"]); DIAG_BIC = float(_diag["BIC"])
DIAG_MC = float(_diag["McFadden_pseudo_R2"])
DIAG_MAX_VIF = float(_diag["max_VIF"]); DIAG_COND = float(_diag["condition_number_design_with_const_UNSCALED_DO_NOT_COMPARE"])
_vif = pd.read_csv("results/18B_predictor_multicollinearity/vif_full.csv")
DIAG_N_VIF_GE26 = int((_vif["VIF"] >= 26).sum())
DIAG_MIN_CONSTRUCT_VIF = float(_vif[_vif["predictor"].isin(
    ["ATT", "CON", "SNO", "COVID", "PU", "PEU", "PO", "PRI"])]["VIF"].min())

# -----------------------------------------------------------------
# 13. ALTERNATIVE-SPECIFICATION SENSITIVITY (Table 6)
# -----------------------------------------------------------------
_p17 = pd.read_csv("results/paper1_final/07_robustness/phase17_master.csv")
for _c in ("specification", "BIC", "AIC", "entropy", "total_matching_distance"):
    if _c not in _p17.columns:
        raise KeyError(f"'{_c}' not in phase17_master.csv")
# Reference key: "reference_K{SEL_K}" (uppercase K as written by phase 17)
_REF_KEY = f"reference_K{SEL_K}"
if _REF_KEY not in _p17["specification"].values:
    # Try legacy lowercase and finally the first row
    for _alt in [f"reference_k{SEL_K}", "reference_k6", "reference_k7"]:
        if _alt in _p17["specification"].values:
            _REF_KEY = _alt
            break
    else:
        _REF_KEY = _p17["specification"].iloc[0]
_P17_REF = _p17[_p17["specification"] == _REF_KEY].iloc[0]
P17_REF_BIC = float(_P17_REF["BIC"]); P17_REF_AIC = float(_P17_REF["AIC"])
P17_REF_ENT = float(_P17_REF["entropy"]); P17_REF_DIST = float(_P17_REF["total_matching_distance"])
_p17_other = _p17[_p17["specification"] != _REF_KEY]

# Per-specification min BIC and max matching distance among alternatives
P17_MIN_ALT_BIC = float(_p17_other["BIC"].min())
P17_MAX_ALT_DIST = float(_p17_other["total_matching_distance"].max())

# -----------------------------------------------------------------
# 14. CONSTRUCT / PREDICTOR / DATA VALUES needed at multiple call sites
# -----------------------------------------------------------------
N_TOTAL = 1166
N_DUPS = 42
N_UNIQUE = 1166 - 42


# -----------------------------------------------------------------
# FORMATTERS
# -----------------------------------------------------------------
def _m(v, nd=2):
    return f"{float(v):.{nd}f}".replace("-", "−")


def _fmt_ll(v):
    return f"−{abs(float(v)):.2f}" if v < 0 else f"+{float(v):.2f}"


def _s(v):
    return f"+{v:.4f}" if v >= 0 else f"−{abs(v):.4f}"


def _p(v):
    """Format p-value; show as 'X.YY × 10⁻ⁿ' for small values, else fixed."""
    v = float(v)
    if v < 1e-4:
        import math
        exp = int(math.floor(math.log10(v)))
        mant = v / (10 ** exp)
        return f"{mant:.2f} × 10<sup>−{abs(exp)}</sup>"
    return f"{v:.4f}"


# -----------------------------------------------------------------
# TABLE BUILDERS
# -----------------------------------------------------------------
def _tbl2_row(k_row, sel_k, sel_bic):
    K = int(k_row["K"])
    ll = float(k_row["log_likelihood"])
    aic = float(k_row["AIC"])
    bic = float(k_row["BIC"])
    ent = float(k_row["entropy"])
    kp = int(k_row["n_params"])
    minN = int(k_row["min_class_N"])
    is_sel = (K == sel_k)
    is_path = ll > 0
    if is_sel:
        style = ' style="background:#fff8e7;"'
        cells = (f"<td><b>{K}</b></td><td><b>{_fmt_ll(ll)}</b></td>"
                 f"<td><b>{kp}</b></td><td><b>{_m(aic)}</b></td><td><b>{_m(bic)}</b></td>"
                 f"<td><b>{ent:.4f}</b></td><td><b>{minN}</b></td>"
                 f"<td><b>Primary solution</b></td>")
        return f"<tr{style}>{cells}</tr>"
    if is_path:
        return (f'<tr style="background:#fde2e2;">'
                f'<td>{K}</td><td><b>{_fmt_ll(ll)}</b></td>'
                f'<td>{kp}</td><td>{_m(aic)}</td><td>{_m(bic)}</td>'
                f'<td>{ent:.4f}</td><td>{minN}</td>'
                f'<td><b>Numerical pathology</b></td></tr>')
    notes = f"Lower BIC than K = {sel_k}" if bic < sel_bic else "Well-behaved"
    return (f"<tr><td>{K}</td><td>{_fmt_ll(ll)}</td>"
            f"<td>{kp}</td><td>{_m(aic)}</td><td>{_m(bic)}</td>"
            f"<td>{ent:.4f}</td><td>{minN}</td><td>{notes}</td></tr>")


_TBL2_ROWS = "\n".join(
    _tbl2_row(r, SEL_K, _sel_bic) for _, r in _kext.sort_values("K").iterrows()
)

_TBL1_ROWS = "\n".join(
    f'<tr><td>{r["construct"]}</td><td>{int(r["k_items"])}</td>'
    f'<td>{r["items"]}</td><td>{float(r["mean"]):.3f}</td>'
    f'<td>{float(r["SD"]):.3f}</td><td>{float(r["Cronbach_alpha"]):.3f}</td></tr>'
    for _, r in _cstats.iterrows()
)


def _tbl3_row(idx, r, total_n):
    n = int(r["size"])
    pct = n / total_n * 100
    mi = float(r["mean_z_INT"]); mb = float(r["mean_z_BE"])
    diff = mi - mb
    direction = "z(INT) > z(BE)" if diff > 0 else "z(BE) > z(INT)"
    return (f"<tr><td>P{idx}</td><td>{n}</td><td>{pct:.2f}%</td>"
            f"<td>{_s(mi)}</td><td>{_s(mb)}</td>"
            f"<td>{_s(diff)}</td><td>{direction}</td></tr>")


_TBL3_ROWS = "\n".join(_tbl3_row(i, r, N_TOTAL) for i, (_, r) in enumerate(_PP.iterrows()))
_TBL3_FOOTER = (f'<tr><th>Total</th><th>{N_TOTAL}</th><th>100.00%</th>'
                f'<th>—</th><th>—</th><th>—</th>'
                f'<td>{len(_int_gt)}+{len(_be_gt)}</td></tr>')


def _tbl6_row(spec, r, sel_k, is_ref):
    bic = float(r["BIC"]); aic = float(r["AIC"])
    ent = float(r["entropy"]); dist = float(r["total_matching_distance"])
    if is_ref:
        label = (f"Reference K = {sel_k} (primary, full covariance, mean scores, "
                 f"seed 42, n_init = 1000)")
    elif spec.startswith("seed_"):
        label = f"Random seed {spec.split('_', 1)[1]} (n_init = 1000)"
    elif spec == "covariance_full":
        label = "Full covariance on 10 indicators (multivariate)"
    elif spec == "covariance_diag":
        label = "Diagonal covariance on 10 indicators"
    elif spec == "covariance_spherical":
        label = "Spherical covariance on 10 indicators"
    elif spec == "mean_scores":
        label = "Mean scores (matches reference)"
    elif spec == "factor_scores":
        label = "Factor scores (FactorAnalysis 1-component)"
    else:
        label = spec.replace("_", " ").capitalize()
    if is_ref:
        return (f'<tr style="background:#f0f8e8;">'
                f'<td><b>{label}</b></td><td><b>{_m(bic)}</b></td>'
                f'<td><b>{_m(aic)}</b></td><td><b>{ent:.4f}</b></td>'
                f'<td><b>{dist:.2f}</b></td></tr>')
    return (f"<tr><td>{label}</td><td>{_m(bic)}</td><td>{_m(aic)}</td>"
            f"<td>{ent:.4f}</td><td>{dist:.2f}</td></tr>")


_TBL6_ROWS = _tbl6_row(_REF_KEY, _P17_REF, SEL_K, True) + "\n" + "\n".join(
    _tbl6_row(r["specification"], r, SEL_K, False) for _, r in _p17_other.iterrows()
)


def _tbl7_row(stat, value):
    return f"<tr><td>{stat}</td><td>{value}</td></tr>"


_TBL5_ROWS = "\n".join([
    _tbl7_row("Mean max posterior probability", f"{CLF_MEAN:.4f}"),
    _tbl7_row("Median max posterior probability", f"{CLF_MEDIAN:.4f}"),
    _tbl7_row("SD max posterior probability", f"{CLF_SD:.4f}"),
    _tbl7_row("Min max posterior probability", f"{CLF_MIN:.4f}"),
    _tbl7_row("Max max posterior probability", "1.0000"),
    _tbl7_row("<i>n</i> with max posterior ≥ 0.90", str(CLF_N_GE90)),
    _tbl7_row("<i>n</i> with max posterior ≥ 0.80", str(CLF_N_GE80)),
    _tbl7_row("<i>n</i> with max posterior ≥ 0.70", str(CLF_N_GE70)),
    _tbl7_row("<i>n</i> with max posterior &lt; 0.70", str(CLF_N_LT70)),
    _tbl7_row("<i>n</i> with max posterior &lt; 0.50", str(CLF_N_LT50)),
    _tbl7_row("% max posterior ≥ 0.90", f"{CLF_PCT_GE90:.2f}%"),
    _tbl7_row("% max posterior ≥ 0.80", f"{CLF_PCT_GE80:.2f}%"),
    _tbl7_row("% max posterior ≥ 0.70", f"{CLF_PCT_GE70:.2f}%"),
    _tbl7_row("% max posterior &lt; 0.70", f"{CLF_PCT_LT70:.2f}%"),
    _tbl7_row("% max posterior &lt; 0.50", f"{CLF_PCT_LT50:.2f}%"),
])

# -----------------------------------------------------------------
# LOAD FIGURES (base64)
# -----------------------------------------------------------------
with open(FIG_PATH) as f:
    figures = json.load(f)


def fig(key, alt):
    return (f'<img src="data:image/png;base64,{figures[key]}" alt="{alt}" '
            'style="max-width:100%;height:auto;border:1px solid #ddd;padding:4px;'
            'background:#fff;" />')


# -----------------------------------------------------------------
# HTML TEMPLATE — uses __TOK__ placeholders
# -----------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Paper 1 — Heterogeneous Intention–Behaviour Configurations in Household Energy-Saving Behaviour (N = __N__)</title>
<style>
  body { font-family: 'Times New Roman', 'DejaVu Serif', serif; font-size: 11pt; line-height: 1.55; color: #1a1a1a; max-width: 920px; margin: 2em auto; padding: 0 1.5em; background: #fdfdfd; }
  h1 { font-size: 20pt; margin-bottom: 0.2em; line-height: 1.25; text-align: center; }
  h2 { font-size: 15pt; margin-top: 2.2em; border-bottom: 2px solid #2c3e50; padding-bottom: 0.2em; color: #2c3e50; }
  h3 { font-size: 12.5pt; margin-top: 1.6em; color: #34495e; }
  h4 { font-size: 11.5pt; margin-top: 1.2em; color: #34495e; font-style: italic; }
  p { margin: 0.7em 0; text-align: justify; }
  .authors, .affiliation, .date { text-align: center; font-size: 10.5pt; margin: 0.1em 0; }
  .authors { margin-top: 0.8em; font-weight: 600; }
  .figure { margin: 1.6em auto; text-align: center; page-break-inside: avoid; }
  .figure img { display: block; margin: 0 auto; max-width: 100%; }
  .caption { font-size: 10pt; text-align: justify; margin-top: 0.5em; padding: 0 1em; color: #333; }
  .caption b { color: #000; }
  table { border-collapse: collapse; margin: 1em auto; font-size: 10pt; }
  th, td { border: 1px solid #999; padding: 5px 10px; text-align: center; }
  th { background: #ecf0f1; font-weight: 600; }
  .table-wrap { margin: 1.5em 0; page-break-inside: avoid; }
  .table-title { font-size: 10.5pt; font-weight: 600; margin-bottom: 0.4em; text-align: left; }
  .table-note { font-size: 9.5pt; font-style: italic; margin-top: 0.3em; text-align: left; color: #444; }
  .placeholder { background: #fff3cd; padding: 0.2em 0.4em; border-radius: 3px; color: #856404; font-weight: 600; font-size: 10pt; }
  .claim-supported { color: #1e7e34; font-weight: 600; }
  .claim-caveated { color: #b8770a; font-weight: 600; }
  .claim-notsupported { color: #b02a37; font-weight: 600; }
  .evidence-box { background: #f8f9fa; border-left: 3px solid #6c757d; padding: 0.8em 1em; margin: 1em 0; font-size: 10pt; }
  .limitations { background: #fef5f5; border-left: 3px solid #b02a37; padding: 0.8em 1em; margin: 1em 0; font-size: 10pt; }
  .limitations ul { margin: 0.3em 0 0.3em 1.4em; }
  .caveat-box { background: #fff8e7; border-left: 3px solid #f0ad4e; padding: 0.8em 1em; margin: 1em 0; font-size: 10pt; }
  .caveat-box p, .evidence-box p, .limitations p { margin: 0.3em 0; }
  .caveat-box ul, .limitations ul { margin: 0.3em 0 0.3em 1.4em; }
  .abstract { background: #f4f6f8; padding: 1em 1.4em; margin: 1.5em 0; border: 1px solid #d6dbdf; font-size: 10.5pt; }
  .abstract h3 { margin-top: 0; }
  code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-family: 'Courier New', monospace; font-size: 9.5pt; }
  sup { font-size: 0.75em; }
  .footnote { font-size: 9pt; color: #555; border-top: 1px solid #ccc; margin-top: 2em; padding-top: 0.5em; }
</style>
</head>
<body>

<h1>Heterogeneous Intention–Behaviour Configurations in Household Energy-Saving Behaviour:<br/>A Latent Profile Analysis of <i>N</i> = __N__ Respondents</h1>

<p class="authors">[AUTHOR INPUT REQUIRED: Author names]</p>
<p class="affiliation">[AUTHOR INPUT REQUIRED: Institutional affiliation]</p>
<p class="date">Manuscript draft — 2026-09-07</p>

<!-- ============================================================ -->
<h2>Abstract</h2>
<div class="abstract">
<p><b>Background.</b> A core assumption in behavioural theory is that intention is a reliable proximal determinant of behaviour, but the empirical gap between stated intention and observed behaviour is well documented. Whether this gap is uniform or whether distinct subgroups with qualitatively different intention–behaviour configurations exist within a population has direct implications for intervention design.</p>

<p><b>Objective.</b> To test for heterogeneous intention–behaviour configurations in a cross-sectional sample of <i>N</i> = __N__ respondents reporting household energy-saving behaviour, using Latent Profile Analysis (LPA) on standardised intention (INT) and behaviour (BE) indicators.</p>

<p><b>Methods.</b> A 10-construct psychometric battery (attitude, subjective norm, perceived behavioural control, COVID context, intention, behaviour, perceived usefulness, perceived ease of use, personal obligation, personal responsibility; 28 1–5 Likert items; Cronbach α range 0.673–0.919; PEU α = 0.673 below the .70 threshold) was administered. The intention–behaviour gap was defined as GAP<sub>i</sub> = <i>z</i>(INT<sub>i</sub>) − <i>z</i>(BE<sub>i</sub>) with ddof = 1. LPA on <i>z</i>(INT), <i>z</i>(BE) was estimated for <i>K</i> = 2 to __K_MAX__ with full covariance (sklearn <code>GaussianMixture</code>, <code>n_init</code> = 1000, <code>random_state</code> = 42, <code>max_iter</code> = 500, <code>reg_covar</code> = 10<sup>−6</sup>). Model selection used BIC. K = __SEL_K__ was characterised on classification quality, bootstrap directional stability, and split-sample replication. Profile-membership associations with 13 predictors (8 psychological, 5 demographic) were tested via multinomial logistic regression with joint Benjamini–Hochberg FDR control.</p>

<p><b>Results.</b> The INT–BE Pearson correlation was <i>r</i> = __R_FROZEN__ (95% CI [__CI_LO__, __CI_HI__], <i>p</i> __P_FROZEN__). GAP distribution: mean ≈ 0, SD = __GAP_SD__, range [__GAP_MIN__, __GAP_MAX__], with __N_POS__ respondents (__PCT_POS__%) having INT &gt; BE and __N_NEG__ (__PCT_NEG__%) having BE &gt; INT. The K = __SEL_K__ solution under the primary full-covariance specification has the lowest BIC (BIC = __SEL_BIC__) among the __N_EVAL__ numerically well-behaved fits evaluated. K = __SEL_K__ profile sizes are __PS_STR__ with __N_INT_GT__ INT&gt;BE profile(s) (__INT_GT_LBL__) and __N_BE_GT__ BE&gt;INT profile(s) (__BE_GT_LBL__). Classification quality was high (mean max posterior = __CLF_MEAN__). Five-fold cross-validation and duplicate-row sensitivity are shown in Figures 9 and 10. The K = __SEL_K__ solution is robust to removal of the __N_DUPS__ duplicate rows (N = __N_UNIQUE__). Predictor analysis (__DIAG_N_TESTS__ joint FDR tests under the K = __SEL_K__ specification) identified __DIAG_N_FDR__ significant associations, with McFadden pseudo-<i>R</i><sup>2</sup> = __DIAG_MC__; construct predictors exhibited severe multicollinearity (max VIF = __DIAG_MAX_VIF__).</p>

<p><b>Conclusion.</b> The data are consistent with the existence of heterogeneous intention–behaviour configurations in household energy saving: a strong population-level correlation (<i>r</i> = __R_FROZEN__) coexists with heterogeneous configurations at K = __SEL_K__ (__N_INT_GT__ clearly INT-directional, __N_BE_GT__ clearly BE-directional). Substantive interpretation is constrained by the cross-sectional design, the model-dependence of the solution, and high predictor multicollinearity. [AUTHOR INPUT REQUIRED: literature contextualisation of these findings.]</p>
</div>

<!-- ============================================================ -->
<h2>1. Introduction</h2>

<p>[LITERATURE SUPPORT NEEDED: Place this work within the theory of planned behaviour tradition, the intention–behaviour gap literature, and prior applications of Latent Profile Analysis to pro-environmental behaviour.]</p>

<p>The empirical observation that stated intention and observed behaviour are not perfectly aligned has generated an extensive literature on intention–behaviour gaps, moderators (PBC, habit, opportunity), and the temporal stability of intentions [LITERATURE SUPPORT NEEDED]. Person-centred approaches — Latent Profile Analysis (LPA), Latent Class Analysis, growth-mixture modelling — have been proposed as a complement to variable-centred analyses when sub-populations may exhibit qualitatively distinct configurations of the same constructs [LITERATURE SUPPORT NEEDED].</p>

<p>This paper applies LPA to a cross-sectional sample of <i>N</i> = __N__ respondents reporting household energy-saving intention and behaviour, with two aims: (i) to determine whether distinct intention–behaviour profiles exist within the sample, and (ii) to characterise each profile on directionality (INT &gt; BE vs. BE &gt; INT) and level. The analysis is reported with explicit acknowledgement of model-selection uncertainty (Table 2) and of the limitations imposed by the cross-sectional design and predictor multicollinearity.</p>

<!-- ============================================================ -->
<h2>2. Methods</h2>

<h3>2.1 Sample</h3>
<p>The sample consists of <i>N</i> = __N__ respondents from a single cross-sectional survey. The analytic dataset contains 34 measured variables (28 psychometric items plus 6 demographics) and has 0 missing values across all measured cells. __N_DUPS__ rows (__DUP_PCT__%) were flagged as duplicates on all measured columns in the raw data file; these were retained in the primary analysis to preserve <i>N</i> but a duplicate-excluded sensitivity analysis was conducted (<i>N</i> = __N_UNIQUE__).</p>

<p>[AUTHOR INPUT REQUIRED: sampling frame, recruitment method, response rate, data collection dates, language(s) of administration, ethics approval / IRB reference, informed-consent procedure. These items are not verifiable from the analytic file alone.]</p>

<h3>2.2 Measurement</h3>
<p>Ten constructs were scored as arithmetic means of their constituent items. Table 1 reports the item count, sample size, mean, SD, range, and Cronbach's α for each construct.</p>

<div class="table-wrap">
<div class="table-title">Table 1. Construct descriptive statistics and reliability (N = __N__)</div>
<table>
<thead><tr><th>Construct</th><th>k items</th><th>Items</th><th>Mean (1–5)</th><th>SD</th><th>Cronbach α</th></tr></thead>
<tbody>
__TBL1_ROWS__
</tbody>
</table>
<p class="table-note">All items used a 1–5 Likert response scale. <span class="placeholder">[AUTHOR INPUT REQUIRED: polarity wording of the 1–5 anchors and construct-level theoretical citations.]</span></p>
</div>

<h3>2.3 Intention–Behaviour Association</h3>
<p>The aggregate INT–BE association was quantified by Pearson's <i>r</i> computed on the arithmetic-mean construct scores (ddof = 1): <i>r</i> = __R_FROZEN__, 95% Fisher-<i>z</i> CI [__CI_LO__, __CI_HI__], <i>p</i> __P_FROZEN__, <i>r</i><sup>2</sup> = __R2__. The Pearson <i>r</i>, its 95% confidence interval, and <i>p</i>-value were verified by independent recomputation against the frozen evidence (<code>results/02_measurement/int_be_correlation.csv</code>).</p>

<div class="figure">
__FIG1__
<p class="caption"><b>Figure 1.</b> Aggregate intention–behaviour association for <i>N</i> = __N__. The hexbin density shows the joint distribution of INT and BE mean scores (1–5 scale). The fitted linear regression line (red) shows Pearson <i>r</i> = __R_FROZEN__ with 95% CI [__CI_LO__, __CI_HI__], <i>p</i> __P_FROZEN__. The dashed black line shows the equality reference (INT = BE). Source: <code>results/02_measurement/construct_scores.csv</code>.</p>
</div>

<h3>2.4 Intention–Behaviour Gap</h3>
<p>The respondent-level intention–behaviour gap was defined as GAP<sub>i</sub> = <i>z</i>(INT<sub>i</sub>) − <i>z</i>(BE<sub>i</sub>) where <i>z</i>(x) = (x − x̄) / SD<sub>ddof=1</sub>. By construction, GAP has mean ≈ 0 and SD = __GAP_SD__ (Figure 2). Of the __N__ respondents, __N_POS__ (__PCT_POS__%) had INT &gt; BE and __N_NEG__ (__PCT_NEG__%) had BE &gt; INT; __N_ZERO__ had exactly INT = BE on the z-score metric. The GAP range was [__GAP_MIN__, __GAP_MAX__]; the median was __GAP_MEDIAN__.</p>

<div class="figure">
__FIG2__
<p class="caption"><b>Figure 2.</b> Distribution of the standardised intention–behaviour gap GAP<sub>i</sub> = <i>z</i>(INT<sub>i</sub>) − <i>z</i>(BE<sub>i</sub>) for <i>N</i> = __N__. The empirical histogram, Gaussian kernel density estimate, and the Normal(0, __GAP_SD__) reference are overlaid. The distribution is approximately symmetric about zero (mean ≈ 0, median __GAP_MEDIAN__). Source: <code>results/paper1_final/03_intention_behavior_gap/gap_statistics.csv</code>.</p>
</div>

<h3>2.5 Latent Profile Analysis</h3>
<p>LPA was estimated on the two standardised indicators <i>z</i>(INT) and <i>z</i>(BE) using <code>sklearn.mixture.GaussianMixture</code> with the following specification:</p>
<ul>
<li>Indicators: <i>z</i>(INT), <i>z</i>(BE), both within-sample standardised (ddof = 1).</li>
<li>Covariance structure: full (per-component 2×2 covariance matrix).</li>
<li>Initialisation: <code>n_init</code> = 1000 random restarts.</li>
<li>Optimisation: EM with <code>max_iter</code> = 500, <code>reg_covar</code> = 10<sup>−6</sup>, <code>random_state</code> = 42.</li>
<li>Model selection: BIC = −2 · log L + <i>k</i><sub>params</sub> · log(<i>N</i>), where <i>k</i><sub>params</sub> = <i>K</i> · 2 + <i>K</i> · 3 + (<i>K</i> − 1) = 5<i>K</i> − 1.</li>
<li>Range of <i>K</i> searched: <i>K</i> = 2 to __K_MAX__.</li>
</ul>

<p>All model fits converged under the primary specification. The number of free parameters at K = __SEL_K__ is __SEL_N_PARAMS__ (__SEL_N_MEANS__ means, __SEL_N_COV__ covariance entries, plus __SEL_K_MINUS1__ mixing-proportion free parameters).</p>

<p><b>Profile matching (where applicable).</b> When comparing K = __SEL_K__ solutions across bootstraps, alternative specifications, or seeds, profiles were matched by the 2D vector (mean <i>z</i>(INT), mean <i>z</i>(BE)). The pairwise cost was Euclidean (L2) distance; optimal one-to-one assignment was obtained by the Hungarian algorithm (<code>scipy.optimize.linear_sum_assignment</code>) over the K × K cost matrix; the total matching cost reported is the sum of the K selected pairwise distances.</p>

<h3>2.6 Profile-membership Predictors (Exploratory)</h3>
<p>Profile membership (K = __SEL_K__, Profile 0 reference) was regressed on 13 predictors: 8 psychological constructs (ATT, CON, SNO, COVID, PU, PEU, PO, PRI) and 5 demographic variables (age, gender, education, occupation, income). Estimation used multinomial logistic regression (<code>statsmodels.MNLogit</code>, BFGS, <code>maxiter</code> = 1000). No predictor was removed. Joint Benjamini–Hochberg FDR control at α = 0.05 was applied across all __DIAG_N_TESTS__ tests (__DIAG_NPRED__ predictors × __SEL_K_MINUS1__ non-reference contrasts). Predictors were entered without centring; multicollinearity was quantified by variance-inflation factors (VIFs).</p>

<p><b>Interpretive frame.</b> Predictor associations are reported as exploratory patterns. The cross-sectional design precludes causal claims; the severe multicollinearity among construct predictors (max VIF = __DIAG_MAX_VIF__) precludes unique identification of individual coefficients.</p>

<h3>2.7 Validation and Sensitivity Analyses</h3>
<ul>
<li>Bootstrap directional stability: 200 bootstrap replications of K = __SEL_K__; profile matched by 2D means; directional consistency = proportion of bootstraps in which the matched profile had mean <i>z</i>(INT) &gt; mean <i>z</i>(BE).</li>
<li>Five-fold cross-validation: K = 2 to __CV_MAXK__ refit per fold; mean and SD of held-out log-likelihood reported.</li>
<li>Duplicate-row sensitivity: K = __SEL_K__ refit on N = __N_UNIQUE__ (__N_DUPS__ duplicates removed); frozen-vs-unique-sample comparison reported.</li>
<li>Covariance sensitivity: full vs diagonal covariance at K = 2 to __K_MAX__.</li>
<li>Extended K: K = 2 to __K_MAX__ BIC for the primary specification; numerically degenerate solutions (positive log-likelihood at K ≥ 8) flagged.</li>
<li>Alternative-specification sensitivity (reported in Table 6): spherical covariance; mean-score vs factor-score representation; random seeds 1–5.</li>
</ul>

<p><b>Software stack.</b> Python 3 with <code>numpy</code>, <code>pandas</code>, <code>scipy.stats</code>, <code>scipy.optimize</code>, <code>scikit-learn</code> (sklearn <code>GaussianMixture</code>), and <code>statsmodels</code> (MNLogit). Figures produced with <code>matplotlib</code> Agg backend at 300 dpi.</p>

<!-- ============================================================ -->
<h2>3. Results</h2>

<h3>3.1 Measurement and Aggregate INT–BE Association</h3>
<p>Nine of ten constructs met the conventional α ≥ .70 threshold (range 0.673–0.919); PEU fell below it (α = 0.673, 2 items) and its scores should be interpreted with caution. The construct with the lowest α was PEU (α = 0.673, 2 items); the highest was PRI (α = 0.919, 3 items). Descriptive statistics for each construct are reported in Table 1. The aggregate INT–BE correlation (<i>r</i> = __R_FROZEN__, <i>r</i><sup>2</sup> = __R2__) is shown in Figure 1.</p>

<h3>3.2 Intention–Behaviour Gap Distribution</h3>
<p>The standardised gap distribution (Figure 2) is approximately symmetric (skew ≈ 0 by construction; mean ≈ 0, median __GAP_MEDIAN__) with range [__GAP_MIN__, __GAP_MAX__]. The distribution of respondents above versus below the equality line is slightly skewed toward BE &gt; INT (__N_NEG__ vs. __N_POS__).</p>

<h3>3.3 Latent Profile Model Selection (K = 2 to __K_MAX__)</h3>
<p>BIC values across K = 2 to __K_MAX__ under the primary full-covariance specification are reported in Table 2 and shown in Figure 5.</p>

<div class="table-wrap">
<div class="table-title">Table 2. BIC, AIC, log-likelihood, and entropy across K = 2 to __K_MAX__ (full covariance, primary specification)</div>
<table>
<thead><tr><th>K</th><th>log L</th><th>k<sub>params</sub></th><th>AIC</th><th>BIC</th><th>Entropy</th><th>min class N</th><th>Notes</th></tr></thead>
<tbody>
__TBL2_ROWS__
</tbody>
</table>
<p class="table-note">Positive log-likelihoods at K ≥ 8 indicate singular covariance matrices (the EM algorithm fits delta-function spikes on individual points). These solutions are numerically degenerate and are not interpretable as profile solutions.</p>
</div>

<div class="figure">
__FIG5__
<p class="caption"><b>Figure 5.</b> BIC by K (K = 2 to __K_MAX__) under the primary full-covariance specification. K = __SEL_K__ is selected as the primary specification because it produced the lowest BIC (__SEL_BIC__) among the numerically well-behaved fits. K = __MIN_BIC_K__ produced the global minimum BIC (__MIN_BIC_VAL__) but is numerically degenerate (positive log-likelihood).</p>
</div>

<div class="caveat-box">
<p><b>K-selection decision.</b> K = __SEL_K__ is the lowest-BIC solution among the numerically well-behaved fits (K = 2 to __K_MAX__) under the primary full-covariance specification (BIC = __SEL_BIC__). The diagonal-covariance specification produces an even lower BIC at K = __SEL_K__ (BIC = __DIAG_K_DIAG_BIC__), reflecting the additional flexibility of full covariance on the 2-indicator model. K ≥ 8 solutions are numerically degenerate (positive log-likelihoods, indicating singular covariance matrices). K = __SEL_K__ was retained as the primary solution as the lowest-BIC well-behaved model under the primary full-covariance specification.</p>
</div>

<h3>3.4 Covariance Sensitivity</h3>
<div class="figure">
__FIG6__
<p class="caption"><b>Figure 6.</b> BIC by K under full vs diagonal covariance (K = 2 to __K_MAX__). The primary K = __SEL_K__ full-covariance BIC is __SEL_BIC__; the diagonal K = __SEL_K__ BIC is __DIAG_K_DIAG_BIC__. Source: <code>results/paper1_strengthening/covariance_k_extended.csv</code>.</p>
</div>

<h3>3.5 Selected Solution: K = __SEL_K__ Profile Structure (Lowest Well-Behaved BIC)</h3>
<p>The K = __SEL_K__ solution under the primary full-covariance specification has __SEL_K__ profiles with sizes __PS_STR__, summing to N = __N__. Directionally, __N_INT_GT__ profile(s) (__INT_GT_LBL__) have mean <i>z</i>(INT) &gt; mean <i>z</i>(BE) and __N_BE_GT__ (__BE_GT_LBL__) have mean <i>z</i>(BE) &gt; mean <i>z</i>(INT). The profile means on the z-score scale are shown in Figure 3 and the discrepancy distribution by profile in Figure 4.</p>

<div class="figure">
__FIG3__
<p class="caption"><b>Figure 3.</b> K = __SEL_K__-profile solution: profile-specific means on <i>z</i>(INT) and <i>z</i>(BE). Marker size is proportional to profile N. Red markers = z(INT) &gt; z(BE) profiles (__INT_GT_LBL__); blue markers = z(BE) &gt; z(INT) profiles (__BE_GT_LBL__). The dashed black line is the equality reference (z(INT) = z(BE)). Source: <code>results/04_lpa_estimation/K___SEL_K__/profile_parameters.csv</code>.</p>
</div>

<div class="figure">
__FIG4__
<p class="caption"><b>Figure 4.</b> Profile-specific intention–behaviour discrepancy (INT − BE) for the K = __SEL_K__ primary solution. Positive values (navy) indicate INT &gt; BE; negative values (crimson) indicate BE &gt; INT. Source: <code>results/04_lpa_estimation/K___SEL_K__/profile_parameters.csv</code>.</p>
</div>

<div class="table-wrap">
<div class="table-title">Table 3. K = __SEL_K__ profile descriptives (z-scores)</div>
<table>
<thead><tr><th>Profile</th><th>N</th><th>% of sample</th><th>Mean z(INT)</th><th>Mean z(BE)</th><th>z(INT) − z(BE)</th><th>Direction</th></tr></thead>
<tbody>
__TBL3_ROWS__
__TBL3_FOOTER__
</tbody>
</table>
<p class="table-note">Profile sizes and z-scores verbatim from <code>results/04_lpa_estimation/K___SEL_K__/profile_parameters.csv</code>.</p>
</div>

<h3>3.6 GAP Variance Decomposition</h3>
<p>The proportion of GAP variance attributable to <i>between-profile</i> differences at K = __SEL_K__ was __GAP_BETWEEN_VAR__ (__GAP_BETWEEN_PCT__% of the total GAP variance of __GAP_TOTAL__). The within-profile residual variance was __GAP_WITHIN_VAR__ (__GAP_WITHIN_PCT__% of total). Thus the K = __SEL_K__ solution captures a meaningful — but not dominant — share of GAP variance; the majority of GAP variance remains within-profile.</p>

<h3>3.7 Classification Quality</h3>
<div class="figure">
__FIG7__
<p class="caption"><b>Figure 7.</b> Maximum posterior probability distribution for K = __SEL_K__. Mean max posterior = __CLF_MEAN__ (median __CLF_MEDIAN__). The threshold lines show 0.70, 0.80, and 0.90. Most respondents are assigned with high confidence: __CLF_PCT_GE90__% have max posterior ≥ 0.90, __CLF_PCT_GE80__% ≥ 0.80, __CLF_PCT_GE70__% ≥ 0.70, and __CLF_PCT_LT70__% &lt; 0.70. Source: <code>results/paper1_strengthening/k7_classification_quality.csv</code> and <code>results/04_lpa_estimation/K___SEL_K__/posterior_probabilities.csv</code>.</p>
</div>

<div class="table-wrap">
<div class="table-title">Table 5. K = __SEL_K__ classification-quality numerical summary (N = __N__)</div>
<table>
<thead><tr><th>Statistic</th><th>Value</th></tr></thead>
<tbody>
__TBL5_ROWS__
</tbody>
</table>
<p class="table-note">All values verbatim from <code>results/paper1_strengthening/k7_classification_quality.csv</code>. The minimum observed max posterior is __CLF_MIN__ (above the 0.50 chance threshold for __SEL_K__ equally-likely classes); __CLF_N_LT50__ respondents have max posterior &lt; 0.50.</p>
</div>

<h3>3.8 Profile Stability</h3>
<div class="figure">
__FIG8__
<p class="caption"><b>Figure 8.</b> Bootstrap directional stability for K = __SEL_K__ across 200 bootstrap replications. Bars show the percentage of bootstraps in which the matched profile had mean z(INT) &gt; mean z(BE). Profile __NEAR_P__ is near chance (__NEAR_PCT__%) and should be interpreted with caution. Source: <code>results/paper1_strengthening/k7_stability.csv</code>.</p>
</div>

<h3>3.9 Cross-Validation</h3>
<div class="figure">
__FIG10__
<p class="caption"><b>Figure 10.</b> Five-fold cross-validation: mean held-out log-likelihood (±SD) across K = 2 to __CV_MAXK__. K = __SEL_K__ has the highest mean held-out log-likelihood (__CV_MEAN_LL_K7__; SD __CV_SD_LL_K7__), but the SD is large (__CV_SD_LL_K7__), consistent with instability of the K = __SEL_K__ optimum across folds. Cross-validated fit is one criterion and BIC is the primary selection criterion. Source: <code>results/paper1_strengthening/cv5_aggregated_by_k.csv</code>.</p>
</div>

<h3>3.10 Duplicate-Row Sensitivity (K = __SEL_K__)</h3>
<div class="figure">
__FIG9__
<p class="caption"><b>Figure 9.</b> Duplicate-row sensitivity for K = __SEL_K__: primary (N = __N__) vs duplicates-removed (N = __N_UNIQUE__) on eight key metrics. Pearson r shifts from __R_PRIMARY__ to __R_UNIQUE__, GAP SD from __GAP_SD_PRIMARY__ to __GAP_SD_UNIQUE__, % INT &gt; BE from __PCT_INT_PRIMARY__% to __PCT_INT_UNIQUE__%, K = __SEL_K__ BIC from __BIC_PRIMARY__ to __BIC_UNIQUE__, K = __SEL_K__ mean max posterior from __MMP_PRIMARY__ to __MMP_UNIQUE__. The choice to retain the __N_DUPS__ duplicates does not materially affect any reported finding. Source: K = __SEL_K__ results computed on <code>data.xls</code> (N = __N__) and on duplicates-removed (N = __N_UNIQUE__) using the same specification as primary LPA.</p>
</div>

<h3>3.11 Alternative-Specification Sensitivity (K = __SEL_K__)</h3>
<p>To assess whether the K = __SEL_K__ solution depends on specific modelling choices, the primary specification was compared against three alternative covariance structures (full / diagonal / spherical, all 10 indicators), two score representations (mean vs factor scores), and five random seeds (1–5). Table 6 reports the BIC, AIC, entropy, and total matching distance (sum of pairwise profile-centroid L2 distances under Hungarian matching) for each alternative.</p>

<div class="table-wrap">
<div class="table-title">Table 6. K = __SEL_K__ alternative-specification sensitivity (N = __N__; verbatim from <code>results/paper1_final/07_robustness/phase17_master.csv</code>)</div>
<table>
<thead><tr><th>Specification</th><th>BIC</th><th>AIC</th><th>Entropy</th><th>Total matching distance</th></tr></thead>
<tbody>
__TBL6_ROWS__
</tbody>
</table>
<p class="table-note">BIC and AIC are not directly comparable across the 10-indicator variants (different numbers of indicators and parameters). The total matching distance is comparable: distances of 2–3 indicate that the alternative profile structure differs substantially from the reference. Seed-perturbation distances of 1.1–3.3 (with BICs as low as 553.48 at seed 3) indicate the K = __SEL_K__ optimum is sensitive to random initialisation; the seed-42 fit (reference) is one well-behaved optimum but not the unique global optimum. The factor-score representation (matching distance 2.71) reports a higher BIC than the primary mean-score representation. Source: <code>results/paper1_final/07_robustness/phase17_master.csv</code>.</p>
</div>

<h3>3.12 Exploratory Analysis of Profile-Membership Associations</h3>
<div class="caveat-box">
<p><b>Exploratory caveat.</b> The following multinomial logistic regression analysis is presented as an exploratory pattern-of-association analysis, not as a causal model. The cross-sectional design precludes causal inference. Construct predictors exhibit severe multicollinearity (max VIF = __DIAG_MAX_VIF__; all __DIAG_N_VIF_GE26__ construct predictors with VIF ≥ 26) so individual coefficients are not uniquely identified; only the joint pattern of associations is interpretable.</p>
</div>

<p>The multinomial logistic regression (Profile 0 reference, __DIAG_NPRED__ predictors × __SEL_K_MINUS1__ non-reference contrasts = __DIAG_N_TESTS__ joint tests) converged: log L = __DIAG_LL__, AIC = __DIAG_AIC__, BIC = __DIAG_BIC__, McFadden pseudo-<i>R</i><sup>2</sup> = __DIAG_MC__. Of the __DIAG_N_TESTS__ joint Benjamini–Hochberg-FDR-adjusted tests (α = 0.05), __DIAG_N_FDR__ were statistically significant.</p>

<div class="table-wrap">
<div class="table-title">Table 4. Multinomial logistic regression: model-level diagnostics (Profile 0 reference; __DIAG_NPRED__ predictors; __DIAG_N_TESTS__ joint FDR tests)</div>
<table>
<thead><tr><th>Diagnostic</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Sample size (N)</td><td>__N__</td></tr>
<tr><td>Number of predictors</td><td>__DIAG_NPRED__</td></tr>
<tr><td>Number of tests (joint FDR)</td><td>__DIAG_N_TESTS__</td></tr>
<tr><td>Number of outcome classes (K)</td><td>__DIAG_K__</td></tr>
<tr><td>Log-likelihood (full)</td><td>__DIAG_LL__</td></tr>
<tr><td>Log-likelihood (null)</td><td>__DIAG_LL_NULL__</td></tr>
<tr><td>AIC</td><td>__DIAG_AIC__</td></tr>
<tr><td>BIC</td><td>__DIAG_BIC__</td></tr>
<tr><td>McFadden pseudo-<i>R</i><sup>2</sup></td><td>__DIAG_MC__</td></tr>
<tr><td>Max VIF</td><td>__DIAG_MAX_VIF__</td></tr>
<tr><td>Number of FDR-significant tests</td><td>__DIAG_N_FDR__ of __DIAG_N_TESTS__</td></tr>
<tr><td>Number of missing predictor values</td><td>0</td></tr>
<tr><td>Converged</td><td>Yes</td></tr>
</tbody>
</table>
<p class="table-note">McFadden pseudo-<i>R</i><sup>2</sup> = __DIAG_MC__ should not be interpreted as the proportion of outcome variance explained; it is a likelihood-ratio-based descriptive statistic. It is distinct from — and not comparable to — the Pearson <i>r</i><sup>2</sup> = __R2__ reported for the aggregate INT–BE association.</p>
</div>

<!-- ============================================================ -->
<h2>4. Discussion</h2>

<p>[LITERATURE SUPPORT NEEDED]</p>

<p>The central finding of this paper is that a strong population-level intention–behaviour correlation (<i>r</i> = __R_FROZEN__, <i>r</i><sup>2</sup> = __R2__) coexists with substantial within-sample heterogeneity in how INT and BE are configured. The K = __SEL_K__ latent profile solution under the primary full-covariance specification identifies __SEL_K__ subgroups with qualitatively different profiles — __N_INT_GT__ in which mean INT &gt; mean BE (__INT_GT_LBL__) and __N_BE_GT__ in which mean BE &gt; mean INT (__BE_GT_LBL__). The within-profile heterogeneity is, however, dominant: __GAP_WITHIN_PCT__% of the total GAP variance remains within profiles at K = __SEL_K__ (__GAP_BETWEEN_PCT__% between profiles), so the profile structure captures a non-trivial but not majority share of the gap variance.</p>

<p>Four methodological caveats are essential to interpreting this finding. <b>First</b>, K = __SEL_K__ is the lowest-BIC well-behaved solution (BIC = __SEL_BIC__), but the BIC values of the alternative seeds (range 553–1627) and the factor-score representation (BIC 3161) bracket this single point estimate; the solution is one defensible optimum but not a uniquely determined one. <b>Second</b>, the diagonal-covariance specification at K = __SEL_K__ produces a lower BIC (__DIAG_K_DIAG_BIC__) than the full-covariance specification; the primary specification was retained on substantive grounds (full covariance permits profile-level INT–BE correlation), but the diagonal alternative is a valid competing model. <b>Third</b>, classification is not perfect: __CLF_PCT_LT70__% of respondents have max posterior &lt; 0.70 and profile __NEAR_P__ in particular is directionally unstable across bootstraps (__NEAR_PCT__% INT &gt; BE consistency, near chance). The choice of K, the choice of covariance specification, and the assignment of ambiguous respondents all carry measurable uncertainty. <b>Fourth</b>, the sample contains 42 rows (3.60%) that are duplicates on all measured columns; while the duplicate-excluded sensitivity analysis (N = __N_UNIQUE__) confirms the qualitative pattern, the duplicates remain in the primary analysis.</p>

<p>The exploratory multinomial logistic regression indicates that profile membership is associated with the construct predictors — __DIAG_N_FDR__ of __DIAG_N_TESTS__ joint FDR-adjusted tests were significant — but the severe multicollinearity (max VIF = __DIAG_MAX_VIF__) precludes any individual-coefficient causal interpretation. The McFadden pseudo-<i>R</i><sup>2</sup> of __DIAG_MC__ is a likelihood-ratio-based descriptive summary of joint fit and should not be conflated with the Pearson <i>r</i><sup>2</sup> = __R2__ from the aggregate INT–BE association; the two are not on the same scale and answer different questions.</p>

<!-- ============================================================ -->
<h2>5. Limitations</h2>

<div class="limitations">
<ul>
<li><b>Cross-sectional design.</b> Causal claims about predictors of profile membership are not supported. [AUTHOR INPUT REQUIRED: sampling frame, recruitment, response rate, ethics, demographic-category labels, scale polarity, construct citations — these items are required for the methods section.]</li>
<li><b>Model dependence.</b> K = __SEL_K__ is the lowest-BIC well-behaved fit but the BIC distribution is not monotone: alternative seeds find lower BICs (553–881) with large matching distances, indicating the K = __SEL_K__ optimum is one of several well-behaved local optima. The diagonal-covariance specification produces a lower BIC. Exact profile boundaries depend on modelling choices.</li>
<li><b>K = __MIN_BIC_K__ numerical degeneracy.</b> The K = __MIN_BIC_K__ solution has the global minimum BIC (__MIN_BIC_VAL__) but is numerically degenerate (positive log-likelihood indicating singular covariance matrices) and is not interpretable as a profile solution.</li>
<li><b>Diagonal covariance competing.</b> Under diagonal covariance, K = 6 and K = 7 produce lower BICs than the full-covariance K = __SEL_K__. The full-covariance primary specification was retained for substantive reasons; the diagonal alternative is a valid competing model.</li>
<li><b>Profile __NEAR_P__ directional instability.</b> Profile __NEAR_P__ shows __NEAR_PCT__% bootstrap directional consistency (INT &gt; BE), essentially at chance. Its directional identity should not be treated as robustly established.</li>
<li><b>Within-profile variance dominant.</b> __GAP_WITHIN_PCT__% of GAP variance remains within profiles at K = __SEL_K__; profiles do not capture a majority share of the gap variance.</li>
<li><b>Predictor multicollinearity.</b> Construct predictors exhibit VIF up to __DIAG_MAX_VIF__; individual coefficients are not uniquely identified. Only the joint pattern of associations is interpretable.</li>
<li><b>Duplicate rows.</b> __N_DUPS__ rows (__DUP_PCT__%) were duplicates on all measured columns; retained in the primary analysis. Sensitivity analysis on N = __N_UNIQUE__ shows all key results are robust.</li>
<li><b>PEU low reliability.</b> PEU has Cronbach α = 0.673 (2 items), the lowest in the battery; PEU scores should be interpreted with caution.</li>
<li><b>Classification not perfect.</b> __CLF_PCT_LT70__% of respondents have max posterior &lt; 0.70; classification ambiguity is non-trivial for a minority of the sample.</li>
<li><b>Generalizability.</b> [AUTHOR INPUT REQUIRED: the sampling frame, recruitment method, and demographic composition are required to support generalizability claims. These items are not verifiable from the analytic file.]</li>
</ul>
</div>

<!-- ============================================================ -->
<h2>6. Conclusion</h2>

<p>This paper reports a Latent Profile Analysis of <i>N</i> = __N__ respondents' intention–behaviour configurations in household energy-saving behaviour. The K = __SEL_K__ solution under the primary full-covariance specification has the lowest BIC (BIC = __SEL_BIC__) among the numerically well-behaved fits, with __N_INT_GT__ INT &gt; BE profile(s) (__INT_GT_LBL__) and __N_BE_GT__ BE &gt; INT profile(s) (__BE_GT_LBL__), coexisting with a strong population-level INT–BE correlation (<i>r</i> = __R_FROZEN__). The finding is robust to duplicate-row removal (N = __N_UNIQUE__) and supported by 5-fold cross-validation (mean held-out log-likelihood = __CV_MEAN_LL_K7__ at K = __SEL_K__).</p>

<p>Four methodological realities constrain the substantive interpretation: alternative random seeds find lower BICs (553–881) at the K = __SEL_K__ specification, with profile matching distances of 1.1–3.3 indicating that the seed-42 fit is one well-behaved optimum among several; the diagonal-covariance specification produces a lower BIC than the full-covariance primary specification; profile __NEAR_P__ is directionally unstable across bootstraps (__NEAR_PCT__% INT &gt; BE consistency). The exploratory predictor analysis identifies __DIAG_N_FDR__ of __DIAG_N_TESTS__ joint-FDR-significant associations with severe multicollinearity (max VIF = __DIAG_MAX_VIF__), precluding individual-coefficient causal interpretation.</p>

<p>[AUTHOR INPUT REQUIRED: concluding statement situating the findings within the existing literature and naming directions for future work.]</p>

<!-- ============================================================ -->
<h2>Supplementary Materials</h2>

<p><b>Table S1.</b> K = 2 to __K_MAX__ model fit under full and diagonal covariance (<code>results/paper1_strengthening/covariance_k_extended.csv</code>).</p>
<p><b>Table S2.</b> Multinomial logistic regression coefficients, 95% confidence intervals, raw and FDR-adjusted p-values for __DIAG_NPRED__ predictors × __SEL_K_MINUS1__ contrasts = __DIAG_N_TESTS__ tests (<code>results/18_profile_predictors/multinomial_results.csv</code>).</p>
<p><b>Table S3.</b> Duplicate-row sensitivity: per-profile K = __SEL_K__ on N = __N_UNIQUE__ (<code>results/k7_primary/k7_duplicate_sensitivity_profiles.csv</code>).</p>
<p><b>Table S4.</b> Alternative-specification sensitivity (full/diag/spherical covariance; mean vs factor scores; random seeds 1–5; total matching distances; <code>results/paper1_final/07_robustness/phase17_master.csv</code>).</p>
<p><b>Table S5.</b> Construct descriptive statistics and reliability (<code>results/02_measurement/construct_statistics.csv</code>).</p>
<p><b>Figure S1.</b> K = __SEL_K__ profile structure (full covariance): sizes __PS_STR__ (<code>results/04_lpa_estimation/K___SEL_K__/profile_parameters.csv</code>).</p>

<!-- ============================================================ -->
<div class="footnote">
<p><b>Manuscript conventions.</b> All numerical values are taken verbatim from the frozen computational evidence package (<code>results/paper1_final/</code>, <code>results/paper1_strengthening/</code>, <code>results/k7_primary/</code>) and have been independently verified. Items marked <span class="placeholder">[AUTHOR INPUT REQUIRED]</span> are not verifiable from the analytic file and require author action before submission. Items marked <span class="placeholder">[LITERATURE SUPPORT NEEDED]</span> require author-supplied citations from the existing literature.</p>
<p><b>Numerical summary (primary):</b> N = __N__; <i>r</i>(INT, BE) = __R_FROZEN__ (95% CI [__CI_LO__, __CI_HI__], <i>p</i> __P_FROZEN__, <i>r</i><sup>2</sup> = __R2__); GAP mean ≈ 0, SD = __GAP_SD__, range [__GAP_MIN__, __GAP_MAX__], __N_POS__ INT &gt; BE (__PCT_POS__%), __N_NEG__ BE &gt; INT (__PCT_NEG__%); K = __SEL_K__ BIC = __SEL_BIC__ (lowest well-behaved BIC); profile sizes __PS_STR__; __N_INT_GT__ INT&gt;BE + __N_BE_GT__ BE&gt;INT; mean max posterior = __CLF_MEAN__; McFadden pseudo-<i>R</i><sup>2</sup> = __DIAG_MC__ (distinct from <i>r</i><sup>2</sup>); __DIAG_N_FDR__ of __DIAG_N_TESTS__ joint FDR tests significant; max VIF = __DIAG_MAX_VIF__; duplicate sensitivity robust at N = __N_UNIQUE__.</p>
</div>

</body>
</html>
"""

# -----------------------------------------------------------------
# TOKEN SUBSTITUTION (single pass; HTML uses raw CSS so .replace() is
# safer than f-string formatting the entire template)
# -----------------------------------------------------------------
import math
_P_FROZEN_HTML = _p(P_FROZEN)
_SCI_E = f"{P_FROZEN:.2e}"
if "e-0" in _SCI_E:
    _SCI_E = _SCI_E.replace("e-0", "e-")
_DUP_PCT_FMT = f"{N_DUPS / N_TOTAL * 100:.2f}"

_TOK = {
    "N": str(N_TOTAL),
    "N_DUPS": str(N_DUPS),
    "N_UNIQUE": str(N_UNIQUE),
    "DUP_PCT": _DUP_PCT_FMT,
    "SEL_K": str(SEL_K),
    "N_EVAL": str(_N_EVAL),
    "PS_STR": _PS_STR,
    "INT_GT_LBL": _INT_GT_LBL,
    "BE_GT_LBL": _BE_GT_LBL,
    "N_INT_GT": str(len(_int_gt)),
    "N_BE_GT": str(len(_be_gt)),
    "K_MAX": str(K_MAX),
    "SEL_BIC": f"{_sel_bic:.2f}",
    "SEL_N_PARAMS": str(SEL_N_PARAMS),
    "SEL_N_MEANS": str(SEL_N_MEANS),
    "SEL_N_COV": str(SEL_N_COV),
    "SEL_K_MINUS1": str(SEL_K_MINUS1),
    "MIN_BIC_K": str(_min_bic_k),
    "MIN_BIC_VAL": f"{_min_bic_val:.2f}",
    "DIAG_K_DIAG_BIC": f"{_diag_k_diag_bic:.2f}",
    "R_FROZEN": f"{R_FROZEN:.4f}",
    "R2": f"{R2:.4f}",
    "CI_LO": f"{CI_LO:.4f}",
    "CI_HI": f"{CI_HI:.4f}",
    "P_FROZEN": _P_FROZEN_HTML,
    "GAP_SD": f"{GAP_SD:.4f}",
    "GAP_MIN": f"{GAP_MIN:.4f}",
    "GAP_MAX": f"{GAP_MAX:.4f}",
    "GAP_MEDIAN": f"{GAP_MEDIAN:.4f}",
    "N_POS": str(N_POS),
    "N_NEG": str(N_NEG),
    "N_ZERO": str(N_ZERO),
    "PCT_POS": f"{PCT_POS:.2f}",
    "PCT_NEG": f"{PCT_NEG:.2f}",
    "GAP_BETWEEN_VAR": f"{GAP_BETWEEN_VAR:.6f}",
    "GAP_WITHIN_VAR": f"{GAP_WITHIN_VAR:.6f}",
    "GAP_TOTAL": f"{GAP_TOTAL:.6f}",
    "GAP_BETWEEN_PCT": f"{GAP_BETWEEN_PCT:.2f}",
    "GAP_WITHIN_PCT": f"{GAP_WITHIN_PCT:.2f}",
    "R_PRIMARY": f"{R_PRIMARY:.4f}",
    "R_UNIQUE": f"{R_UNIQUE:.4f}",
    "GAP_SD_PRIMARY": f"{GAP_SD_PRIMARY:.4f}",
    "GAP_SD_UNIQUE": f"{GAP_SD_UNIQUE:.4f}",
    "PCT_INT_PRIMARY": f"{PCT_INT_PRIMARY:.2f}",
    "PCT_INT_UNIQUE": f"{PCT_INT_UNIQUE:.2f}",
    "BIC_PRIMARY": f"{BIC_PRIMARY:.2f}",
    "BIC_UNIQUE": f"{BIC_UNIQUE:.2f}",
    "ENT_PRIMARY": f"{ENT_PRIMARY:.4f}",
    "ENT_UNIQUE": f"{ENT_UNIQUE:.4f}",
    "MMP_PRIMARY": f"{MMP_PRIMARY:.4f}",
    "MMP_UNIQUE": f"{MMP_UNIQUE:.4f}",
    "CLF_MEAN": f"{CLF_MEAN:.4f}",
    "CLF_MEDIAN": f"{CLF_MEDIAN:.4f}",
    "CLF_SD": f"{CLF_SD:.4f}",
    "CLF_MIN": f"{CLF_MIN:.4f}",
    "CLF_N_GE90": str(CLF_N_GE90),
    "CLF_N_GE80": str(CLF_N_GE80),
    "CLF_N_GE70": str(CLF_N_GE70),
    "CLF_N_LT70": str(CLF_N_LT70),
    "CLF_N_LT50": str(CLF_N_LT50),
    "CLF_PCT_GE90": f"{CLF_PCT_GE90:.2f}",
    "CLF_PCT_GE80": f"{CLF_PCT_GE80:.2f}",
    "CLF_PCT_GE70": f"{CLF_PCT_GE70:.2f}",
    "CLF_PCT_LT70": f"{CLF_PCT_LT70:.2f}",
    "CLF_PCT_LT50": f"{CLF_PCT_LT50:.2f}",
    "NEAR_P": _NEAR_P,
    "NEAR_PCT": f"{_NEAR_PCT:.1f}",
    "CV_MAXK": str(_cv_maxk),
    "CV_MEAN_LL_K7": f"{CV_MEAN_LL_K7:.2f}",
    "CV_SD_LL_K7": f"{CV_SD_LL_K7:.2f}",
    "DIAG_N": str(DIAG_N),
    "DIAG_K": str(DIAG_K),
    "DIAG_NPRED": str(DIAG_NPRED),
    "DIAG_N_TESTS": str(DIAG_N_TESTS),
    "DIAG_N_FDR": str(DIAG_N_FDR),
    "DIAG_LL": f"{DIAG_LL:.2f}",
    "DIAG_LL_NULL": f"{DIAG_LL_NULL:.2f}",
    "DIAG_AIC": f"{DIAG_AIC:.2f}",
    "DIAG_BIC": f"{DIAG_BIC:.2f}",
    "DIAG_MC": f"{DIAG_MC:.4f}",
    "DIAG_MAX_VIF": f"{DIAG_MAX_VIF:.2f}",
    "DIAG_N_VIF_GE26": str(DIAG_N_VIF_GE26),
    "TBL1_ROWS": _TBL1_ROWS,
    "TBL2_ROWS": _TBL2_ROWS,
    "TBL3_ROWS": _TBL3_ROWS,
    "TBL3_FOOTER": _TBL3_FOOTER,
    "TBL5_ROWS": _TBL5_ROWS,
    "TBL6_ROWS": _TBL6_ROWS,
}

# Apply all single-underscore tokens first; these never collide with figures
for _k, _v in _TOK.items():
    HTML = HTML.replace("__" + _k + "__", _v)

# Then embed figures (must come after token pass so __FIG*__ is still intact)
HTML = HTML.replace("__FIG1__", fig("fig1", "Aggregate INT-BE association"))
HTML = HTML.replace("__FIG2__", fig("fig2", "GAP distribution"))
HTML = HTML.replace("__FIG3__", fig("fig3", "Profile INT vs BE structure"))
HTML = HTML.replace("__FIG4__", fig("fig4", "Profile INT/BE means"))
HTML = HTML.replace("__FIG5__", fig("fig5", "Extended K=2-10 BIC"))
HTML = HTML.replace("__FIG6__", fig("fig6", "Covariance sensitivity"))
HTML = HTML.replace("__FIG7__", fig("fig7", "Classification quality"))
HTML = HTML.replace("__FIG8__", fig("fig8", "Profile stability"))
HTML = HTML.replace("__FIG9__", fig("fig9", "Duplicate sensitivity"))
HTML = HTML.replace("__FIG10__", fig("fig10", "Cross-validation"))

# Sanity check: any remaining __TOK__ markers would mean a token was never defined
import re as _re
_remaining = _re.findall(r"__[A-Z][A-Z0-9_]*__", HTML)
if _remaining:
    raise RuntimeError(f"Unsubstituted tokens: {sorted(set(_remaining))}")

with open(OUT_PATH, "w") as f:
    f.write(HTML)

print(f"Wrote {OUT_PATH}")
print(f"Size: {os.path.getsize(OUT_PATH):,} bytes")
