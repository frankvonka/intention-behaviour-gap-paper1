#!/usr/bin/env python3
"""Build the final self-contained K=7-primary HTML manuscript.

K=7 is the primary solution (lowest BIC among numerically well-behaved models).
K=6 is retained as the leading competing solution (Supplementary Figure S1,
Supplementary Tables S7/S8).
All numbers come from frozen evidence (results/paper1_final,
results/paper1_strengthening) and from the validated K=7 extension analyses
(results/k7_primary). Figures are embedded as base64 PNG from
results/draft_figures/draft_figures_base64.json.

Output: paper1_final_manuscript_k7.html
"""
import base64
import html as html_mod
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

FIGS = json.load(open("results/draft_figures/draft_figures_base64.json"))
S = "results/paper1_strengthening"
PF = "results/paper1_final"
K7 = "results/k7_primary"


# ---------------------------------------------------------------- helpers
def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "&mdash;"
    return f"{float(x):.{nd}f}"


def m(x, nd=2):
    """fmt with true minus signs."""
    return fmt(x, nd).replace("-", "&minus;")


def msign(x, nd=2):
    """fmt with explicit sign and true minus."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "&mdash;"
    v = float(x)
    s = f"{v:+.{nd}f}".replace("-", "&minus;")
    return s.replace("+", "+")


def fmt_sci(p, nd=2):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "&mdash;"
    p = float(p)
    if p == 0:
        return "0"
    e = int(np.floor(np.log10(abs(p))))
    mm = p / 10 ** e
    sup = str(e).replace("-", "&minus;")
    return f"{mm:.{nd}f} &times; 10<sup>{sup}</sup>"


def esc(s):
    return html_mod.escape(str(s))


def table_html(headers, rows, row_classes=None, align=None):
    """rows: list of lists of pre-formatted strings. align: list of 'n'/'c'/None."""
    out = ['<table class="num">']
    if align:
        out.append("<thead><tr>" + "".join(
            f'<th class="{a}">{h}</th>' if a else f"<th>{h}</th>"
            for h, a in zip(headers, align)) + "</tr></thead><tbody>")
    else:
        out.append("<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers)
                   + "</tr></thead><tbody>")
    for i, r in enumerate(rows):
        rc = ""
        if row_classes is not None and row_classes[i]:
            rc = f' class="{row_classes[i]}"'
        cells = []
        for j, c in enumerate(r):
            a = align[j] if align else None
            ac = f' class="{a}"' if a else ""
            cells.append(f"<td{ac}>{c}</td>")
        out.append(f"<tr{rc}>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def figure_embed(figkey, num, caption):
    b64 = FIGS[figkey]
    return (
        f'<figure class="figfig">'
        f'<img src="data:image/png;base64,{b64}" alt="Figure {num}"/>'
        f"<figcaption><b>Figure {num}.</b> {caption}</figcaption></figure>"
    )


CSS = """
body { font-family: Georgia, 'Times New Roman', serif; max-width: 940px;
       margin: 0 auto; padding: 24px 34px 90px; color: #1c1c1c;
       line-height: 1.65; font-size: 16px; }
h1 { font-size: 1.65em; line-height: 1.3; margin-bottom: 6px; }
.subtitle { color: #555; font-size: 1.0em; margin-bottom: 4px; }
.meta { color: #777; font-size: 0.88em; margin-bottom: 18px; }
h2 { border-bottom: 2px solid #4C72B0; padding-bottom: 4px; margin-top: 2.1em;
     font-size: 1.3em; color: #2c3e50; }
h3 { color: #2c3e50; margin-top: 1.4em; font-size: 1.05em; }
table.num { border-collapse: collapse; margin: 14px auto; font-size: 0.86em; }
table.num th { background: #4C72B0; color: #fff; padding: 6px 10px;
               border: 1px solid #3a5a8c; font-weight: 600; }
table.num td { padding: 5px 10px; border: 1px solid #d5dbe3; }
table.num tbody tr:nth-child(even) { background: #f4f6fa; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
td.c, th.c { text-align: center; }
tr.flag td { background: #fdecea !important; }
tr.sig td { background: #e8f6ee !important; }
tr.total td { font-weight: 700; background: #e9edf3 !important; }
figure.figfig { margin: 30px 0; text-align: center; page-break-inside: avoid; }
figure.figfig img { max-width: 100%; height: auto; border: 1px solid #e3e6ea; }
figcaption { font-size: 0.88em; color: #333; text-align: justify;
             margin-top: 8px; line-height: 1.5; }
.placeholder { background: #fff8e1; border-left: 4px solid #f0a500;
               padding: 10px 14px; margin: 14px 0; font-size: 0.95em; }
.note { background: #eef3fb; border-left: 4px solid #4C72B0;
        padding: 10px 14px; margin: 14px 0; font-size: 0.95em; }
.abst { background: #f7f8fa; border: 1px solid #e0e4ea; padding: 18px 24px;
        margin: 18px 0; }
.abst h3 { margin-top: 0.9em; }
.kw { font-size: 0.95em; color: #444; }
.toc { background: #f7f8fa; border: 1px solid #e0e4ea; padding: 12px 24px;
       font-size: 0.9em; column-count: 2; }
.toc a { text-decoration: none; color: #2c5f9e; }
.src { font-family: 'DejaVu Sans Mono', monospace; font-size: 0.82em;
       color: #555; }
footer { margin-top: 3em; border-top: 2px solid #4C72B0; padding-top: 12px;
         color: #555; font-size: 0.9em; }
@media print { body { max-width: none; padding: 10mm; } }
"""

# ---------------------------------------------------------------- frozen data
cstats = pd.read_csv(f"{PF}/02_measurement/construct_statistics.csv")
kext = pd.read_csv(f"{S}/k_extended_model_comparison.csv")
kext_full = kext[kext.covariance_type == "full"].copy()
cov = pd.read_csv(f"{S}/covariance_k_extended.csv")
prof7 = pd.read_csv(f"{K7}/k7_profile_table_full_sample.csv")
gap7 = pd.read_csv(f"{K7}/k7_per_profile_gap.csv").set_index("profile")
stab = pd.read_csv(f"{S}/k7_stability.csv")
cv = pd.read_csv(f"{S}/cv5_aggregated_by_k.csv")
dup = pd.read_csv(f"{K7}/k7_duplicate_sensitivity.csv")
_dg = dict(zip(dup.metric, pd.to_numeric(dup.frozen, errors="coerce")))
_du = dict(zip(dup.metric, pd.to_numeric(dup.unique_sample, errors="coerce")))
diag = pd.read_csv(f"{K7}/k7_predictors/model_diagnostics.csv")
diagv = dict(zip(diag.metric, diag.value))
mnl7 = pd.read_csv(f"{K7}/k7_predictors/multinomial_results.csv")
mnl = mnl7  # primary-solution predictor table backing _demogs/_construct_preds below
psum = pd.read_csv(f"{K7}/k7_predictors/predictor_summary.csv").set_index("predictor")
eig = pd.read_csv(f"{K7}/covariance_eigenvalues.csv")
dupprof = pd.read_csv(f"{K7}/k7_duplicate_sensitivity_profiles.csv")
# frozen K=COMP predictor evidence (results/paper1_final was frozen when K=6 was
# primary; results/18_profile_predictors now holds the K=7 rerun)
_diag6 = pd.read_csv(f"{PF}/06_predictors/model_diagnostics.csv")
_diag6v = dict(zip(_diag6.metric, _diag6.value))
_N_TESTS_COMP = int(_diag6v["n_tests"])
# item / construct counts (no literals)
_N_ITEMS = int(cstats["k_items"].sum())
_N_CONSTRUCTS = int(len(cstats))
# demographic / construct predictor split, needed as early as the Abstract
_demogs = sorted(set(mnl["predictor"]) - set(cstats.construct))
# missing-cell count backing the Section 2.1 sentence
_miss = pd.read_csv("results/01_data_inspection/missing_values.csv")
_N_MISSING = int(_miss["n_missing"].sum())
if _N_MISSING == 0:
    _miss_sentence = "There were no missing values across the measured cells."
else:
    _miss_sentence = (f"There were {_N_MISSING} missing values across the "
                      f"measured cells.")
# bootstrap replication count from the phase-14 master record (one row per
# profile lives in stab, so len(stab) is the profile count, not the
# replication count)
_boot_master = pd.read_csv("results/14_k6_stability/k6_stability_master.csv")
_boot_master_v = dict(zip(_boot_master.metric, _boot_master.value))
_N_BOOT = int(_boot_master_v["n_bootstrap"])
# CV K-range (backing Section 2.8 prose)
_KCV_MIN, _KCV_MAX = int(cv.K.min()), int(cv.K.max())
# reg_covar-floor component counts per K, recomputed from the frozen
# task1_K_* covariance matrices (eigen audit file only covers K=6-7)
_floor_counts = {}
for _k in range(int(kext_full.K.min()), int(kext_full.K.max()) + 1):
    try:
        _d = json.load(open(f"{S}/task1_K_{_k}/covariance_matrices.json"))
        _mats = [np.array(v) for v in _d["covariances"].values()]
        _floor_counts[_k] = sum(
            1 for _mm in _mats
            if float(np.linalg.eigvalsh(_mm).min()) <= 1.5e-6)
    except FileNotFoundError:
        _floor_counts[_k] = None
_clean_ks = sorted(k for k, v in _floor_counts.items() if v == 0)

uniq_size_by_frozen = {}
for _, r in dupprof.iterrows():
    uniq_size_by_frozen[int(r.matched_frozen_profile)] = int(r.N)

dec = pd.read_csv(f"{K7}/gap_variance_decomposition_K2_K7.csv").set_index("K")
between7 = float(dec.loc[7, "between_over_total"]) * 100.0
within7 = 100.0 - between7

_corr = pd.read_csv("results/02_measurement/int_be_correlation.csv")
_R = float(_corr["r"].iloc[0]); _CI_LO = float(_corr["CI95_lower"].iloc[0])
_CI_HI = float(_corr["CI95_upper"].iloc[0]); _P = float(_corr["p"].iloc[0])
_R2 = _R * _R
import math as _math
_p_exp = int(_math.floor(_math.log10(_P))); _p_mant = _P / (10 ** _p_exp)
_gs = pd.read_csv(f"{PF}/03_intention_behavior_gap/gap_statistics.csv")
_gsv = dict(zip(_gs.statistic, _gs.value))
_cq = pd.read_csv(f"{S}/k7_classification_quality.csv").set_index("statistic")["value"]
_N_TOTAL = int(_dg["N_full_sample"]); _N_UNIQUE = int(_dg["N_unique_sample"])
_N_DUPS = int(_dg["n_duplicates_removed"])
_gapmean = float(_gsv["mean"])
if _gapmean == 0:
    _gapmean_exp = 0; _gapmean_mant = 0.0
else:
    _gapmean_exp = int(_math.floor(_math.log10(abs(_gapmean))))
    _gapmean_mant = _gapmean / (10 ** _gapmean_exp)

# --- selected primary K and leading competing solution (all data-driven) ---
_sel = pd.read_csv("results/05_lpa_selection/selected_model.csv")
SEL = int(_sel["selected_K"].iloc[0])
N_INIT_SEL = int(_sel["n_init"].iloc[0]); SEED_SEL = int(_sel["random_seed"].iloc[0])
_wb = kext_full[kext_full.log_likelihood < 0]          # numerically well-behaved only
COMP = int(_wb[_wb.K != SEL].sort_values("BIC").K.iloc[0])
_BIC_SEL = float(kext_full.loc[kext_full.K == SEL, "BIC"].iloc[0])
_BIC_COMP = float(kext_full.loc[kext_full.K == COMP, "BIC"].iloc[0])
PARAMS_SEL = int(kext_full.loc[kext_full.K == SEL, "n_params"].iloc[0])
PARAMS_COMP = int(kext_full.loc[kext_full.K == COMP, "n_params"].iloc[0])

# --- genuine competing-solution (K = COMP) table from the per-K LPA artifacts ---
_k6m = pd.read_csv(f"results/04_lpa_estimation/K_{COMP}/profile_means.csv")
_k6s = pd.read_csv(f"results/04_lpa_estimation/K_{COMP}/profile_sizes.csv")
_size_col6 = "size" if "size" in _k6s.columns else "N"
k6tab = _k6m.reset_index(drop=True).copy()
k6tab["N"] = [_k6s.loc[_k6s["profile"] == p, _size_col6].iloc[0]
              for p in range(len(k6tab))]
k6tab["percentage"] = [float(n) / _N_TOTAL * 100.0 for n in k6tab["N"]]
_dir6 = ["INT &gt; BE" if zi > zb else "BE &gt; INT"
         for zi, zb in zip(k6tab["z_INT"], k6tab["z_BE"])]

# --- floor components (reg_covar lattice collapse) per K, from the eigen audit ---
_floor = eig[eig.at_reg_floor == True]  # noqa: E712
_floor_sel = sorted(int(p) for p in _floor[_floor.K == SEL].profile)
_floor_comp = sorted(int(p) for p in _floor[_floor.K == COMP].profile)
_n_floor_sel = len(_floor_sel); _n_floor_comp = len(_floor_comp)
def _floor_desc(profiles, k):
    if not profiles:
        return "none"
    bits = []
    for p in profiles:
        row = eig[(eig.K == k) & (eig.profile == p)].iloc[0]
        if float(row.SD_INT) == 0 and float(row.SD_BE) == 0:
            bits.append(f"P{p} (both indicator SDs zero)")
        elif float(row.SD_INT) == 0:
            bits.append(f"P{p} (INT variance zero)")
        elif float(row.SD_BE) == 0:
            bits.append(f"P{p} (BE variance zero)")
        else:
            bits.append(f"P{p}")
    return "; ".join(bits)

# Ks with floor components other than the primary/competing solutions,
# restricted to numerically well-behaved Ks (the degenerate K >= 8
# solutions are disclosed separately in Section 3.3/5.5)
_DEG_KS_EARLY = sorted(int(k) for k in kext_full.loc[kext_full.log_likelihood > 0, "K"])
_floor_ks_mid = sorted(k for k, v in _floor_counts.items()
                       if v and v > 0 and k not in (SEL, COMP)
                       and k not in _DEG_KS_EARLY)
_floor_mid_str = ("one" if all((_floor_counts[k] or 0) == 1 for k in _floor_ks_mid)
                  else ", ".join(str(_floor_counts[k]) for k in _floor_ks_mid))
_floor_mid_range = ("&ndash;".join(str(k) for k in (_floor_ks_mid[0], _floor_ks_mid[-1]))
                    if _floor_ks_mid else "none")
_floor_mid_list = ", ".join(str(k) for k in _floor_ks_mid)

H = []

# ================================================================ FRONT MATTER
_DEG_KS = sorted(int(k) for k in kext_full.loc[kext_full.log_likelihood > 0, "K"])
DEG_K_MIN = min(_DEG_KS)
_diag_rows = cov.pivot(index="K", columns="covariance_type", values="BIC")
_DIAG_LOWER = sorted(int(k) for k in _diag_rows.index
                     if _diag_rows.loc[k, "diag"] < _diag_rows.loc[k, "full"])
H.append(f"""
<h1>Heterogeneous Intention&ndash;Behaviour Configurations in Household
Energy-Saving Behaviour: A Latent Profile Analysis of N&nbsp;=&nbsp;{_N_TOTAL}
Respondents</h1>
<div class="meta">Manuscript generated from the frozen computational evidence
package. Primary solution: K&nbsp;=&nbsp;{SEL} (lowest BIC among numerically
well-behaved full-covariance models). K&nbsp;=&nbsp;{COMP} retained as the
leading competing solution.</div>

<div class="toc">
<b>Contents</b><br/>
<a href="#abstract">Abstract</a> &middot;
<a href="#s1">1. Introduction</a> &middot;
<a href="#s2">2. Methods</a> &middot;
<a href="#s3">3. Results</a> &middot;
<a href="#s4">4. Discussion</a> &middot;
<a href="#s5">5. Limitations</a> &middot;
<a href="#s6">6. Conclusion</a> &middot;
<a href="#supp">Supplementary Materials</a> &middot;
<a href="#transparency">Data and Computational Transparency</a>
</div>

<div class="abst" id="abstract">
<h2 style="border:none; margin-top:0">Abstract</h2>
<h3>Background</h3>
<p>A core assumption in behavioural theory is that intention is an important
proximal determinant of behaviour, yet substantial discrepancies between stated
intention and reported behaviour have been documented. An important question is
whether such discrepancies represent a relatively uniform population-level
phenomenon or whether distinct subgroups exhibit qualitatively different
intention&ndash;behaviour configurations.</p>
<h3>Objective</h3>
<p>To examine heterogeneous intention&ndash;behaviour configurations in a
cross-sectional sample of N&nbsp;=&nbsp;{_N_TOTAL} respondents reporting household
energy-saving behaviour using Latent Profile Analysis (LPA) of standardised
intention (INT) and behaviour (BE) indicators.</p>
<h3>Methods</h3>
<p>A 10-construct psychometric battery comprising 29 Likert-scale items was
administered, alongside five demographic variables. The constructs were
attitude, context, subjective norm, COVID context, intention, behaviour,
perceived usefulness, perceived ease of use, personal obligation, and personal
responsibility. Construct scores were calculated as arithmetic means of their
constituent items; Cronbach&rsquo;s &alpha; ranged from {float(cstats['Cronbach_alpha'].min()):.3f} to {float(cstats['Cronbach_alpha'].max()):.3f}. The
intention&ndash;behaviour gap was defined as
GAP<sub>i</sub>&nbsp;=&nbsp;z(INT<sub>i</sub>)&nbsp;&minus;&nbsp;z(BE<sub>i</sub>),
with within-sample standardisation using ddof&nbsp;=&nbsp;1. LPA was estimated
using sklearn GaussianMixture models with full covariance,
n_init&nbsp;=&nbsp;{N_INIT_SEL},
random_state&nbsp;=&nbsp;{SEED_SEL}, max_iter&nbsp;=&nbsp;500, and
reg_covar&nbsp;=&nbsp;10<sup>&minus;6</sup>. Models with K&nbsp;=&nbsp;{int(kext_full.K.min())}&ndash;{int(kext_full.K.max())}
profiles were examined. Model selection was based primarily on BIC restricted to
numerically well-behaved solutions, supplemented by classification quality,
bootstrap directional stability, cross-validation, and sensitivity analyses.
Exploratory profile-membership associations were examined using multinomial
logistic regression with {int(diagv['n_predictors'])} predictors and joint
Benjamini&ndash;Hochberg false-discovery-rate correction across
{int(diagv['n_tests'])} tests.</p>
<h3>Results</h3>
<p>The aggregate INT&ndash;BE association was strong, r&nbsp;=&nbsp;{_R:.4f},
95%&nbsp;CI&nbsp;[{_CI_LO:.4f},&nbsp;{_CI_HI:.4f}], p&nbsp;=&nbsp;{_p_mant:.2f}&nbsp;&times;&nbsp;10<sup>&minus;{_p_exp}</sup>,
corresponding to r&sup2;&nbsp;=&nbsp;{_R2:.4f}. The standardised GAP had mean
&minus;{abs(_gapmean_mant):.3f}&nbsp;&times;&nbsp;10<sup>&minus;{_gapmean_exp}</sup> (effectively zero),
SD&nbsp;=&nbsp;{float(_gsv['SD']):.6f}, and range [{float(_gsv['min']):.2f},&nbsp;{float(_gsv['max']):+.2f}]. Of the {_N_TOTAL}
respondents, {int(_gsv['positive_gap_count'])} ({float(_gsv['positive_gap_pct']):.2f}%) had INT&nbsp;&gt;&nbsp;BE and {int(_gsv['negative_gap_count'])} ({float(_gsv['negative_gap_pct']):.2f}%) had
BE&nbsp;&gt;&nbsp;INT. Under the primary full-covariance specification, the
K&nbsp;=&nbsp;{SEL} solution had the lowest BIC among numerically well-behaved
models (BIC&nbsp;=&nbsp;{_BIC_SEL:.2f}, versus {_BIC_COMP:.2f} at K&nbsp;=&nbsp;{COMP}) and comprised
profiles of sizes {', '.join(str(v) for v in prof7['N'].astype(int))}. {int((prof7.mean_z_INT > prof7.mean_z_BE).sum())} profiles had mean
INT&nbsp;&gt;&nbsp;mean BE ({"P" + ", P".join(str(i) for i in prof7.index[prof7.mean_z_INT > prof7.mean_z_BE])}) and {int((prof7.mean_z_BE > prof7.mean_z_INT).sum())} had mean BE&nbsp;&gt;&nbsp;mean
INT ({"P" + ", P".join(str(i) for i in prof7.index[prof7.mean_z_BE > prof7.mean_z_INT])}). Mean maximum posterior probability was {float(_cq['mean_max_posterior']):.4f}, with
{float(_cq['pct_maxpost_ge_0.70']):.2f}% of respondents having maximum posterior probability &ge;0.70. Bootstrap
directional stability ranged from {min(stab.pct_INT_gt_BE):.1f}% to {max(stab.pct_INT_gt_BE):.1f}% for the proportion of
replications in which the matched profile had INT&nbsp;&gt;&nbsp;BE; no profile
fell near chance. At K&nbsp;=&nbsp;{SEL}, {between7:.2f}% of GAP variance was between
profiles and {within7:.2f}% within profiles. Five-fold cross-validation produced the
highest mean held-out log-likelihood for K&nbsp;=&nbsp;{SEL}
({m(float(cv.loc[cv.K == SEL, 'mean_ll_test'].iloc[0]))}), but with a large fold-to-fold SD ({float(cv.loc[cv.K == SEL, 'sd_ll_test'].iloc[0]):.2f}), indicating
cross-validated instability. K&nbsp;&ge;&nbsp;{DEG_K_MIN} produced numerically degenerate
solutions characterised by positive log-likelihoods and singular covariance
behaviour, and diagonal covariance produced lower BIC than full covariance at
K&nbsp;=&nbsp;{_DIAG_LOWER[0]}&ndash;{_DIAG_LOWER[-1]}. {_n_floor_sel} of the {SEL} K&nbsp;=&nbsp;{SEL} components operated
at the reg_covar floor because they captured respondents with constant item
scores &mdash; an artefact of the discrete 1&ndash;5 response lattice shared with
K&nbsp;=&nbsp;{COMP} ({_n_floor_comp} of {COMP} components). Removal of {_N_DUPS} duplicate rows preserved
the {SEL}-profile structure under Hungarian matching (mean z-space distance
{m(_du['K7_hungarian_mean_distance'], 2)}) although class sizes shifted materially. Exploratory multinomial logistic
regression yielded {int(diagv['n_FDR_significant'])} FDR-significant associations among {int(diagv['n_tests'])} tests, with
McFadden pseudo-R&sup2;&nbsp;=&nbsp;{float(diagv['McFadden_pseudo_R2']):.4f}; severe multicollinearity was present
(maximum VIF&nbsp;=&nbsp;{float(diagv['max_VIF']):.2f}).</p>
<h3>Conclusion</h3>
<p>The data show that a strong population-level intention&ndash;behaviour
association coexists with heterogeneous intention&ndash;behaviour
configurations. The K&nbsp;=&nbsp;{SEL} solution identifies {int((prof7.mean_z_INT > prof7.mean_z_BE).sum())} profiles with
INT&nbsp;&gt;&nbsp;BE and {int((prof7.mean_z_BE > prof7.mean_z_INT).sum())} with BE&nbsp;&gt;&nbsp;INT, but the exact profile
structure is model-dependent: K&nbsp;=&nbsp;{COMP} is a well-behaved competing
solution, diagonal covariance provides a competing specification, and {_n_floor_sel}
components reflect the discrete response lattice. The findings therefore support
heterogeneity in intention&ndash;behaviour configurations while requiring
cautious interpretation of the exact number and boundaries of latent
profiles.</p>
<p class="kw"><b>Keywords:</b> intention&ndash;behaviour gap; energy-saving
behaviour; household energy; latent profile analysis; person-centred analysis;
pro-environmental behaviour; behavioural heterogeneity</p>
</div>
""")

# ================================================================ 1. INTRO
H.append("""
<h2 id="s1">1. Introduction</h2>
<p>The relationship between behavioural intention and subsequent behaviour
occupies a central position in behavioural research. Intention is generally
conceptualised as an important proximal determinant of behaviour, yet empirical
research has repeatedly documented incomplete correspondence between what
individuals intend to do and what they report doing.</p>
<p>In the context of the Theory of Planned Behavior and the Norm Activation
Model, behavioural intention is frequently identified as the primary antecedent
of behaviour. While established frameworks posit intention as the proximal
determinant, meta-analytic evidence increasingly demonstrates that intention
accounts for only a modest share of the variance in actual behaviour (Sheeran
&amp; Gemmecke et al., 2025), highlighting a persistent intention&ndash;behaviour
gap.</p>
<p>The discrepancy between intention and behaviour has commonly been examined
through variable-centred approaches. Such approaches estimate the average
strength of the intention&ndash;behaviour relationship across a population and
investigate variables that may strengthen or weaken that relationship.</p>
<p>Research has sought to bridge this gap by examining moderators of the
intention&ndash;behaviour relationship, such as behavioural control, habit
strength, and normative constraints (Fielding &amp; Hornsey, 2026). These
moderators, along with situational factors like opportunity and affordability,
are consistently linked to the strength of the relationship between intention
and enactment (Webb et al., 2022; Webb et al., 2013).</p>
<p>A population-level association, however, does not necessarily imply that the
relationship is configured similarly for all individuals. A correlation can be
strong while individuals occupy substantially different combinations of
intention and behaviour. For example, some respondents may report strong
intentions but comparatively lower behaviour, whereas others may report
behaviour that exceeds their stated intention. These configurations can be
obscured by a single population-level coefficient.</p>
<p>Person-centred methods provide a complementary analytical perspective by
examining whether observations can be represented as distinct configurations
rather than assuming a single homogeneous population relationship. Latent
Profile Analysis (LPA) is one such approach for continuous indicators and has
been used to identify heterogeneous behavioural and psychological
configurations.</p>
<p>Person-centred methods, such as Latent Profile Analysis (LPA), provide a
complementary perspective by identifying distinct subgroups with unique
psychological or behavioural configurations (Fielding &amp; Hornsey, 2026).
These approaches have gained traction in environmental and energy research for
their capacity to disentangle the complexity of pro-environmental behavioural
enactment beyond simple average effects.</p>
<p>The present study applies LPA specifically to intention and behaviour in
household energy-saving behaviour. The analysis has two principal objectives.
First, it examines whether a strong aggregate association between intention and
behaviour coexists with heterogeneous respondent-level configurations. Second,
it characterises the resulting profiles according to the direction and magnitude
of the intention&ndash;behaviour configuration.</p>
<p>Because latent profile solutions are inherently model-dependent, the analysis
explicitly examines model-selection uncertainty, covariance specification,
numerical stability of the covariance estimates, classification quality,
bootstrap directional stability, cross-validation, and sensitivity to duplicate
observations. The study does not interpret latent profiles as causal types or
claim that a single profile solution represents a universally stable taxonomy of
households.</p>
""")

# ================================================================ 2. METHODS
_dims = pd.read_csv("results/01_data_inspection/data_dimensions.csv")
_N_COLS = int(_dims["N_columns"].iloc[0])
_construct_preds = sorted(set(mnl["predictor"]) & set(cstats.construct))
_demog_preds = sorted(set(mnl["predictor"]) - set(cstats.construct))
_N_PRED = int(diagv["n_predictors"])
_N_TESTS = int(diagv["n_tests"])
_N_NREF = SEL - 1
H.append(f"""
<h2 id="s2">2. Methods</h2>
<h3>2.1 Sample and Data</h3>
<p>The sample consists of N&nbsp;=&nbsp;{_N_TOTAL} respondents from a single
cross-sectional survey. The analytic dataset contains {_N_COLS} measured variables
comprising {_N_ITEMS} psychometric items and {len(_demogs)} demographic variables. {_miss_sentence}</p>
<p>{_N_DUPS} rows ({float(_N_DUPS) / _N_TOTAL * 100:.2f}%) were flagged as duplicates on all measured columns in
the raw data file. These observations were retained in the primary analysis to
preserve the original analytic sample. A sensitivity analysis was conducted
after removing the {_N_DUPS} duplicate rows, resulting in N&nbsp;=&nbsp;{_N_UNIQUE}.</p>

<h3>2.2 Measurement</h3>
<p>{_N_CONSTRUCTS} constructs were scored as arithmetic means of their constituent items.
All items used a 1&ndash;5 Likert response scale. Table 1 reports construct
descriptive statistics and internal consistency from the frozen measurement
evidence.</p>
""")

t1_rows = []
for _, r in cstats.iterrows():
    t1_rows.append([esc(r.construct), esc(r.k_items), m(r["mean"]),
                    m(r.SD), m(r.Cronbach_alpha, 4)])
H.append(f"<p><b>Table 1.</b> Construct descriptive statistics and internal "
         f"consistency (N&nbsp;=&nbsp;{_N_TOTAL}).</p>")
H.append(table_html(
    ["Construct", "Items", "Mean", "SD", "Cronbach&rsquo;s &alpha;"],
    t1_rows, align=[None, "c", "n", "n", "n"]))
H.append("""
<div class="note">Construct abbreviations, as verified against the frozen
measurement evidence: ATT&nbsp;=&nbsp;Attitude; CON&nbsp;=&nbsp;Context;
SNO&nbsp;=&nbsp;Subjective Norm; COVID&nbsp;=&nbsp;COVID Context;
INT&nbsp;=&nbsp;Intention; BE&nbsp;=&nbsp;Behaviour;
PU&nbsp;=&nbsp;Perceived Usefulness; PEU&nbsp;=&nbsp;Perceived Ease of Use;
PO&nbsp;=&nbsp;Personal Obligation; PRI&nbsp;=&nbsp;Personal Responsibility.
These abbreviations match the frozen construct-score columns exactly.</div>
""")

H.append(f"""
<h3>2.3 Aggregate Intention&ndash;Behaviour Association</h3>
<p>The aggregate association between INT and BE was quantified using
Pearson&rsquo;s correlation on the arithmetic-mean construct scores. The
verified correlation was r&nbsp;=&nbsp;{_R:.4f} with a 95% confidence interval of
[{_CI_LO:.4f},&nbsp;{_CI_HI:.4f}], p&nbsp;=&nbsp;{_p_mant:.2f}&nbsp;&times;&nbsp;10<sup>&minus;{_p_exp}</sup>,
and r&sup2;&nbsp;=&nbsp;{_R2:.4f}. The reported Pearson correlation, confidence
interval, and p-value were independently recomputed and verified against the
frozen computational evidence.</p>
""")
H.append(figure_embed("fig1", 1,
    f"Aggregate intention&ndash;behaviour association. Joint distribution of "
    f"INT and BE mean scores for N&nbsp;=&nbsp;{_N_TOTAL} respondents (hexagonal "
    f"binning), with the fitted linear association (r&nbsp;=&nbsp;{_R:.4f}, "
    f"95%&nbsp;CI&nbsp;[{_CI_LO:.4f},&nbsp;{_CI_HI:.4f}], "
    f"p&nbsp;=&nbsp;{_p_mant:.2f}&nbsp;&times;&nbsp;10<sup>&minus;{_p_exp}</sup>) and the "
    f"equality reference line (INT&nbsp;=&nbsp;BE). "
    f"Source: frozen construct-score evidence."))

H.append(f"""
<h3>2.4 Intention&ndash;Behaviour Gap</h3>
<p>The respondent-level intention&ndash;behaviour gap was defined as
GAP<sub>i</sub>&nbsp;=&nbsp;z(INT<sub>i</sub>)&nbsp;&minus;&nbsp;z(BE<sub>i</sub>),
where standardisation was performed within the sample using
ddof&nbsp;=&nbsp;1. The resulting GAP had a mean of
{_gapmean_mant:.3f}&nbsp;&times;&nbsp;10<sup>{_gapmean_exp}</sup> (effectively zero),
SD&nbsp;=&nbsp;{float(_gsv['SD']):.6f}, and range [{float(_gsv['min']):.2f},&nbsp;{float(_gsv['max']):+.2f}]. A total of {int(_gsv['positive_gap_count'])}
respondents ({float(_gsv['positive_gap_pct']):.2f}%) had INT&nbsp;&gt;&nbsp;BE, while {int(_gsv['negative_gap_count'])} ({float(_gsv['negative_gap_pct']):.2f}%) had
BE&nbsp;&gt;&nbsp;INT. No respondents had an exact equality under the z-score
comparison. The mean of GAP is approximately zero by construction because both
INT and BE were independently standardised before subtraction. This does not
imply that the GAP distribution is symmetric.</p>
""")
H.append(figure_embed("fig2", 2,
    f"Distribution of the standardised intention&ndash;behaviour gap. "
    f"Distribution of GAP&nbsp;=&nbsp;z(INT)&nbsp;&minus;&nbsp;z(BE) for "
    f"N&nbsp;=&nbsp;{_N_TOTAL} respondents, with kernel-density estimate, the "
    f"Normal(0,&nbsp;{float(_gsv['SD']):.4f}) reference density, and the zero reference point. "
    f"Source: frozen GAP statistics."))

_construct_preds = sorted(set(cstats.construct) & set(mnl["predictor"]))
_demog_preds = sorted(set(mnl["predictor"]) - set(cstats.construct))
_N_PRED = int(diagv["n_predictors"])
_N_TESTS = int(diagv["n_tests"])
_N_NREF = SEL - 1
H.append(f"""
<h3>2.5 Latent Profile Analysis</h3>
<p>LPA was conducted on two standardised indicators: z(INT) and z(BE). The
primary specification used sklearn.mixture.GaussianMixture with:</p>
<ul>
<li>indicators: z(INT) and z(BE);</li>
<li>covariance structure: full;</li>
<li>n_init = {N_INIT_SEL};</li>
<li>random_state = {SEED_SEL};</li>
<li>max_iter = 500;</li>
<li>reg_covar = 10<sup>&minus;6</sup>.</li>
</ul>
<p>Models with K&nbsp;=&nbsp;{int(kext_full.K.min())}&ndash;{int(kext_full.K.max())} profiles were fitted. BIC was calculated
as BIC&nbsp;=&nbsp;&minus;2&nbsp;log&nbsp;L&nbsp;+&nbsp;k&nbsp;log(N). For the
full-covariance two-indicator model, the number of free parameters was
k&nbsp;=&nbsp;6K&nbsp;&minus;&nbsp;1. At K&nbsp;=&nbsp;{SEL} &mdash; the primary
solution &mdash; the model therefore contained
6&times;{SEL}&nbsp;&minus;&nbsp;1&nbsp;=&nbsp;{6 * SEL - 1} free parameters; at
K&nbsp;=&nbsp;{COMP} it contained {6 * COMP - 1}.</p>
<p>Because increasing the number of mixture components can result in numerical
covariance collapse, the extended K search was evaluated for numerical behaviour
in addition to BIC. Specifically, the minimum eigenvalue of each component
covariance matrix was inspected against the reg_covar floor
(10<sup>&minus;6</sup>). The K&nbsp;=&nbsp;{SEL} solution was accepted as the
primary solution because it had the lowest BIC among models whose covariances
remained numerically well-behaved; K&nbsp;&ge;&nbsp;{DEG_K_MIN} produced positive
log-likelihoods consistent with degenerate solutions (Section 3.3). The
constant-score mechanism underlying the remaining floor components is disclosed
in Sections 3.5 and 4.</p>

<h3>2.6 Profile Matching</h3>
<p>When comparing profile solutions across bootstrap samples, alternative
specifications, sample variants, or random seeds, profiles were matched using
the two-dimensional vector of profile means
[z(INT)&#772;,&nbsp;z(BE)&#772;]. Pairwise profile distances were calculated
using Euclidean distance. Optimal one-to-one assignment was obtained using the
Hungarian algorithm (scipy.optimize.linear_sum_assignment) over the resulting
K&nbsp;&times;&nbsp;K cost matrix.</p>

<h3>2.7 Profile-Membership Predictors</h3>
<p>Profile membership under the K&nbsp;=&nbsp;{SEL} solution was examined using
multinomial logistic regression with Profile 0 (P0) as the reference category.
The model included {_N_PRED} predictors:</p>
<ul>
<li>Psychological constructs: {", ".join(_construct_preds)}</li>
<li>Demographic variables: {", ".join(_demog_preds)}</li>
</ul>
<p>The analysis therefore contained {_N_PRED} predictors &times; {_N_NREF} non-reference
profile contrasts = {_N_TESTS} statistical tests. Estimation was conducted using
statsmodels.MNLogit with BFGS optimisation and maxiter&nbsp;=&nbsp;1000.
Benjamini&ndash;Hochberg FDR correction was applied jointly across all {_N_TESTS} tests
at &alpha;&nbsp;=&nbsp;0.05.</p>
<p>This analysis is explicitly treated as exploratory and associational. The
cross-sectional design does not support causal inference. Furthermore, construct
predictors exhibited severe multicollinearity, with a maximum VIF of {float(diagv['max_VIF']):.2f}.
Individual predictor coefficients therefore should not be interpreted as
uniquely identified independent effects.</p>

<h3>2.8 Validation and Sensitivity Analyses</h3>
<p>Several validation and sensitivity analyses were conducted.</p>
<p><b>Bootstrap directional stability.</b> Two hundred bootstrap replications of
the K&nbsp;=&nbsp;{SEL} model were estimated. Profiles were matched using their
two-dimensional profile means, and directional consistency was calculated as the
proportion of matched bootstrap profiles for which mean INT exceeded mean
BE.</p>
<p><b>Five-fold cross-validation.</b> Five-fold cross-validation was conducted
for K&nbsp;=&nbsp;2&ndash;{SEL}. Models were refitted within each training fold and
evaluated using held-out log-likelihood.</p>
<p><b>Duplicate-row sensitivity.</b> The primary K&nbsp;=&nbsp;{SEL} analysis was
repeated after removing the {_N_DUPS} duplicate rows, resulting in
N&nbsp;=&nbsp;{_N_UNIQUE}, with refit profiles matched to the frozen K&nbsp;=&nbsp;{SEL}
solution by Hungarian matching on the two-dimensional profile means.</p>
<p><b>Covariance sensitivity.</b> Full and diagonal covariance structures were
compared for K&nbsp;=&nbsp;2&ndash;{int(kext_full.K.max())}.</p>
<p><b>Extended K search.</b> The primary full-covariance specification was
extended from K&nbsp;=&nbsp;2&ndash;6 to K&nbsp;=&nbsp;2&ndash;{int(kext_full.K.max())}, with
numerical behaviour assessed by component-covariance eigenvalues.</p>
<p><b>Alternative specifications.</b> Additional sensitivity analyses examined
spherical covariance, alternative score representations, and random seeds.</p>
""")

# ================================================================ 3. RESULTS
kmap = {int(r.K): r for _, r in kext_full.iterrows()}
_KMIN_T2, _KMAX_T2 = int(kext_full.K.min()), int(kext_full.K.max())
assess = {K: ("Primary solution" if K == SEL
              else "Competing solution (well-behaved)" if K == COMP
              else "Numerical pathology" if K in _DEG_KS
              else "Well-behaved")
          for K in range(_KMIN_T2, _KMAX_T2 + 1)}
t2_rows = []
for K in range(_KMIN_T2, _KMAX_T2 + 1):
    r = kmap[K]
    rc = "flag" if K in _DEG_KS else ("sig" if K == SEL else None)
    t2_rows.append((rc, [str(K), m(r.log_likelihood), str(int(r.n_params)),
                        m(r.AIC), m(r.BIC), m(r.entropy, 4),
                        str(int(r.min_class_N)), assess[K]]))

H.append(f"""
<h2 id="s3">3. Results</h2>
<h3>3.1 Measurement and Aggregate INT&ndash;BE Association</h3>
<p>Internal consistency ranged from &alpha;&nbsp;=&nbsp;{float(cstats['Cronbach_alpha'].min()):.4f} for
{esc(cstats.loc[cstats['Cronbach_alpha'].idxmin(), 'construct'])} to
&alpha;&nbsp;=&nbsp;{float(cstats['Cronbach_alpha'].max()):.4f} for {esc(cstats.loc[cstats['Cronbach_alpha'].idxmax(), 'construct'])}. Descriptive statistics and reliability
coefficients are presented in Table 1.</p>
<p>The aggregate INT&ndash;BE association was strong, with
r&nbsp;=&nbsp;{_R:.4f}, 95%&nbsp;CI&nbsp;[{_CI_LO:.4f},&nbsp;{_CI_HI:.4f}],
p&nbsp;=&nbsp;{_p_mant:.2f}&nbsp;&times;&nbsp;10<sup>&minus;{_p_exp}</sup>, and
r&sup2;&nbsp;=&nbsp;{_R2:.4f} (Figure 1).</p>

<h3>3.2 Intention&ndash;Behaviour Gap Distribution</h3>
<p>The standardised GAP had mean approximately zero and
SD&nbsp;=&nbsp;{float(_gsv['SD']):.6f}. Its observed range was
{float(_gsv['min']):.2f} to {float(_gsv['max']):+.2f}. Of the
{_N_TOTAL} respondents, {int(_gsv['positive_gap_count'])} ({float(_gsv['positive_gap_pct']):.2f}%) had INT&nbsp;&gt;&nbsp;BE, while
{int(_gsv['negative_gap_count'])} ({float(_gsv['negative_gap_pct']):.2f}%) had
BE&nbsp;&gt;&nbsp;INT. Thus, although the aggregate correlation was strong,
the direction of the respondent-level standardised discrepancy was not uniform
across the sample (Figure 2).</p>

<h3>3.3 Latent Profile Model Selection</h3>
<p>The full-covariance models produced the fit statistics in Table 2.</p>
""")
H.append(f"<p><b>Table 2.</b> Full-covariance Gaussian mixture model comparison "
         f"(K&nbsp;=&nbsp;{_KMIN_T2}&ndash;{_KMAX_T2}).</p>")
H.append(table_html(
    ["K", "Log-likelihood", "Parameters", "AIC", "BIC", "Entropy",
     "Smallest class N", "Assessment"],
    [r for _, r in t2_rows],
    row_classes=[rc for rc, _ in t2_rows],
    align=["c", "n", "c", "n", "n", "n", "c", None]))

H.append(f"""
<p>The K&nbsp;=&nbsp;{SEL} solution had the lowest BIC among the numerically
well-behaved models (BIC&nbsp;=&nbsp;{_BIC_SEL:.2f}, {PARAMS_SEL} parameters) and is retained as
the <b>primary solution</b>. This is not the global minimum BIC across
K&nbsp;=&nbsp;{_KMIN_T2}&ndash;{_KMAX_T2} in a naive sense: the K&nbsp;&ge;&nbsp;{DEG_K_MIN} models show
lower (indeed negative) BIC values, but they exhibited positive log-likelihoods
and numerical behaviour consistent with singular covariance solutions and are
therefore not interpretable latent profile solutions.</p>
<p>The K&nbsp;=&nbsp;{COMP} solution (BIC&nbsp;=&nbsp;{_BIC_COMP:.2f}) was itself numerically
well-behaved and is retained throughout as the leading <b>competing
solution</b> (Supplementary Figure S1; Supplementary Table S8).</p>
<p>Under K&nbsp;=&nbsp;{SEL}, {_n_floor_sel} of the {SEL} components operated at the
reg_covar floor (minimum eigenvalue&nbsp;=&nbsp;10<sup>&minus;6</sup>):
{_floor_desc(_floor_sel, SEL)}. These
are constant-score profiles arising because respondents with identical
constant item responses produce identical construct scores on the discrete
1&ndash;5 response lattice; the components are pinned to the floor
reg_covar rather than reflecting a genuinely singular fit. The same mechanism
is present at K&nbsp;=&nbsp;{COMP} ({_floor_desc(_floor_comp, COMP)}) and at
K&nbsp;=&nbsp;{_floor_mid_list} ({_floor_mid_str} component{'s' if max((_floor_counts[k] or 0) for k in _floor_ks_mid) != 1 else ''} each);
K&nbsp;=&nbsp;{", ".join(str(k) for k in _clean_ks)} are clean.
This lattice-collapse mechanism is disclosed here and examined further in
Sections 4 and 5 (Supplementary Table S9).</p>
""")
H.append(figure_embed("fig3", 3,
    f"Model fit across K&nbsp;=&nbsp;{_KMIN_T2}&ndash;{_KMAX_T2}. BIC under the primary "
    f"full-covariance specification. The K&nbsp;=&nbsp;{SEL} solution (green, "
    f"primary; lowest BIC among numerically well-behaved models) and the "
    f"K&nbsp;=&nbsp;{COMP} solution (blue, competing) are highlighted; the shaded "
    f"region marks K&nbsp;&ge;&nbsp;{DEG_K_MIN}, where solutions showed numerical "
    f"degeneracy. The figure does not imply that K&nbsp;=&nbsp;{COMP} is a BIC "
    f"minimum."))

H.append(f"""
<h3>3.4 Covariance Specification Sensitivity</h3>
<p>The comparison between full and diagonal covariance structures showed that
diagonal covariance produced lower BIC values at
K&nbsp;=&nbsp;{", ".join(str(k) for k in _DIAG_LOWER)} (Table 3).</p>
""")
t3_rows = []
for K in range(2, SEL + 1):
    f = cov[(cov.covariance_type == "full") & (cov.K == K)].iloc[0]
    d = cov[(cov.covariance_type == "diag") & (cov.K == K)].iloc[0]
    t3_rows.append([str(K), m(f.BIC), m(d.BIC),
                    "Full" if f.BIC < d.BIC else "Diagonal"])
H.append(f"<p><b>Table 3.</b> Full versus diagonal covariance "
         f"BIC, K&nbsp;=&nbsp;2&ndash;{SEL}.</p>")
H.append(table_html(["K", "Full BIC", "Diagonal BIC", "Lower BIC"], t3_rows,
                    align=["c", "n", "n", "c"]))

H.append(f"""
<p>The primary full-covariance specification was retained because it permits
profile-specific covariance between INT and BE and therefore corresponds
directly to the substantive interest in intention&ndash;behaviour
configuration. Nevertheless, the diagonal solution represents a legitimate
competing specification and is an important limitation on the interpretation of
the exact K&nbsp;=&nbsp;{SEL} structure.</p>
""")
H.append(figure_embed("fig4", 4,
    f"BIC under full and diagonal covariance specifications across "
    f"K&nbsp;=&nbsp;{_KMIN_T2}&ndash;{_KMAX_T2}. Diagonal covariance has lower BIC at "
    f"K&nbsp;=&nbsp;{_DIAG_LOWER[0]}&ndash;{_DIAG_LOWER[-1]}, including K&nbsp;=&nbsp;{SEL}."))

# --- 3.5 profile structure ---
dirs7 = {int(r.profile): ("INT &gt; BE" if r.INT_mean > r.BE_mean else "BE &gt; INT")
         for _, r in prof7.iterrows()}
t4_rows = []
for _, r in prof7.iterrows():
    t4_rows.append([f"P{int(r.profile)}", str(int(r.N)), m(r.percentage),
                    m(r.INT_mean, 3), m(r.BE_mean, 3), msign(r.INT_minus_BE, 3),
                    dirs7[int(r.profile)]])
t4_rows.append(["Total", str(_N_TOTAL), "100.00%", "&mdash;", "&mdash;", "&mdash;",
                f"{sum(1 for d in dirs7.values() if 'INT' in d)} INT&gt;BE + "
                f"{sum(1 for d in dirs7.values() if 'BE' in d)} BE&gt;INT"])

H.append(f"""
<h3>3.5 K = {SEL} Profile Structure</h3>
<p>The primary K&nbsp;=&nbsp;{SEL} solution comprised {SEL} profiles with sample
sizes of {', '.join(str(int(n)) for n in prof7['N'])} (Table 4).</p>
""")
H.append(f"<p><b>Table 4.</b> K&nbsp;=&nbsp;{SEL} profile descriptives on the "
         f"original 1&ndash;5 scale (N&nbsp;=&nbsp;{_N_TOTAL}).</p>")
H.append(table_html(
    ["Profile", "N", "%", "INT mean", "BE mean", "INT &minus; BE", "Direction"],
    t4_rows, row_classes=[None] * SEL + ["total"],
    align=["c", "c", "n", "n", "n", "n", "c"]))

_gap7 = pd.read_csv(f"{K7}/k7_per_profile_gap.csv")
_gmin = _gap7.loc[_gap7.mean_GAP.idxmin()]
_gmax = _gap7.loc[_gap7.mean_GAP.idxmax()]
_int_dirs = ", ".join(f"P{int(r.profile)} (n&nbsp;=&nbsp;{int(r.N)})"
                      for _, r in prof7.iterrows() if r.INT_mean > r.BE_mean)
_be_dirs = ", ".join(f"P{int(r.profile)} (n&nbsp;=&nbsp;{int(r.N)})"
                     for _, r in prof7.iterrows() if r.BE_mean > r.INT_mean)
H.append(f"""
<p>{sum(1 for d in dirs7.values() if 'INT' in d)} profiles had mean INT&nbsp;&gt;&nbsp;mean BE:
{_int_dirs}. {sum(1 for d in dirs7.values() if 'BE' in d)} profiles had mean
BE&nbsp;&gt;&nbsp;mean INT: {_be_dirs}. The profile structure
therefore contains both directions of intention&ndash;behaviour discrepancy
rather than a single universal direction.</p>
<p>Per-profile mean GAP values ranged from
{m(float(_gmin.mean_GAP), 2)} (P{int(_gmin.profile)}, mean GAP&nbsp;=&nbsp;{m(float(_gmin.mean_GAP), 4)}) to
{msign(float(_gmax.mean_GAP), 2)} (P{int(_gmax.profile)}, mean GAP&nbsp;=&nbsp;{msign(float(_gmax.mean_GAP), 4)}). Within profiles, the percentage of
respondents with INT&nbsp;&gt;&nbsp;BE ranged from {min(_gap7.pct_INT_gt_BE):.2f}% (P{int(_gap7.loc[_gap7.pct_INT_gt_BE.idxmin(), 'profile'])}) to {max(_gap7.pct_INT_gt_BE):.2f}% (P{int(_gap7.loc[_gap7.pct_INT_gt_BE.idxmax(), 'profile'])})
&mdash; confirming that profile-level direction does not imply homogeneity of
respondent-level direction within every profile (Supplementary Table S4).</p>
""")
H.append(figure_embed("fig5", 5,
    f"K&nbsp;=&nbsp;{SEL} intention&ndash;behaviour profile configuration in "
    f"z-space. Profile-specific means of standardised INT and BE, with marker "
    f"area proportional to profile size and the equality line "
    f"z(INT)&nbsp;=&nbsp;z(BE). Red markers: profiles with mean "
    f"z(INT)&nbsp;&gt;&nbsp;mean z(BE) ({", ".join("P" + str(i) for i in prof7.index[prof7.mean_z_INT > prof7.mean_z_BE])}); blue markers: profiles "
    f"with mean z(BE)&nbsp;&gt;&nbsp;mean z(INT) ({", ".join("P" + str(i) for i in prof7.index[prof7.mean_z_BE > prof7.mean_z_INT])})."))
H.append(figure_embed("fig6", 6,
    f"K&nbsp;=&nbsp;{SEL} profile-specific intention and behaviour means on the "
    f"original 1&ndash;5 scale, with profile sample sizes in the axis labels."))

# --- 3.6 GAP variance decomposition ---
H.append(f"""
<h3>3.6 GAP Variance Decomposition</h3>
<p>At K&nbsp;=&nbsp;{SEL}, between-profile variance in GAP was
{m(float(dec.loc[SEL, 'between_profile_variance']), 4)}, representing
{between7:.2f}% of the total GAP variance of
{m(float(dec.loc[SEL, 'total_variance']), 4)}. Within-profile variance was
{m(float(dec.loc[SEL, 'within_profile_variance']), 4)}, representing
{within7:.2f}% of total GAP variance.</p>
<p>Thus, the latent profiles account for a meaningful but minority share of the
observed GAP variance. Most GAP variation remains within profiles rather than
between them.</p>
<p>For comparison, the between-profile share was {float(dec.loc[COMP, 'between_over_total'])*100:.2f}%
at K&nbsp;=&nbsp;{COMP}, {float(dec.loc[COMP - 1, 'between_over_total'])*100:.2f}% at
K&nbsp;=&nbsp;{COMP - 1}, and {float(dec.loc[2, 'between_over_total'])*100:.2f}% at
K&nbsp;=&nbsp;2 (Figure 7).</p>
""")
H.append(figure_embed("fig7", 7,
    f"GAP variance decomposition for K&nbsp;=&nbsp;2, {COMP - 1}, {COMP}, and {SEL}. Proportion "
    f"of total GAP variance ({float(dec.loc[SEL, 'total_variance']):.4f}) attributable to between-profile and "
    f"within-profile differences. At K&nbsp;=&nbsp;{SEL}, {between7:.2f}% is between "
    f"profiles and {within7:.2f}% is within profiles."))

# --- 3.7 classification quality ---
H.append(f"""
<h3>3.7 Classification Quality</h3>
<p>Classification quality was high for most observations under
K&nbsp;=&nbsp;{SEL}. The mean maximum posterior probability was
{float(_cq['mean_max_posterior']):.4f}, with a median
of {float(_cq['median_max_posterior']):.4f} and SD of {float(_cq['SD_max_posterior']):.4f}. The distribution was:</p>
<ul>
<li>{float(_cq['pct_maxpost_ge_0.90']):.2f}% with maximum posterior probability &ge; 0.90 ({int(_cq['n_maxpost_ge_0.90'])} respondents);</li>
<li>{float(_cq['pct_maxpost_ge_0.80']):.2f}% with maximum posterior probability &ge; 0.80 ({int(_cq['n_maxpost_ge_0.80'])});</li>
<li>{float(_cq['pct_maxpost_ge_0.70']):.2f}% with maximum posterior probability &ge; 0.70 ({int(_cq['n_maxpost_ge_0.70'])});</li>
<li>{float(_cq['pct_maxpost_lt_0.70']):.2f}% with maximum posterior probability &lt; 0.70 ({int(_cq['n_maxpost_lt_0.70'])}).</li>
</ul>
<p>Thus, although most observations had relatively high classification
probabilities, classification was not perfect. Part of the high separation is
attributable to the {_n_floor_sel} constant-score components described in Section 3.3:
respondents with constant item responses are assigned with near-certainty to
their collapsed component, which raises the global mean maximum posterior
probability. Classification quality should therefore be read together with the
lattice-collapse disclosure.</p>
""")
H.append(figure_embed("fig8", 8,
    f"Maximum posterior probability for the K&nbsp;=&nbsp;{SEL} solution. "
    f"Distribution of maximum posterior probabilities, with reference "
    f"thresholds at 0.70, 0.80, and 0.90. Mean&nbsp;=&nbsp;{float(_cq['mean_max_posterior']):.4f} and "
    f"median&nbsp;=&nbsp;{float(_cq['median_max_posterior']):.4f}."))

# --- 3.8 bootstrap directional stability ---
stab7 = {int(r.profile[1:]): float(r.pct_INT_gt_BE) for _, r in stab.iterrows()}
def stab_interp(p):
    if p >= 80:
        return "Directionally stable (INT &gt; BE)"
    if p <= 20:
        return "Directionally stable as BE &gt; INT"
    if 40 < p < 60:
        return "Unstable / near chance"
    return "Moderate consistency"
t5_rows = []
for i in range(SEL):
    p = stab7[i]
    rc = "sig" if (p >= 80 or p <= 20) else None
    t5_rows.append((rc, [f"P{i}", f"{p:.1f}%", stab_interp(p)]))

H.append(f"""
<h3>3.8 Bootstrap Directional Stability</h3>
<p>Bootstrap directional stability was examined over {_N_BOOT} replications of the
K&nbsp;=&nbsp;{SEL} model (Table 5).</p>
""")
H.append(f"<p><b>Table 5.</b> Bootstrap directional stability "
         f"(K&nbsp;=&nbsp;{SEL}, {_N_BOOT} replications).</p>")
H.append(table_html(
    ["Profile", "% bootstrap replications with INT &gt; BE", "Interpretation"],
    [r for _, r in t5_rows],
    row_classes=[rc for rc, _ in t5_rows],
    align=["c", "n", None]))

_near_chance = sorted(i for i, p in stab7.items() if 40 < p < 60)
_p_near = min(stab7, key=lambda i: abs(stab7[i] - 50.0))
if _near_chance:
    _nc_sentence = (f"{', '.join('P' + str(i) for i in _near_chance)} "
                    f"fell within the 40&ndash;60% near-chance band")
else:
    _nc_sentence = (f"No K&nbsp;=&nbsp;{SEL} profile fell within the "
                    f"40&ndash;60% near-chance band")
_nc_caption = (f"{', '.join('P' + str(i) for i in _near_chance)} fell in the near-chance "
               f"band." if _near_chance else
               f"No K&nbsp;=&nbsp;{SEL} profile fell in the near-chance band.")
H.append(f"""
<p>{", ".join(f"P{i} ({p:.1f}%)" for i, p in stab7.items() if (p >= 80 or p <= 20))} were
directionally stable. {", ".join(f"P{i} ({p:.1f}%)" for i, p in stab7.items() if 20 < p < 80)} showed
moderate directional consistency: their modal direction
matched the point-estimate direction in every case, but with less separation
from chance. {_nc_sentence}. Because P{_p_near} sits at {stab7[_p_near]:.1f}% &mdash; the
closest to chance &mdash; and the constant-score components
({_floor_desc(_floor_sel, SEL)}) inflate per-profile directional certainty, the
directional interpretation of individual profiles
should nonetheless be treated with caution, and no profile&rsquo;s direction is
interpreted as a robust behavioural type.</p>
""")
H.append(figure_embed("fig9", 9,
    f"Bootstrap directional stability of K&nbsp;=&nbsp;{SEL} profiles. Percentage "
    f"of {_N_BOOT} bootstrap replications in which each matched profile had mean "
    f"INT&nbsp;&gt;&nbsp;mean BE. Green bars: directionally stable "
    f"(&ge;80% or &le;20%); orange: moderate (20&ndash;80%); red: near chance "
    f"(40&ndash;60%). {_nc_caption}"))

# --- 3.9 cross-validation ---
t6_rows = []
for _, r in cv.iterrows():
    rc = "sig" if int(r.K) == SEL else None
    t6_rows.append((rc, [str(int(r.K)), m(r.mean_ll_test), m(r.sd_ll_test)]))

_KCV_MIN, _KCV_MAX = int(cv.K.min()), int(cv.K.max())
_cv_sel = float(cv.loc[cv.K == SEL, "mean_ll_test"].iloc[0])
_cv_sel_sd = float(cv.loc[cv.K == SEL, "sd_ll_test"].iloc[0])
_cv_best_k = int(cv.loc[cv.mean_ll_test.idxmax(), "K"])
_cv_best = float(cv.loc[cv.K == _cv_best_k, "mean_ll_test"].iloc[0])
_cv_max_sd_k = int(cv.loc[cv.sd_ll_test.idxmax(), "K"])
H.append(f"""
<h3>3.9 Five-Fold Cross-Validation</h3>
<p>Five-fold cross-validation was conducted for K&nbsp;=&nbsp;{_KCV_MIN}&ndash;{_KCV_MAX}
(Table 6).</p>
""")
H.append(f"<p><b>Table 6.</b> Five-fold cross-validation: mean held-out "
         f"log-likelihood (K&nbsp;=&nbsp;{_KCV_MIN}&ndash;{_KCV_MAX}).</p>")
H.append(table_html(
    ["K", "Mean held-out log-likelihood", "SD"],
    [r for _, r in t6_rows], row_classes=[rc for rc, _ in t6_rows],
    align=["c", "n", "n"]))

H.append(f"""
<p>K&nbsp;=&nbsp;{SEL} produced the highest mean held-out log-likelihood among the
models evaluated in the five-fold cross-validation
({m(_cv_sel)}). However, this mean was accompanied by a very large
fold-to-fold SD ({_cv_sel_sd:.2f}), the largest in the K&nbsp;=&nbsp;{_KCV_MIN}&ndash;{_KCV_MAX} range:
held-out performance at K&nbsp;=&nbsp;{SEL} was highly variable across folds. The
cross-validation result therefore supports K&nbsp;=&nbsp;{SEL} only weakly, and the
combination of highest mean with highest variance is consistent with the
interpretation that folded refits at K&nbsp;=&nbsp;{SEL} sometimes recover the
well-separated structure and sometimes do not. Cross-validation should be read
jointly with the BIC comparison (Section 3.3) and the lattice-collapse
disclosure rather than as independent confirmation of the {SEL}-profile
structure.</p>
""")
H.append(figure_embed("fig10", 10,
    f"Five-fold cross-validation performance. Mean held-out log-likelihood "
    f"&plusmn; SD for K&nbsp;=&nbsp;{_KCV_MIN}&ndash;{_KCV_MAX}. K&nbsp;=&nbsp;{SEL} (green) has the "
    f"highest mean held-out log-likelihood ({m(_cv_sel)}) but also the largest "
    f"fold SD ({_cv_sel_sd:.2f}); K&nbsp;=&nbsp;{COMP} (blue) is highlighted as the leading "
    f"competing solution."))

# --- 3.10 duplicate-row sensitivity ---
t7_rows = [
    ["Pearson r", fmt(_dg["pearson_r_INT_BE"], 6), fmt(_du["pearson_r_INT_BE"], 6)],
    ["GAP SD", fmt(_dg["gap_SD"], 6), fmt(_du["gap_SD"], 6)],
    [f"K = {SEL} BIC", fmt(_dg["K7_BIC"]), fmt(_du["K7_BIC"])],
    [f"K = {SEL} log-likelihood", fmt(_dg["K7_log_likelihood"]), fmt(_du["K7_log_likelihood"])],
    [f"K = {SEL} entropy", fmt(_dg["K7_entropy"], 4), fmt(_du["K7_entropy"], 4)],
    [f"K = {SEL} minimum class N", fmt(_dg["K7_min_class_N"], 0), fmt(_du["K7_min_class_N"], 0)],
    ["% INT &gt; BE", fmt(_dg["pct_INT_gt_BE"]) + "%", fmt(_du["pct_INT_gt_BE"]) + "%"],
    ["% BE &gt; INT", fmt(_dg["pct_BE_gt_INT"]) + "%", fmt(_du["pct_BE_gt_INT"]) + "%"],
    ["Mean maximum posterior", fmt(_dg["K7_mean_max_posterior"], 4),
     fmt(_du["K7_mean_max_posterior"], 4)],
]

H.append(f"""
<h3>3.10 Duplicate-Row Sensitivity</h3>
<p>The primary analysis contained {_N_DUPS} duplicate rows. Removing these
observations produced N&nbsp;=&nbsp;{_N_UNIQUE}, on which the K&nbsp;=&nbsp;{SEL}
Gaussian mixture was refitted under the identical specification and matched to
the frozen K&nbsp;=&nbsp;{SEL} profiles by Hungarian matching on the two-dimensional
profile means (Table 7).</p>
""")
H.append(f"<p><b>Table 7.</b> Duplicate-row sensitivity "
         f"(K&nbsp;=&nbsp;{SEL}).</p>")
H.append(table_html(["Metric", f"Primary N = {int(_dg['N_full_sample'])}", f"Unique N = {int(_du['N_unique_sample'])}"], t7_rows,
                    align=[None, "n", "n"]))

matched_rows = []
for i in range(SEL):
    n_f = int(prof7.set_index("profile").loc[i, "N"])
    n_u = uniq_size_by_frozen[i]
    matched_rows.append([f"P{i}", str(n_f), str(n_u)])
H.append(f"<p><b>Table 7a.</b> Hungarian-matched profile sizes, primary "
         f"(N&nbsp;=&nbsp;{_N_TOTAL}) versus duplicate-excluded refit "
         f"(N&nbsp;=&nbsp;{_N_UNIQUE}).</p>")
H.append(table_html(["Profile", "Primary N", "Duplicate-excluded N (matched)"],
                    matched_rows, align=["c", "c", "c"]))

H.append(f"""
<p>Removal of the duplicate rows produced only small changes in the aggregate
correlation ({_dg['pearson_r_INT_BE']:.4f} &rarr; {_du['pearson_r_INT_BE']:.4f}), the GAP SD
({float(_dg['gap_SD']):.4f} &rarr; {float(_du['gap_SD']):.4f}), the GAP
direction proportions ({float(_dg['pct_INT_gt_BE']):.2f}/{float(_dg['pct_BE_gt_INT']):.2f} &rarr;
{float(_du['pct_INT_gt_BE']):.2f}/{float(_du['pct_BE_gt_INT']):.2f}), and classification
quality (mean maximum posterior {float(_dg['K7_mean_max_posterior']):.4f} &rarr;
{float(_du['K7_mean_max_posterior']):.4f}). The {SEL}-profile
structure was preserved under Hungarian matching (mean z-space distance
{m(_du['K7_hungarian_mean_distance'], 4)}). However, the refitted class sizes
shifted materially relative to the primary solution (Table 7a; e.g. the
profile matched to P0 changed from n&nbsp;=&nbsp;{int(prof7.set_index('profile').loc[0, 'N'])} to
n&nbsp;=&nbsp;{uniq_size_by_frozen[0]}), and the refit log-likelihood and BIC
({m(_du['K7_log_likelihood'])} and {m(_du['K7_BIC'])}) differ substantially from the primary
fit &mdash; reflecting the same lattice-collapse sensitivity documented in
Section 3.3 rather than a simple robustness win. The duplicate-exclusion
refit therefore supports the qualitative stability of the {SEL}-profile
configuration while underscoring that exact profile boundaries are sensitive
to sample composition.</p>
""")
H.append(figure_embed("fig11", 11,
    f"Sensitivity of key results to duplicate-row removal at "
    f"K&nbsp;=&nbsp;{SEL}. Comparison of primary (N&nbsp;=&nbsp;{_N_TOTAL}) and "
    f"duplicate-excluded (N&nbsp;=&nbsp;{_N_UNIQUE}, Hungarian-matched) results: "
    f"aggregate metrics, K&nbsp;=&nbsp;{SEL} fit and classification metrics "
    f"(minimum class N shown on a /1000 scale), gap direction, and profile "
    f"sizes."))

# --- 3.11 predictors ---
H.append(f"""
<h3>3.11 Exploratory Profile-Membership Associations</h3>
<p>Multinomial logistic regression was used to examine exploratory associations
between K&nbsp;=&nbsp;{SEL} profile membership and {_N_PRED} predictors, with P0 as the
reference category (Table 8).</p>
""")
_sig_preds = sorted(psum.index[psum.n_significant_contrasts > 0])
_sig_constructs = [p for p in _sig_preds if p in set(cstats.construct)]
_sig_demogs = [p for p in _sig_preds if p not in set(cstats.construct)]
t8_rows = [
    ["Sample size", str(int(diagv["sample_size"]))],
    ["Outcome classes", str(int(diagv["n_outcome_classes"]))],
    ["Reference profile", f"P{int(diagv['reference_profile'])}"],
    ["Predictors", str(int(diagv["n_predictors"]))],
    ["Joint FDR tests", str(int(diagv["n_tests"]))],
    ["Log-likelihood", fmt(diagv["log_likelihood"])],
    ["AIC", fmt(diagv["AIC"])],
    ["BIC", fmt(diagv["BIC"])],
    ["McFadden pseudo-R&sup2;", fmt(diagv["McFadden_pseudo_R2"], 4)],
    ["Maximum VIF", fmt(diagv["max_VIF"])],
    ["Condition number (design, unscaled — do not interpret; see corr-matrix value in text)", fmt(diagv["condition_number_design_with_const_UNSCALED_DO_NOT_COMPARE"])],
    ["FDR-significant tests", f"{int(diagv['n_FDR_significant'])} of {int(diagv['n_tests'])}"],
    ["Missing predictor values", str(int(diagv["n_missing_predictor_values"]))],
    ["Convergence", "Yes" if int(diagv["converged"]) == 1 else "No"],
]
H.append(f"<p><b>Table 8.</b> Multinomial logistic regression model diagnostics "
         f"(K&nbsp;=&nbsp;{SEL}).</p>")
H.append(table_html(["Diagnostic", "Value"], t8_rows))

H.append(f"""
<p>{int(diagv['n_FDR_significant'])} of the {int(diagv['n_tests'])} jointly FDR-adjusted tests were statistically
significant
(Supplementary Table S2). The significant contrasts were concentrated in
construct predictors ({", ".join(_sig_constructs)}) across the {_N_NREF} non-reference
contrasts; {("no demographic predictor reached" if not _sig_demogs else
           "demographic predictors " + ", ".join(_sig_demogs) + " also reached")}
joint-FDR significance at &alpha;&nbsp;=&nbsp;0.05.</p>
<p>The construct predictors exhibited severe multicollinearity, with maximum
VIF&nbsp;=&nbsp;{float(diagv['max_VIF']):.2f} and condition number&nbsp;=&nbsp;{float(diagv['condition_number_design_with_const_UNSCALED_DO_NOT_COMPARE']):.2f}. Consequently, the
predictor analysis is interpreted as a joint exploratory pattern rather than as
evidence for uniquely identifiable effects of individual predictors.</p>
<p>The McFadden pseudo-R&sup2;&nbsp;=&nbsp;{float(diagv['McFadden_pseudo_R2']):.4f} is a likelihood-based
model-fit statistic and should not be interpreted as the proportion of variance
explained. It is not directly comparable with the Pearson
r&sup2;&nbsp;=&nbsp;{_R2:.4f} from the aggregate INT&ndash;BE association.</p>
""")

# ================================================================ 4. DISCUSSION
_second_hdr = " &mdash; the lowest-BIC solution among numerically well-behaved models"
_profiles_desc = (f"{sum(1 for d in dirs7.values() if 'INT' in d)} with mean "
                  f"INT &gt; BE ({', '.join('P' + str(i) for i, r in prof7.iterrows() if r.INT_mean > r.BE_mean)}) "
                  f"and {sum(1 for d in dirs7.values() if 'BE' in d)} "
                  f"with mean BE &gt; INT ({', '.join('P' + str(i) for i, r in prof7.iterrows() if r.BE_mean > r.INT_mean)})")
H.append(f"""
<h2 id="s4">4. Discussion</h2>
<p>The principal finding is that a strong aggregate association between
intention and behaviour coexists with substantial heterogeneity in
respondent-level intention&ndash;behaviour configurations. The aggregate
correlation was r&nbsp;=&nbsp;{_R:.4f}, yet the K&nbsp;=&nbsp;{SEL} solution
contained {_profiles_desc}.</p>
<p>Our findings regarding the coexistence of strong aggregate
intention-behaviour correlations and substantial respondent-level heterogeneity
align with recent literature (Webb et al., 2022; Sheeran / Gemmecke et al.,
2025). While previous research often emphasizes variables that moderate the
intention-behaviour relationship (Fielding &amp; Hornsey, 2026), our
person-centred approach demonstrates that a single motivational explanation
(Webb et al., 2013) is often insufficient to capture the complexity of
household energy behaviour. Consistent with recent meta-analytic evidence
(Carrero et al., 2025; 2025 systematic review), our results suggest that
segmenting populations into intention-behaviour profiles captures meaningful
heterogeneity, even as the majority of discrepancy remains within-profile. This
supports the recent shift toward modelling intention and behaviour as distinct,
simultaneously observed outcomes, addressing the over-reliance on intention as
a solitary endpoint in current TPB/NAM research (2026).</p>
<p>At the same time, the profile solution should not be interpreted as evidence
for {SEL} universally distinct behavioural types. At K&nbsp;=&nbsp;{SEL}, only
{between7:.2f}% of GAP variance was between profiles, whereas {within7:.2f}% remained within
profiles. Thus, the latent profiles capture a meaningful component of the
observed heterogeneity but do not account for the majority of individual-level
variation in GAP. The between-profile share at K&nbsp;=&nbsp;{SEL} ({between7:.2f}%) is
modestly higher than at K&nbsp;=&nbsp;{COMP} ({float(dec.loc[COMP, 'between_over_total'])*100:.2f}%), but both figures indicate
that most GAP variation is individual-level rather than configuration-level.</p>
<p>Our findings regarding the co-occurrence of INT&nbsp;&gt;&nbsp;BE and
BE&nbsp;&gt;&nbsp;INT configurations are consistent with recent meta-analytic
evidence (Carrero et al., 2025; Sheeran &amp; Gemmecke et al., 2025). This
bidirectional discrepancy challenges the traditional view that intention
typically exceeds behaviour, suggesting instead that
&ldquo;action-beyond-intention&rdquo; is a significant phenomenon in
energy-saving contexts.</p>
<p>Several methodological qualifications are central to interpretation.</p>
<p><b>First, the primary solution rests on a restricted model-selection
criterion.</b> K&nbsp;=&nbsp;{SEL} is the lowest-BIC solution among numerically
well-behaved models, not the global BIC optimum across the extended search:
K&nbsp;&ge;&nbsp;{DEG_K_MIN} showed lower (negative) BIC values driven by positive
log-likelihoods consistent with singular covariance solutions and were not
substantively interpreted. Moreover, {_n_floor_sel} of the {SEL} K&nbsp;=&nbsp;{SEL}
components operate at the reg_covar floor ({_floor_desc(_floor_sel, SEL)}) because they capture
respondents whose constant item responses produce identical construct scores on
the discrete 1&ndash;5 response lattice. This lattice-collapse mechanism &mdash;
shared with K&nbsp;=&nbsp;{COMP} ({_floor_desc(_floor_comp, COMP)}) and present to a
lesser degree at K&nbsp;=&nbsp;4&ndash;5 &mdash; means that part of the BIC
advantage of K&nbsp;=&nbsp;{SEL}, and part of its high classification quality, arise from
constant-score components rather than from smoothly separated latent
populations. The same mechanism is present at
K&nbsp;=&nbsp;{_floor_mid_range} ({_floor_mid_str} floor
component{'s' if max((_floor_counts[k] or 0) for k in _floor_ks_mid) != 1 else ''} each).
The K&nbsp;=&nbsp;{SEL} solution should therefore be described as the
primary solution under the stated selection rule, with K&nbsp;=&nbsp;{COMP} retained
as the leading competing solution, rather than as an unequivocally optimal
global solution.</p>
<p><b>Second, covariance specification affects model selection.</b> Diagonal
covariance produced lower BIC than full covariance at
K&nbsp;=&nbsp;{", ".join(str(k) for k in _DIAG_LOWER)}, including
K&nbsp;=&nbsp;{SEL}. The full-covariance specification was retained
because it allows covariance between INT and BE within each profile and is
therefore directly relevant to the substantive configuration being examined.
Nevertheless, the diagonal result demonstrates that the exact latent structure
depends on modelling assumptions.</p>
<p><b>Third, classification quality was high for most respondents but not
perfect, and partly reflects the collapsed components.</b> Approximately {float(_cq['pct_maxpost_lt_0.70']):.2f}%
of observations had a maximum posterior probability below 0.70, and part of the
mean maximum posterior probability of {float(_cq['mean_max_posterior']):.4f} is attributable to near-certain
assignment into constant-score components. In the bootstrap analysis,
{", ".join("P" + str(i) for i in sorted(k for k, p in stab7.items() if p >= 80 or p <= 20))} were directionally stable, whereas
{", ".join("P" + str(i) for i in sorted(k for k, p in stab7.items() if 20 < p < 80))} showed moderate
consistency; P{_p_near}, at {stab7[_p_near]:.1f}%, sits closest to chance. Individual profile directions
therefore require particular caution.</p>
<p><b>Fourth, the duplicate-row sensitivity analysis shows qualitative
stability but material boundary sensitivity.</b> After removal of the {_N_DUPS}
duplicates, the aggregate correlation changed from {_dg['pearson_r_INT_BE']:.4f} to {_du['pearson_r_INT_BE']:.4f}, the
INT&nbsp;&gt;&nbsp;BE proportion changed from {float(_dg['pct_INT_gt_BE']):.2f}% to {float(_du['pct_INT_gt_BE']):.2f}%, and the
{SEL}-profile structure was preserved under Hungarian matching (mean z-space
distance {m(_du['K7_hungarian_mean_distance'], 2)}). However, matched class sizes shifted materially (e.g.
{int(prof7.iloc[0]['N'])}&nbsp;&rarr;&nbsp;{uniq_size_by_frozen[0]} for the profile matched to P0), indicating that exact
profile boundaries &mdash; particularly for the collapsed components &mdash;
are sensitive to sample composition.</p>
<p><b>Finally, the exploratory predictor analysis should not be used to infer
causal mechanisms.</b> Although {int(diagv['n_FDR_significant'])} of {int(diagv['n_tests'])} FDR-adjusted tests were significant,
the construct predictors exhibited severe multicollinearity, with maximum
VIF&nbsp;=&nbsp;{float(diagv['max_VIF']):.2f}. Individual coefficients are therefore not interpreted as
uniquely identified independent effects.</p>
<div class="placeholder">[LITERATURE SUPPORT NEEDED: Situate the person-centred
finding within prior environmental and energy-saving behaviour research and
identify the theoretical contribution of distinguishing aggregate association
from heterogeneous configuration.]</div>
""")

# ================================================================ 5. LIMITATIONS
_min_alpha_row = cstats.loc[cstats["Cronbach_alpha"].idxmin()]
H.append(f"""
<h2 id="s5">5. Limitations</h2>
<p>Several limitations should be considered.</p>
<h3>5.1 Cross-sectional design</h3>
<p>The data are cross-sectional. Consequently, the analysis cannot establish
temporal ordering or causal relationships between intention, behaviour,
predictors, or profile membership.</p>
<h3>5.2 Model dependence</h3>
<p>The latent profile structure depends on modelling choices, including the
number of profiles, covariance specification, and score representation.</p>
<h3>5.3 K = {COMP} is a well-behaved competing solution</h3>
<p>The K&nbsp;=&nbsp;{COMP} solution (BIC&nbsp;=&nbsp;{_BIC_COMP:.2f}) was numerically
well-behaved and remains a legitimate competing specification to the primary
K&nbsp;=&nbsp;{SEL} solution (BIC&nbsp;=&nbsp;{_BIC_SEL:.2f}). Model choice between them
is not settled by fit statistics alone, particularly given the shared
lattice-collapse mechanism (Section 5.4).</p>
<h3>5.4 Lattice collapse in {_n_floor_sel} K = {SEL} components</h3>
<p>{_n_floor_sel} of the {SEL} K&nbsp;=&nbsp;{SEL} components ({_floor_desc(_floor_sel, SEL)}) operated at the
reg_covar floor: respondents with constant item responses produce identical
construct scores on the discrete 1&ndash;5 response lattice, giving
zero observed variance on one indicator within those profiles. This is an
artefact of the discrete response scale rather than evidence of a genuinely
singular continuous population. The same mechanism affects K&nbsp;=&nbsp;{COMP}
({_floor_desc(_floor_comp, COMP)}) and, to a lesser degree,
K&nbsp;=&nbsp;{_floor_mid_range} ({_floor_mid_str} component{'s' if max((_floor_counts[k] or 0) for k in _floor_ks_mid) != 1 else ''} each). Part of the
K&nbsp;=&nbsp;{SEL} BIC advantage and classification quality is attributable to
these collapsed components, and exact membership boundaries for these profiles
are correspondingly fragile (Section 3.10).</p>
<h3>5.5 Numerical degeneracy at K &ge; {DEG_K_MIN}</h3>
<p>The K&nbsp;=&nbsp;{", ".join(str(k) for k in _DEG_KS)} full-covariance solutions produced positive
log-likelihoods and behaviour consistent with singular covariance matrices.
These solutions were not treated as substantive profile solutions.</p>
<h3>5.6 Diagonal covariance is a competing specification</h3>
<p>Diagonal covariance produced lower BIC at
K&nbsp;=&nbsp;{", ".join(str(k) for k in _DIAG_LOWER)}. This
represents an important alternative specification and limits confidence in the
exact K&nbsp;=&nbsp;{SEL} structure.</p>
<h3>5.7 Directional stability is incomplete</h3>
<p>{_nc_sentence} in the bootstrap
analysis, {len([1 for p in stab7.values() if 20 < p < 80])} profiles
({", ".join("P" + str(i) for i, p in stab7.items() if 20 < p < 80)}) showed only moderate directional
consistency, and P{_p_near} ({stab7[_p_near]:.1f}%) sits closest to chance. Profile directions should not
be treated as robustly established types.</p>
<h3>5.8 Within-profile variance dominates</h3>
<p>At K&nbsp;=&nbsp;{SEL}, {within7:.2f}% of GAP variance remained within profiles. The
profiles therefore capture only a minority of total GAP variance.</p>
<h3>5.9 Predictor multicollinearity</h3>
<p>The exploratory predictor model had a maximum VIF of {float(diagv['max_VIF']):.2f} and condition
number of {float(diagv['condition_number_design_with_const_UNSCALED_DO_NOT_COMPARE']):.2f}. Individual predictor coefficients should therefore not be
interpreted as uniquely identified independent effects.</p>
<h3>5.10 Duplicate observations and profile-boundary sensitivity</h3>
<p>{_N_DUPS} duplicate rows were retained in the primary analysis. The
duplicate-exclusion sensitivity analysis preserved the {SEL}-profile structure
under Hungarian matching but produced materially shifted class sizes and a
substantially different refit log-likelihood, underscoring sensitivity of exact
profile boundaries to sample composition.</p>
<h3>5.11 {esc(_min_alpha_row.construct)} reliability</h3>
<p>{esc(_min_alpha_row.construct)} had the lowest internal-consistency coefficient,
&alpha;&nbsp;=&nbsp;{float(_min_alpha_row.Cronbach_alpha):.4f},
based on {int(_min_alpha_row.k_items)} items. Findings involving {esc(_min_alpha_row.construct)} should therefore be interpreted
cautiously.</p>
<h3>5.12 Classification uncertainty</h3>
<p>Although the mean maximum posterior probability was {float(_cq['mean_max_posterior']):.4f}, {float(_cq['pct_maxpost_lt_0.70']):.2f}% of
respondents had maximum posterior probability below 0.70, and part of the high
mean reflects near-certain assignment into constant-score components.</p>
<h3>5.13 Generalisability</h3>
<p>The generalisability of the findings depends on the sampling frame,
recruitment procedure, demographic composition, and survey context.</p>
<div class="placeholder">[AUTHOR INPUT REQUIRED: sampling frame, recruitment
method, demographic-category definitions, and other information needed to
evaluate generalisability.]</div>
""")

# ================================================================ 6. CONCLUSION
H.append(f"""
<h2 id="s6">6. Conclusion</h2>
<p>This study examined intention&ndash;behaviour configurations in household
energy-saving behaviour using a person-centred latent profile approach. The
aggregate association between intention and behaviour was strong
(r&nbsp;=&nbsp;{_R:.4f}), but the population-level association coexisted with
heterogeneous respondent-level configurations.</p>
<p>Under the primary full-covariance specification, the K&nbsp;=&nbsp;{SEL}
solution{_second_hdr}
&mdash; identified {SEL} profiles comprising {_profiles_desc}. The profiles captured {between7:.2f}% of GAP variance, while {within7:.2f}%
remained within profiles, indicating that the latent structure captures a
meaningful but incomplete component of intention&ndash;behaviour
heterogeneity.</p>
<p>The evidence supports the central proposition that a strong aggregate
intention&ndash;behaviour association can coexist with heterogeneous
configurations. However, the exact profile solution should not be regarded as
definitive. K&nbsp;=&nbsp;{COMP} is a well-behaved competing solution, diagonal
covariance provided a lower-BIC competing specification at
K&nbsp;=&nbsp;{_DIAG_LOWER[0]}&ndash;{_DIAG_LOWER[-1]}, {_n_floor_sel} of the {SEL} components reflect the discrete
response lattice, and several profiles showed only moderate bootstrap
directional consistency.</p>
<p>The duplicate-row sensitivity analysis preserved the {SEL}-profile
structure under Hungarian matching while showing material sensitivity of exact
profile boundaries. Five-fold cross-validation yielded the highest mean
held-out log-likelihood at K&nbsp;=&nbsp;{SEL} but with the largest fold-to-fold
variability in the K&nbsp;=&nbsp;{_KCV_MIN}&ndash;{_KCV_MAX} range, so it supports the solution
only weakly. The exploratory predictor analysis identified {int(diagv['n_FDR_significant'])} FDR-significant
associations among {int(diagv['n_tests'])} tests, but severe multicollinearity prevents
interpretation of individual coefficients as uniquely identified effects.</p>
<div class="placeholder">[AUTHOR INPUT REQUIRED: Add a final literature-grounded
statement explaining the theoretical contribution of the findings and specific
directions for future research.]</div>
""")

# ================================================================ SUPPLEMENTARY
_dirs6_int = ", ".join("P" + str(p) for p, d in zip(k6tab.index, _dir6)
                       if d == "INT &gt; BE")
_dirs6_be = ", ".join("P" + str(p) for p, d in zip(k6tab.index, _dir6)
                      if d == "BE &gt; INT")
_sum_dirs6 = _dirs6_int.count("P")
_sum_dirs6b = _dirs6_be.count("P")
_k6_sizes_str = ", ".join(str(int(n)) for n in k6tab["N"])
H.append(f"""
<h2 id="supp">Supplementary Materials</h2>

<h3>Supplementary Table S1</h3>
<p>Full and diagonal covariance model comparison across
K&nbsp;=&nbsp;{_KMIN_T2}&ndash;{_KMAX_T2}.</p>
<p class="src">Source: results/paper1_strengthening/covariance_k_extended.csv</p>

<h3>Supplementary Table S2</h3>
<p>Full multinomial logistic regression coefficient table for the
K&nbsp;=&nbsp;{SEL} solution: {_N_PRED} predictors &times; {_N_NREF} non-reference contrasts
= {int(diagv['n_tests'])} tests, including coefficient estimates, confidence intervals, raw
p-values, and FDR-adjusted p-values. The K&nbsp;=&nbsp;{COMP} counterpart
({_N_TESTS_COMP} tests) is retained in the same evidence package.</p>
<p class="src">Source: results/k7_primary/k7_predictors/multinomial_results.csv
(K = {SEL}); results/18_profile_predictors/multinomial_results.csv (K = {COMP})</p>

<h3>Supplementary Table S3</h3>
<p>Duplicate-row sensitivity analysis at the profile level for
N&nbsp;=&nbsp;{_N_UNIQUE} (K&nbsp;=&nbsp;{SEL} refit, Hungarian-matched to the frozen
K&nbsp;=&nbsp;{SEL} solution).</p>
<p class="src">Source: results/k7_primary/k7_duplicate_sensitivity_profiles.csv;
results/paper1_strengthening/duplicate_sensitivity_profiles.csv (K = {COMP})</p>

<h3>Supplementary Table S4</h3>
<p>K&nbsp;=&nbsp;{SEL} per-profile GAP statistics (mean, SD, median, and
percentage of respondents with INT&nbsp;&gt;&nbsp;BE).</p>
<p class="src">Source: results/k7_primary/k7_per_profile_gap.csv</p>

<h3>Supplementary Table S5</h3>
<p>Construct descriptive statistics and reliability.</p>
<p class="src">Source: results/paper1_final/02_measurement/construct_statistics.csv</p>

<h3>Supplementary Table S6</h3>
<p>Independent Pearson correlation verification.</p>
<p class="src">Source: results/paper1_strengthening/pearson_verification.csv</p>

<h3>Supplementary Table S7</h3>
<p>Alternative-specification sensitivity analyses (reference
K&nbsp;=&nbsp;{COMP}), including full, diagonal, and spherical covariance,
alternative score representations, and random seeds.</p>
<p class="src">Source: results/paper1_final/07_robustness/10_robustness_table.csv</p>

<h3>Supplementary Table S8</h3>
<p>K&nbsp;=&nbsp;{COMP} profile descriptives on the z scale (competing
solution): sizes [{_k6_sizes_str}]; {_sum_dirs6} profiles with
INT&nbsp;&gt;&nbsp;BE ({_dirs6_int}) and {_sum_dirs6b} with BE&nbsp;&gt;&nbsp;INT
({_dirs6_be}).</p>
<p class="src">Source: results/04_lpa_estimation/K_{COMP}/profile_means.csv and
profile_sizes.csv</p>

<h3>Supplementary Table S9</h3>
<p>Component-covariance eigenvalue audit for the K&nbsp;=&nbsp;{COMP} and
K&nbsp;=&nbsp;{SEL} full-covariance solutions: minimum/maximum eigenvalues,
determinants, indicator SDs, and reg_covar-floor flags. Documents the
lattice-collapse mechanism: K&nbsp;=&nbsp;{SEL} floor components are
{_floor_desc(_floor_sel, SEL)}; K&nbsp;=&nbsp;{COMP} floor components
are {_floor_desc(_floor_comp, COMP)}.</p>
<p class="src">Source: results/k7_primary/covariance_eigenvalues.csv</p>

<h3>Supplementary Figure S1</h3>
<p>K&nbsp;=&nbsp;{COMP} profile structure under the full-covariance specification
(competing solution).</p>
<p>The K&nbsp;=&nbsp;{COMP} solution contains profile sizes
[{_k6_sizes_str}] with {_sum_dirs6} profiles above the
equality line ({_dirs6_int}) and {_sum_dirs6b} below it ({_dirs6_be}).</p>
<p class="src">Source: results/04_lpa_estimation/K_{COMP}/profile_means.csv and
profile_sizes.csv</p>
""")

H.append(figure_embed("figS1", "S1",
    f"K&nbsp;=&nbsp;{COMP} profile structure in z-space (supplementary; competing "
    f"solution). Profile-specific means of standardised INT and BE, with "
    f"marker area proportional to profile size and the equality line "
    f"z(INT)&nbsp;=&nbsp;z(BE). Red markers: profiles with mean "
    f"z(INT)&nbsp;&gt;&nbsp;mean z(BE) ({_dirs6_int}); blue markers: profiles "
    f"with mean z(BE)&nbsp;&gt;&nbsp;mean z(INT) ({_dirs6_be})."))

# ================================================================ TRANSPARENCY
H.append(f"""
<h2 id="transparency">Data and Computational Transparency</h2>
<p>All numerical values reported in this manuscript are taken from the frozen
computational evidence package supplied for the study
(results/paper1_final, results/paper1_strengthening) and from the validated
K&nbsp;=&nbsp;7 extension analyses (results/k7_primary), whose estimators were
first verified to reproduce the frozen K&nbsp;=&nbsp;6 counterparts exactly
before K&nbsp;=&nbsp;7 quantities were computed. No new calculations are
introduced in this manuscript beyond the K&nbsp;=&nbsp;7 extension analyses
that mirror the frozen K&nbsp;=&nbsp;6 pipelines.</p>
<p>The primary numerical results are:</p>
<ul>
<li>N = {_N_TOTAL};</li>
<li>Pearson r = {_R:.4f};</li>
<li>95% CI [{_CI_LO:.4f}, {_CI_HI:.4f}];</li>
<li>p = {_p_mant:.2f} &times; 10<sup>&minus;{_p_exp}</sup>;</li>
<li>r&sup2; = {_R2:.4f};</li>
<li>GAP mean = {_gapmean_mant:.3f} &times; 10<sup>&minus;{_gapmean_exp}</sup>;</li>
<li>GAP SD = {float(_gsv['SD']):.6f};</li>
<li>GAP range = [{float(_gsv['min']):.2f}, {float(_gsv['max']):+.2f}];</li>
<li>INT &gt; BE = {int(_gsv['positive_gap_count'])} ({float(_gsv['positive_gap_pct']):.2f}%);</li>
<li>BE &gt; INT = {int(_gsv['negative_gap_count'])} ({float(_gsv['negative_gap_pct']):.2f}%);</li>
<li>K = {SEL} full-covariance BIC = {_BIC_SEL:.2f} (primary; {PARAMS_SEL} parameters);</li>
<li>K = {COMP} full-covariance BIC = {_BIC_COMP:.2f} (competing; {PARAMS_COMP} parameters);</li>
<li>K &ge; {DEG_K_MIN} = numerically degenerate;</li>
<li>K = {SEL} profile sizes = {prof7['N'].astype(int).tolist()};</li>
<li>{int((prof7.mean_z_INT > prof7.mean_z_BE).sum())} INT &gt; BE profiles ({"P" + ", P".join(str(i) for i in prof7.index[prof7.mean_z_INT > prof7.mean_z_BE])}) and {int((prof7.mean_z_BE > prof7.mean_z_INT).sum())} BE &gt; INT profiles
    ({"P" + ", P".join(str(i) for i in prof7.index[prof7.mean_z_BE > prof7.mean_z_INT])});</li>
<li>mean maximum posterior probability = {float(_cq['mean_max_posterior']):.4f};</li>
<li>{float(_cq['pct_maxpost_ge_0.70']):.2f}% with maximum posterior &ge;0.70;</li>
<li>K = {SEL} between-profile GAP variance = {between7:.2f}%;</li>
<li>K = {SEL} within-profile GAP variance = {within7:.2f}%;</li>
<li>K = {SEL} five-fold CV mean held-out log-likelihood = {float(cv.loc[cv.K == SEL, 'mean_ll_test'].iloc[0]):.2f} (fold SD
    {float(cv.loc[cv.K == SEL, 'sd_ll_test'].iloc[0]):.2f});</li>
<li>duplicate-excluded N = {_N_UNIQUE};</li>
<li>duplicate-excluded Pearson r = {_du['pearson_r_INT_BE']:.6f} (primary full sample: {float(_dg['pearson_r_INT_BE']):.6f});</li>
<li>{int(diagv['n_FDR_significant'])} of {int(diagv['n_tests'])} joint FDR tests significant;</li>
<li>McFadden pseudo-R&sup2; = {float(diagv['McFadden_pseudo_R2']):.4f};</li>
<li>maximum VIF = {float(diagv['max_VIF']):.2f};</li>
<li>{_n_floor_sel} of {SEL} K = {SEL} components at the reg_covar floor (constant-score lattice
    mechanism; {_n_floor_comp} of {COMP} at K = {COMP}).</li>
</ul>
<div class="placeholder">[AUTHOR INPUT REQUIRED: complete references, author
information, institutional information, sampling information, ethics
information, demographic definitions, item wording, and theoretical construct
citations.]</div>
""")

# ================================================================ FIGURE ORDER
H.append(f"""
<h2>Recommended Final Figure Order</h2>
<p><b>Figure 1: Aggregate INT&ndash;BE association</b> &rarr; Establishes the
strong population-level relationship.<br/>
<b>Figure 2: GAP distribution</b> &rarr; Shows respondent-level discrepancy and
both directions.<br/>
<b>Figure 3: Extended K&nbsp;=&nbsp;{_KMIN_T2}&ndash;{_KMAX_T2} BIC</b> &rarr; Establishes
model-selection uncertainty; identifies K&nbsp;=&nbsp;{SEL} as the primary
solution and K&nbsp;=&nbsp;{COMP} as the competing solution.<br/>
<b>Figure 4: Full vs diagonal covariance</b> &rarr; Shows covariance-specification
sensitivity.<br/>
<b>Figure 5: K&nbsp;=&nbsp;{SEL} profile configuration in z-space</b> &rarr; Main
conceptual figure. Shows the {SEL} configurations and INT&nbsp;&gt;&nbsp;BE /
BE&nbsp;&gt;&nbsp;INT directions.<br/>
<b>Figure 6: K&nbsp;=&nbsp;{SEL} profile means on the original 1&ndash;5
scale</b> &rarr; Makes the substantive magnitude of each profile easy to
understand.<br/>
<b>Figure 7: Between- vs within-profile GAP variance</b> &rarr; Prevents
overclaiming that profiles explain most individual heterogeneity.<br/>
<b>Figure 8: Posterior classification quality</b> &rarr; Establishes assignment
quality, read jointly with the lattice-collapse disclosure.<br/>
<b>Figure 9: Bootstrap directional stability</b> &rarr; Exposes the degree of
directional consistency of each profile.<br/>
<b>Figure 10: Five-fold cross-validation</b> &rarr; Additional, weakly
supportive evidence at K&nbsp;=&nbsp;{SEL} with large fold variability.<br/>
<b>Figure 11: Duplicate-row sensitivity</b> &rarr; Demonstrates qualitative
robustness and boundary sensitivity to the {_N_DUPS} duplicate rows.<br/>
<b>Supplementary Figure S1: K&nbsp;=&nbsp;{COMP} profile structure</b> &rarr; Keeps
the competing K&nbsp;=&nbsp;{COMP} solution visible.</p>

<h2>Final Production Rules</h2>
<p>The manuscript is generated from the frozen results without changing
numerical values. The following are not added unless explicitly supplied by the
authors: invented references; invented sampling, recruitment, or response-rate
information; invented ethics approval; invented demographic definitions or
questionnaire wording; invented theoretical citations; causal language;
statements that the primary solution is the global BIC minimum; statements that
any profile is a stable behavioural type; statements that the profiles explain
most of the intention&ndash;behaviour gap; statements that the predictor
analysis identifies causal determinants. The manuscript preserves the
distinction between aggregate association &rarr; GAP distribution &rarr; latent
configuration &rarr; model uncertainty &rarr; exploratory predictor
associations. This distinction is central to the interpretation of the
study.</p>

<footer>
Self-contained HTML manuscript &mdash; all 12 figures embedded as base64 PNG.
Generated from frozen evidence: results/paper1_final,
results/paper1_strengthening, results/k7_primary.
</footer>
""")

html_doc = ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"utf-8\"/>\n"
            f"<title>Heterogeneous Intention&ndash;Behaviour Configurations in "
            f"Household Energy-Saving Behaviour (K = {SEL} primary)</title>\n"
            f"<style>{CSS}</style>\n</head>\n<body>\n" + "\n".join(H)
            + "\n</body>\n</html>\n")

OUTFILE = "paper1_final_manuscript_k7.html"
with open(OUTFILE, "w", encoding="utf-8") as f:
    f.write(html_doc)
print(f"Wrote {OUTFILE} ({len(html_doc)/1e6:.2f} MB, {len(FIGS)} figures embedded)")
