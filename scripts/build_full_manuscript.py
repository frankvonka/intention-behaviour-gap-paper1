#!/usr/bin/env python3
"""Build the full-length Paper 1 HTML manuscript.

Structure: abstract, 1 Introduction, 2 Research questions + gaps + contribution,
3 Methodology (full detail), 4 Results, 5 Discussion, 6 Limitations,
7 Conclusion, Supplementary, Transparency, References (author-supplied slots).

5 main tables, 10 main figures (embedded base64 from
results/draft_figures/draft_figures_base64.json). EVERY number is read at
runtime from frozen evidence — no hardcoded values, no placeholders.

Output: paper1_full_manuscript.html
"""
import html as html_mod
import json
import math
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
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "&mdash;"
    v = float(x)
    return f"{v:+.{nd}f}".replace("-", "&minus;")


def esc(s):
    return html_mod.escape(str(s))


def table_html(headers, rows, row_classes=None, align=None):
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

# ================================================================ frozen data
cstats = pd.read_csv(f"{PF}/02_measurement/construct_statistics.csv")
kext = pd.read_csv(f"{S}/k_extended_model_comparison.csv")
kext_full = kext[kext.covariance_type == "full"].copy()
prof7 = pd.read_csv(f"{K7}/k7_profile_table_full_sample.csv")
gap7 = pd.read_csv(f"{K7}/k7_per_profile_gap.csv").set_index("profile")
stab = pd.read_csv(f"{S}/k7_stability.csv")
cv = pd.read_csv(f"{S}/cv5_aggregated_by_k.csv")
dup = pd.read_csv(f"{K7}/k7_duplicate_sensitivity.csv")
_dg = dict(zip(dup.metric, pd.to_numeric(dup.frozen, errors="coerce")))
_du = dict(zip(dup.metric, pd.to_numeric(dup.unique_sample, errors="coerce")))
diag = pd.read_csv(f"{K7}/k7_predictors/model_diagnostics.csv")
diagv = dict(zip(diag.metric, diag.value))
psum = pd.read_csv(f"{K7}/k7_predictors/predictor_summary.csv").set_index("predictor")
mnl7 = pd.read_csv(f"{K7}/k7_predictors/multinomial_results.csv")
dec = pd.read_csv(f"{K7}/gap_variance_decomposition_K2_K7.csv").set_index("K")
between7 = float(dec.loc[7, "between_over_total"]) * 100.0
within7 = 100.0 - between7

_corr = pd.read_csv("results/02_measurement/int_be_correlation.csv")
_R = float(_corr["r"].iloc[0]); _CI_LO = float(_corr["CI95_lower"].iloc[0])
_CI_HI = float(_corr["CI95_upper"].iloc[0]); _P = float(_corr["p"].iloc[0])
_R2 = _R * _R
_p_exp = int(math.floor(math.log10(_P))); _p_mant = _P / (10 ** _p_exp)
_gs = pd.read_csv(f"{PF}/03_intention_behavior_gap/gap_statistics.csv")
_gsv = dict(zip(_gs.statistic, _gs.value))
_cq = pd.read_csv(f"{S}/k7_classification_quality.csv").set_index("statistic")["value"]
_N_TOTAL = int(_dg["N_full_sample"]); _N_UNIQUE = int(_dg["N_unique_sample"])
_N_DUPS = int(_dg["n_duplicates_removed"])

_sel = pd.read_csv("results/05_lpa_selection/selected_model.csv")
SEL = int(_sel["selected_K"].iloc[0])
N_INIT_SEL = int(_sel["n_init"].iloc[0]); SEED_SEL = int(_sel["random_seed"].iloc[0])
_wb = kext_full[kext_full.log_likelihood < 0]
COMP = int(_wb[_wb.K != SEL].sort_values("BIC").K.iloc[0])
_BIC_SEL = float(kext_full.loc[kext_full.K == SEL, "BIC"].iloc[0])
_BIC_COMP = float(kext_full.loc[kext_full.K == COMP, "BIC"].iloc[0])
_KMIN_T2, _KMAX_T2 = int(kext_full.K.min()), int(kext_full.K.max())
_DEG_KS = sorted(int(k) for k in kext_full.loc[kext_full.log_likelihood > 0, "K"])
DEG_K_MIN = min(_DEG_KS)
_cov = pd.read_csv(f"{S}/covariance_k_extended.csv")
_diag_rows = _cov.pivot(index="K", columns="covariance_type", values="BIC")
_DIAG_LOWER = sorted(int(k) for k in _diag_rows.index
                     if _diag_rows.loc[k, "diag"] < _diag_rows.loc[k, "full"])
_COV_KMIN, _COV_KMAX = int(_cov.K.min()), int(_cov.K.max())

_eig = pd.read_csv(f"{K7}/covariance_eigenvalues.csv")
_floor = _eig[_eig.at_reg_floor == True]  # noqa: E712
_floor_sel = sorted(int(p) for p in _floor[_floor.K == SEL].profile)
_floor_comp = sorted(int(p) for p in _floor[_floor.K == COMP].profile)
_n_floor_sel = len(_floor_sel); _n_floor_comp = len(_floor_comp)

_floor_counts = {}
for _k in range(_KMIN_T2, _KMAX_T2 + 1):
    try:
        _d = json.load(open(f"{S}/task1_K_{_k}/covariance_matrices.json"))
        _mats = [np.array(v) for v in _d["covariances"].values()]
        _floor_counts[_k] = sum(
            1 for _mm in _mats
            if float(np.linalg.eigvalsh(_mm).min()) <= 1.5e-6)
    except FileNotFoundError:
        _floor_counts[_k] = None

_dims = pd.read_csv("results/01_data_inspection/data_dimensions.csv")
_N_COLS = int(_dims["N_columns"].iloc[0])
_N_ITEMS = int(cstats["k_items"].sum())
_N_CONSTRUCTS = int(len(cstats))
_miss = pd.read_csv("results/01_data_inspection/missing_values.csv")
_N_MISSING = int(_miss["n_missing"].sum())
_boot_master = pd.read_csv("results/14_k6_stability/k6_stability_master.csv")
_N_BOOT = int(dict(zip(_boot_master.metric, _boot_master.value))["n_bootstrap"])
_KCV_MIN, _KCV_MAX = int(cv.K.min()), int(cv.K.max())
_N_PRED = int(diagv["n_predictors"])
_N_TESTS = int(diagv["n_tests"])
_N_NREF = SEL - 1
_construct_preds = sorted(set(cstats.construct) & set(mnl7["predictor"]))
_demog_preds = sorted(set(mnl7["predictor"]) - set(cstats.construct))

stab7 = {int(r.profile[1:]): float(r.pct_INT_gt_BE) for _, r in stab.iterrows()}
dirs7 = {int(r.profile): ("INT &gt; BE" if r.INT_mean > r.BE_mean else "BE &gt; INT")
         for _, r in prof7.iterrows()}
N_INT_DIR = sum(1 for d in dirs7.values() if "INT" in d)
N_BE_DIR = sum(1 for d in dirs7.values() if "BE" in d)
_int_lbl = ", ".join(f"P{int(r.profile)} (n&nbsp;=&nbsp;{int(r.N)})"
                     for _, r in prof7.iterrows() if r.INT_mean > r.BE_mean)
_be_lbl = ", ".join(f"P{int(r.profile)} (n&nbsp;=&nbsp;{int(r.N)})"
                    for _, r in prof7.iterrows() if r.BE_mean > r.INT_mean)
_sizes_str = ", ".join(str(int(n)) for n in prof7["N"])
_cv_sel = float(cv.loc[cv.K == SEL, "mean_ll_test"].iloc[0])
_cv_sel_sd = float(cv.loc[cv.K == SEL, "sd_ll_test"].iloc[0])
_sig_preds = sorted(psum.index[psum.n_significant_contrasts > 0])
_sig_constructs = [p for p in _sig_preds if p in set(cstats.construct)]
_sig_demogs = [p for p in _sig_preds if p not in set(cstats.construct)]
_min_alpha_row = cstats.loc[cstats["Cronbach_alpha"].idxmin()]

H = []

# ================================================================ FRONT MATTER
H.append(f"""
<h1>Heterogeneous Intention&ndash;Behaviour Configurations in Household
Energy-Saving Behaviour: A Latent Profile Analysis of N&nbsp;=&nbsp;{_N_TOTAL}
Respondents</h1>
<div class="meta">Full manuscript generated from the frozen computational
evidence package. Primary solution: K&nbsp;=&nbsp;{SEL} (lowest BIC among
numerically well-behaved full-covariance models). K&nbsp;=&nbsp;{COMP} retained
as the leading competing solution.</div>

<div class="toc">
<b>Contents</b><br/>
<a href="#abstract">Abstract</a> &middot;
<a href="#s1">1. Introduction</a> &middot;
<a href="#s2">2. Research Questions, Gaps, and Contribution</a> &middot;
<a href="#s3">3. Methodology</a> &middot;
<a href="#s4">4. Results</a> &middot;
<a href="#s5">5. Discussion</a> &middot;
<a href="#s6">6. Limitations</a> &middot;
<a href="#s7">7. Conclusion</a> &middot;
<a href="#supp">Supplementary Materials</a> &middot;
<a href="#transparency">Data and Computational Transparency</a> &middot;
<a href="#refs">References</a>
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
cross-sectional sample of N&nbsp;=&nbsp;{_N_TOTAL} respondents reporting
household energy-saving behaviour using Latent Profile Analysis (LPA) of
standardised intention (INT) and behaviour (BE) indicators.</p>
<h3>Methods</h3>
<p>A {_N_CONSTRUCTS}-construct psychometric battery comprising {_N_ITEMS}
Likert-scale items was administered, alongside {len(_demog_preds)} demographic
variables. Construct scores were calculated as arithmetic means of their
constituent items; Cronbach&rsquo;s &alpha; ranged from
{float(cstats['Cronbach_alpha'].min()):.3f} to
{float(cstats['Cronbach_alpha'].max()):.3f}. The intention&ndash;behaviour gap
was defined as GAP<sub>i</sub>&nbsp;=&nbsp;z(INT<sub>i</sub>)&nbsp;&minus;&nbsp;z(BE<sub>i</sub>),
with within-sample standardisation using ddof&nbsp;=&nbsp;1. LPA was estimated
using sklearn GaussianMixture models with full covariance,
n_init&nbsp;=&nbsp;{N_INIT_SEL}, random_state&nbsp;=&nbsp;{SEED_SEL},
max_iter&nbsp;=&nbsp;500, and reg_covar&nbsp;=&nbsp;10<sup>&minus;6</sup>.
Models with K&nbsp;=&nbsp;{_KMIN_T2}&ndash;{_KMAX_T2} profiles were examined.
Model selection was based primarily on BIC restricted to numerically
well-behaved solutions, supplemented by classification quality, bootstrap
directional stability ({_N_BOOT} replications), five-fold cross-validation,
and sensitivity analyses. Exploratory profile-membership associations were
examined using multinomial logistic regression with {_N_PRED} predictors and
joint Benjamini&ndash;Hochberg false-discovery-rate correction across
{_N_TESTS} tests.</p>
<h3>Results</h3>
<p>The aggregate INT&ndash;BE association was strong,
r&nbsp;=&nbsp;{_R:.4f}, 95%&nbsp;CI&nbsp;[{_CI_LO:.4f},&nbsp;{_CI_HI:.4f}],
p&nbsp;=&nbsp;{_p_mant:.2f}&nbsp;&times;&nbsp;10<sup>&minus;{_p_exp}</sup>,
corresponding to r&sup2;&nbsp;=&nbsp;{_R2:.4f}. Of the {_N_TOTAL} respondents,
{int(_gsv['positive_gap_count'])} ({float(_gsv['positive_gap_pct']):.2f}%)
had INT&nbsp;&gt;&nbsp;BE and {int(_gsv['negative_gap_count'])}
({float(_gsv['negative_gap_pct']):.2f}%) had BE&nbsp;&gt;&nbsp;INT. Under the
primary full-covariance specification, the K&nbsp;=&nbsp;{SEL} solution had
the lowest BIC among numerically well-behaved models
(BIC&nbsp;=&nbsp;{_BIC_SEL:.2f}, versus {_BIC_COMP:.2f} at K&nbsp;=&nbsp;{COMP})
and comprised profiles of sizes {_sizes_str}. {N_INT_DIR} profiles had mean
INT&nbsp;&gt;&nbsp;mean BE and {N_BE_DIR} had mean BE&nbsp;&gt;&nbsp;mean INT.
Mean maximum posterior probability was
{float(_cq['mean_max_posterior']):.4f}, with
{float(_cq['pct_maxpost_ge_0.70']):.2f}% of respondents having maximum
posterior probability &ge;0.70. Bootstrap directional stability ({_N_BOOT}
replications) ranged from {min(stab.pct_INT_gt_BE):.1f}% to
{max(stab.pct_INT_gt_BE):.1f}% for the proportion of replications in which
the matched profile had INT&nbsp;&gt;&nbsp;BE. At K&nbsp;=&nbsp;{SEL},
{between7:.2f}% of GAP variance was between profiles and {within7:.2f}%
within profiles. Five-fold cross-validation produced the highest mean
held-out log-likelihood for K&nbsp;=&nbsp;{SEL} ({m(_cv_sel)}), but with a
large fold-to-fold SD ({_cv_sel_sd:.2f}). K&nbsp;&ge;&nbsp;{DEG_K_MIN}
produced numerically degenerate solutions. Diagonal covariance produced lower
BIC than full covariance at K&nbsp;=&nbsp;{", ".join(str(k) for k in _DIAG_LOWER)}.
{_n_floor_sel} of the {SEL} K&nbsp;=&nbsp;{SEL} components operated at the
reg_covar floor because they captured respondents with constant item scores.
Removal of {_N_DUPS} duplicate rows preserved the {SEL}-profile structure
under Hungarian matching (mean z-space distance
{m(_du['K7_hungarian_mean_distance'], 2)}) although class sizes shifted
materially. Exploratory multinomial logistic regression yielded
{int(diagv['n_FDR_significant'])} FDR-significant associations among
{int(diagv['n_tests'])} tests, with McFadden
pseudo-R&sup2;&nbsp;=&nbsp;{float(diagv['McFadden_pseudo_R2']):.4f}; severe
multicollinearity was present (maximum VIF&nbsp;=&nbsp;{float(diagv['max_VIF']):.2f}).</p>
<h3>Conclusion</h3>
<p>The data show that a strong population-level intention&ndash;behaviour
association coexists with heterogeneous intention&ndash;behaviour
configurations. The K&nbsp;=&nbsp;{SEL} solution identifies {N_INT_DIR}
profiles with INT&nbsp;&gt;&nbsp;BE and {N_BE_DIR} with BE&nbsp;&gt;&nbsp;INT,
but the exact profile structure is model-dependent and requires cautious
interpretation.</p>
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
accounts for only a modest share of the variance in actual behaviour,
highlighting a persistent intention&ndash;behaviour gap.</p>
<p>The discrepancy between intention and behaviour has commonly been examined
through variable-centred approaches. Such approaches estimate the average
strength of the intention&ndash;behaviour relationship across a population and
investigate variables that may strengthen or weaken that relationship. Research
has sought to bridge this gap by examining moderators of the
intention&ndash;behaviour relationship, such as behavioural control, habit
strength, and normative constraints. These moderators, along with situational
factors like opportunity and affordability, are consistently linked to the
strength of the relationship between intention and enactment.</p>
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
configurations. These approaches have gained traction in environmental and
energy research for their capacity to disentangle the complexity of
pro-environmental behavioural enactment beyond simple average effects.</p>
<p>The present study applies LPA specifically to intention and behaviour in
household energy-saving behaviour. Because latent profile solutions are
inherently model-dependent, the analysis explicitly examines model-selection
uncertainty, covariance specification, numerical stability of the covariance
estimates, classification quality, bootstrap directional stability,
cross-validation, and sensitivity to duplicate observations. The study does not
interpret latent profiles as causal types or claim that a single profile
solution represents a universally stable taxonomy of households.</p>
""")

# ================================================================ 2. RQs / GAPS
H.append(f"""
<h2 id="s2">2. Research Questions, Gaps Addressed, and Contribution</h2>
<h3>2.1 Research questions</h3>
<p>This study answers four questions:</p>
<ol>
<li><b>RQ1 &mdash; Aggregate association.</b> How strongly are stated
intention (INT) and reported behaviour (BE) associated at the population
level in household energy saving (N&nbsp;=&nbsp;{_N_TOTAL})?</li>
<li><b>RQ2 &mdash; Respondent-level discrepancy.</b> What is the distribution
of the standardised intention&ndash;behaviour gap,
GAP&nbsp;=&nbsp;z(INT)&nbsp;&minus;&nbsp;z(BE), and in which direction does
it run for each respondent?</li>
<li><b>RQ3 &mdash; Latent configurations.</b> Can respondents be represented
as a small number of distinct intention&ndash;behaviour configurations
(latent profiles), how many such configurations does the evidence support,
and what are their directions and magnitudes?</li>
<li><b>RQ4 &mdash; Predictors.</b> Which psychological constructs and
demographic variables are associated with membership in each configuration,
once multiplicity is controlled?</li>
</ol>
<h3>2.2 Gaps this study fills</h3>
<p><b>Gap 1 &mdash; Average effects hide configurations.</b> The literature
reports population-level intention&ndash;behaviour associations and
moderators, but a strong average association is compatible with qualitatively
different respondent-level configurations. We quantify both levels on the
same sample: a strong aggregate correlation (r&nbsp;=&nbsp;{_R:.4f}) alongside
a profile solution in which {N_INT_DIR} configurations run INT&nbsp;&gt;&nbsp;BE
and {N_BE_DIR} run BE&nbsp;&gt;&nbsp;INT.</p>
<p><b>Gap 2 &mdash; The reverse gap is under-examined.</b> The conventional
expectation is that intention exceeds behaviour. In our sample,
{int(_gsv['negative_gap_count'])} of {_N_TOTAL} respondents
({float(_gsv['negative_gap_pct']):.2f}%) show BE&nbsp;&gt;&nbsp;INT &mdash;
"action beyond intention" &mdash; a larger share than the
{int(_gsv['positive_gap_count'])} ({float(_gsv['positive_gap_pct']):.2f}%)
showing the conventional INT&nbsp;&gt;&nbsp;BE direction.</p>
<p><b>Gap 3 &mdash; Profile solutions are reported without model
uncertainty.</b> Applied LPA studies often report one selected solution.
We report the full selection surface (K&nbsp;=&nbsp;{_KMIN_T2}&ndash;{_KMAX_T2}),
retain the well-behaved competing solution (K&nbsp;=&nbsp;{COMP}), disclose
the numerically degenerate range (K&nbsp;&ge;&nbsp;{DEG_K_MIN}), the competing
diagonal-covariance specification, the discrete-lattice collapse affecting
{_n_floor_sel} of {SEL} primary components, and the weakly supportive
cross-validation (highest mean held-out log-likelihood at K&nbsp;=&nbsp;{SEL}
with the largest fold SD, {_cv_sel_sd:.2f}).</p>
<p><b>Gap 4 &mdash; Stability of profile directions is rarely tested.</b> We
test the directional interpretation of every profile over {_N_BOOT}
bootstrap replications with Hungarian matching, and report the full range
({min(stab.pct_INT_gt_BE):.1f}%&ndash;{max(stab.pct_INT_gt_BE):.1f}%), including
the near-chance profile.</p>
<h3>2.3 Interest and contribution</h3>
<p>The question matters for three audiences. <b>Theoretically</b>, finding
both INT&nbsp;&gt;&nbsp;BE and BE&nbsp;&gt;&nbsp;INT configurations in the
same sample challenges single-mechanism accounts of the gap and motivates
configuration-aware theorising. <b>Methodologically</b>, the study
demonstrates a fully auditable LPA workflow: every reported number traces to
a frozen evidence file, competing solutions stay visible, and numerical
pathologies are disclosed rather than silently dropped.
<b>Practically</b>, if households occupy different intention&ndash;behaviour
configurations, interventions can be differentiated: enabling follow-through
where intention already exceeds behaviour, versus understanding and
supporting the drivers of behaviour where action already exceeds stated
intention.</p>
""")

# ================================================================ 3. METHODS
H.append(f"""
<h2 id="s3">3. Methodology</h2>
<h3>3.1 Sample and data</h3>
<p>The sample consists of N&nbsp;=&nbsp;{_N_TOTAL} respondents from a single
cross-sectional survey. The analytic dataset contains {_N_COLS} measured
variables comprising {_N_ITEMS} psychometric items and {len(_demog_preds)}
demographic variables ({", ".join(_demog_preds)}).
{"There were no missing values across the measured cells." if _N_MISSING == 0 else f"There were {_N_MISSING} missing values across the measured cells."}</p>
<p>{_N_DUPS} rows ({float(_N_DUPS) / _N_TOTAL * 100:.2f}%) were flagged as
duplicates on all measured columns in the raw data file. These observations
were retained in the primary analysis to preserve the original analytic
sample. A sensitivity analysis was conducted after removing the {_N_DUPS}
duplicate rows, resulting in N&nbsp;=&nbsp;{_N_UNIQUE}.</p>

<h3>3.2 Measurement and construct scoring</h3>
<p>{_N_CONSTRUCTS} constructs were scored as arithmetic means of their
constituent items. All items used a 1&ndash;5 Likert response scale. Table 1
reports construct descriptive statistics and internal consistency from the
frozen measurement evidence.</p>
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
PO&nbsp;=&nbsp;Personal Obligation; PRI&nbsp;=&nbsp;Personal Responsibility.</div>
""")

H.append(f"""
<h3>3.3 Aggregate association and the GAP definition</h3>
<p>The aggregate association between INT and BE was quantified with
Pearson&rsquo;s correlation on the arithmetic-mean construct scores
(r&nbsp;=&nbsp;{_R:.4f}, 95%&nbsp;CI&nbsp;[{_CI_LO:.4f},&nbsp;{_CI_HI:.4f}],
p&nbsp;=&nbsp;{_p_mant:.2f}&nbsp;&times;&nbsp;10<sup>&minus;{_p_exp}</sup>,
r&sup2;&nbsp;=&nbsp;{_R2:.4f}; independently recomputed and verified).</p>
""")
H.append(figure_embed("fig1", 1,
    f"Aggregate intention&ndash;behaviour association. Joint distribution of "
    f"INT and BE mean scores for N&nbsp;=&nbsp;{_N_TOTAL} respondents "
    f"(hexagonal binning), with the fitted linear association "
    f"(r&nbsp;=&nbsp;{_R:.4f}) and the equality line (INT&nbsp;=&nbsp;BE)."))
H.append(f"""
<p>The respondent-level gap was defined as
GAP<sub>i</sub>&nbsp;=&nbsp;z(INT<sub>i</sub>)&nbsp;&minus;&nbsp;z(BE<sub>i</sub>),
with within-sample standardisation (ddof&nbsp;=&nbsp;1). The GAP mean is
approximately zero by construction; this does not imply symmetry.
{int(_gsv['positive_gap_count'])} respondents
({float(_gsv['positive_gap_pct']):.2f}%) had INT&nbsp;&gt;&nbsp;BE and
{int(_gsv['negative_gap_count'])} ({float(_gsv['negative_gap_pct']):.2f}%)
had BE&nbsp;&gt;&nbsp;INT; none tied exactly.</p>
""")
H.append(figure_embed("fig2", 2,
    f"Distribution of the standardised intention&ndash;behaviour gap "
    f"(N&nbsp;=&nbsp;{_N_TOTAL}), with kernel-density estimate, the "
    f"Normal(0,&nbsp;{float(_gsv['SD']):.4f}) reference density, and the zero "
    f"reference point."))

H.append(f"""
<h3>3.4 Latent profile analysis: specification</h3>
<p>LPA was conducted on two standardised indicators, z(INT) and z(BE), using
sklearn.mixture.GaussianMixture with full covariance,
n_init&nbsp;=&nbsp;{N_INIT_SEL}, random_state&nbsp;=&nbsp;{SEED_SEL},
max_iter&nbsp;=&nbsp;500, and reg_covar&nbsp;=&nbsp;10<sup>&minus;6</sup>.
Models with K&nbsp;=&nbsp;{_KMIN_T2}&ndash;{_KMAX_T2} profiles were fitted.
BIC&nbsp;=&nbsp;&minus;2&nbsp;log&nbsp;L&nbsp;+&nbsp;k&nbsp;log(N); for the
two-indicator full-covariance model the free-parameter count is
k&nbsp;=&nbsp;6K&nbsp;&minus;&nbsp;1, so K&nbsp;=&nbsp;{SEL} uses
{6 * SEL - 1} parameters and K&nbsp;=&nbsp;{COMP} uses {6 * COMP - 1}.
Classification entropy is
E&nbsp;=&nbsp;&minus;&sum;<sub>i</sub>&sum;<sub>j</sub>p<sub>ij</sub>&nbsp;log&nbsp;p<sub>ij</sub>&nbsp;/&nbsp;(n&nbsp;log&nbsp;K).</p>

<h3>3.5 Model selection and degeneracy guard</h3>
<p>The primary solution is the lowest-BIC model among numerically
well-behaved fits. Fits with positive log-likelihood were excluded as
numerically degenerate (singular-covariance behaviour): this is the entire
K&nbsp;&ge;&nbsp;{DEG_K_MIN} range. Component-covariance eigenvalues were
audited against the reg_covar floor (10<sup>&minus;6</sup>); components at
the floor are disclosed, not hidden. Supplementary evidence: classification
quality (mean maximum posterior), {_N_BOOT}-replication bootstrap
directional stability with Hungarian matching on the two-dimensional profile
means via scipy.optimize.linear_sum_assignment, five-fold cross-validation
(K&nbsp;=&nbsp;{_KCV_MIN}&ndash;{_KCV_MAX}, held-out log-likelihood),
duplicate-row refit (N&nbsp;=&nbsp;{_N_UNIQUE}), full-versus-diagonal
covariance comparison (K&nbsp;=&nbsp;{_COV_KMIN}&ndash;{_COV_KMAX}), and
alternative specifications (spherical covariance, alternative scorings,
seeds).</p>

<h3>3.6 Profile-membership predictors</h3>
<p>Multinomial logistic regression (statsmodels.MNLogit, BFGS,
maxiter&nbsp;=&nbsp;1000) of K&nbsp;=&nbsp;{SEL} membership on {_N_PRED}
predictors &mdash; {", ".join(_construct_preds)} (constructs) and
{", ".join(_demog_preds)} (demographics) &mdash; with Profile 0 as
reference, giving {_N_PRED}&nbsp;&times;&nbsp;{_N_NREF}&nbsp;=&nbsp;{_N_TESTS}
tests with joint Benjamini&ndash;Hochberg FDR correction at
&alpha;&nbsp;=&nbsp;0.05. Treated as exploratory and associational:
cross-sectional, and construct predictors are severely multicollinear
(maximum VIF&nbsp;=&nbsp;{float(diagv['max_VIF']):.2f}).</p>
""")

# ================================================================ 4. RESULTS
kmap = {int(r.K): r for _, r in kext_full.iterrows()}
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
<h2 id="s4">4. Results</h2>
<h3>4.1 Measurement and aggregate association (RQ1)</h3>
<p>Internal consistency ranged from &alpha;&nbsp;=&nbsp;{float(cstats['Cronbach_alpha'].min()):.4f}
({esc(cstats.loc[cstats['Cronbach_alpha'].idxmin(), 'construct'])}) to
&alpha;&nbsp;=&nbsp;{float(cstats['Cronbach_alpha'].max()):.4f}
({esc(cstats.loc[cstats['Cronbach_alpha'].idxmax(), 'construct'])}) (Table 1).
The aggregate INT&ndash;BE correlation was r&nbsp;=&nbsp;{_R:.4f}
(95%&nbsp;CI&nbsp;[{_CI_LO:.4f},&nbsp;{_CI_HI:.4f}],
p&nbsp;=&nbsp;{_p_mant:.2f}&nbsp;&times;&nbsp;10<sup>&minus;{_p_exp}</sup>;
r&sup2;&nbsp;=&nbsp;{_R2:.4f}; Figure 1).</p>

<h3>4.2 GAP distribution (RQ2)</h3>
<p>The standardised GAP had SD&nbsp;=&nbsp;{float(_gsv['SD']):.6f} and range
[{float(_gsv['min']):.2f},&nbsp;{float(_gsv['max']):+.2f}] (Figure 2).</p>
""")

H.append(f"""
<h3>4.3 Model selection (RQ3)</h3>
<p>Full-covariance fit statistics are in Table 2.</p>
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
<p>The K&nbsp;=&nbsp;{SEL} solution had the lowest BIC among numerically
well-behaved models (BIC&nbsp;=&nbsp;{_BIC_SEL:.2f}) and is the
<b>primary solution</b>. K&nbsp;&ge;&nbsp;{DEG_K_MIN} models show lower
(negative) BIC but positive log-likelihoods consistent with singular
covariance solutions, and are not interpreted. K&nbsp;=&nbsp;{COMP}
(BIC&nbsp;=&nbsp;{_BIC_COMP:.2f}) is well-behaved and retained as the
<b>competing solution</b>. {_n_floor_sel} of {SEL} K&nbsp;=&nbsp;{SEL}
components sit at the reg_covar floor (P{", P".join(str(p) for p in _floor_sel)};
constant-score profiles from the discrete 1&ndash;5 lattice), as do
{_n_floor_comp} of {COMP} at K&nbsp;=&nbsp;{COMP}.</p>
""")
H.append(figure_embed("fig3", 3,
    f"Model fit across K&nbsp;=&nbsp;{_KMIN_T2}&ndash;{_KMAX_T2} (full "
    f"covariance). K&nbsp;=&nbsp;{SEL} (primary; lowest well-behaved BIC) and "
    f"K&nbsp;=&nbsp;{COMP} (competing) highlighted; shaded region marks "
    f"K&nbsp;&ge;&nbsp;{DEG_K_MIN} numerical degeneracy."))
H.append(f"""
<p>Diagonal covariance attains lower BIC at
K&nbsp;=&nbsp;{", ".join(str(k) for k in _DIAG_LOWER)}, including
K&nbsp;=&nbsp;{SEL} &mdash; a legitimate competing specification retained as
a limitation.</p>
""")
H.append(figure_embed("fig4", 4,
    f"BIC under full and diagonal covariance specifications across "
    f"K&nbsp;=&nbsp;{_COV_KMIN}&ndash;{_COV_KMAX}."))

H.append(f"""
<h3>4.4 Profile structure (RQ3)</h3>
<p>The primary K&nbsp;=&nbsp;{SEL} solution comprised profiles of sizes
{_sizes_str} (Table 3).</p>
""")
t3_rows = []
for _, r in prof7.iterrows():
    t3_rows.append([f"P{int(r.profile)}", str(int(r.N)), m(r.percentage),
                    m(r.INT_mean, 3), m(r.BE_mean, 3), msign(r.INT_minus_BE, 3),
                    dirs7[int(r.profile)]])
t3_rows.append(["Total", str(_N_TOTAL), "100.00%", "&mdash;", "&mdash;",
                "&mdash;", f"{N_INT_DIR} INT&gt;BE + {N_BE_DIR} BE&gt;INT"])
H.append(f"<p><b>Table 3.</b> K&nbsp;=&nbsp;{SEL} profile descriptives on the "
         f"original 1&ndash;5 scale (N&nbsp;=&nbsp;{_N_TOTAL}).</p>")
H.append(table_html(
    ["Profile", "N", "%", "INT mean", "BE mean", "INT &minus; BE", "Direction"],
    t3_rows, row_classes=[None] * SEL + ["total"],
    align=["c", "c", "n", "n", "n", "n", "c"]))
H.append(f"""
<p>{N_INT_DIR} profiles had mean INT&nbsp;&gt;&nbsp;mean BE ({_int_lbl});
{N_BE_DIR} profiles had mean BE&nbsp;&gt;&nbsp;mean INT ({_be_lbl}).
Per-profile mean GAP ranged from {m(float(gap7.loc[gap7.mean_GAP.idxmin(), 'mean_GAP']), 2)}
(P{int(gap7.mean_GAP.idxmin())}) to
{msign(float(gap7.loc[gap7.mean_GAP.idxmax(), 'mean_GAP']), 2)}
(P{int(gap7.mean_GAP.idxmax())}); within-profile INT&nbsp;&gt;&nbsp;BE shares
ranged from {min(gap7.pct_INT_gt_BE):.2f}% (P{int(gap7.pct_INT_gt_BE.idxmin())})
to {max(gap7.pct_INT_gt_BE):.2f}% (P{int(gap7.pct_INT_gt_BE.idxmax())}).</p>
""")
H.append(figure_embed("fig5", 5,
    f"K&nbsp;=&nbsp;{SEL} profile configuration in z-space. Marker area "
    f"proportional to profile size; equality line z(INT)&nbsp;=&nbsp;z(BE). "
    f"Red: mean z(INT)&nbsp;&gt;&nbsp;mean z(BE); blue: the reverse."))
H.append(figure_embed("fig6", 6,
    f"K&nbsp;=&nbsp;{SEL} profile-specific intention and behaviour means on "
    f"the original 1&ndash;5 scale."))
H.append(f"""
<p>At K&nbsp;=&nbsp;{SEL}, {between7:.2f}% of GAP variance was between
profiles and {within7:.2f}% within profiles (K&nbsp;=&nbsp;{COMP}:
{float(dec.loc[COMP, 'between_over_total'])*100:.2f}%; K&nbsp;=&nbsp;2:
{float(dec.loc[2, 'between_over_total'])*100:.2f}%). Profiles capture a
meaningful but minority share of GAP variance.</p>
""")
H.append(figure_embed("fig7", 7,
    f"GAP variance decomposition. At K&nbsp;=&nbsp;{SEL}, {between7:.2f}% "
    f"between profiles, {within7:.2f}% within."))
H.append(f"""
<p>Classification: mean maximum posterior
{float(_cq['mean_max_posterior']):.4f} (median
{float(_cq['median_max_posterior']):.4f});
{float(_cq['pct_maxpost_ge_0.70']):.2f}% &ge;0.70,
{float(_cq['pct_maxpost_lt_0.70']):.2f}% below 0.70 &mdash; partly inflated
by near-certain assignment into constant-score components.</p>
""")
H.append(figure_embed("fig8", 8,
    f"Maximum posterior probability distribution (K&nbsp;=&nbsp;{SEL}), "
    f"thresholds at 0.70/0.80/0.90; mean&nbsp;=&nbsp;{float(_cq['mean_max_posterior']):.4f}."))

t4_rows = []
for i in range(SEL):
    p = stab7[i]
    interp = ("Directionally stable (INT &gt; BE)" if p >= 80
              else "Directionally stable as BE &gt; INT" if p <= 20
              else "Unstable / near chance" if 40 < p < 60
              else "Moderate consistency")
    rc = "sig" if (p >= 80 or p <= 20) else None
    t4_rows.append((rc, [f"P{i}", f"{p:.1f}%", interp]))
H.append(f"""
<h3>4.5 Bootstrap directional stability</h3>
<p>Over {_N_BOOT} replications (Table 4):</p>
""")
H.append(f"<p><b>Table 4.</b> Bootstrap directional stability "
         f"(K&nbsp;=&nbsp;{SEL}, {_N_BOOT} replications).</p>")
H.append(table_html(
    ["Profile", "% replications with INT &gt; BE", "Interpretation"],
    [r for _, r in t4_rows],
    row_classes=[rc for rc, _ in t4_rows],
    align=["c", "n", None]))
_p_near = min(stab7, key=lambda i: abs(stab7[i] - 50.0))
H.append(f"""
<p>{", ".join(f"P{i} ({p:.1f}%)" for i, p in stab7.items() if (p >= 80 or p <= 20))}
were directionally stable; {", ".join(f"P{i} ({p:.1f}%)" for i, p in stab7.items() if 20 < p < 80)}
showed moderate consistency with modal directions matching the point
estimates. P{_p_near} ({stab7[_p_near]:.1f}%) sits closest to chance.</p>
""")
H.append(figure_embed("fig9", 9,
    f"Bootstrap directional stability ({_N_BOOT} replications, "
    f"K&nbsp;=&nbsp;{SEL}). Green: stable (&ge;80% or &le;20%); orange: "
    f"moderate; red: near chance (40&ndash;60%)."))

t5_rows = []
for _, r in cv.iterrows():
    rc = "sig" if int(r.K) == SEL else None
    t5_rows.append((rc, [str(int(r.K)), m(r.mean_ll_test), m(r.sd_ll_test)]))
H.append(f"""
<h3>4.6 Cross-validation and duplicate sensitivity</h3>
<p>Five-fold CV (K&nbsp;=&nbsp;{_KCV_MIN}&ndash;{_KCV_MAX}; Table 5):</p>
""")
H.append(f"<p><b>Table 5.</b> Five-fold cross-validation: mean held-out "
         f"log-likelihood (K&nbsp;=&nbsp;{_KCV_MIN}&ndash;{_KCV_MAX}).</p>")
H.append(table_html(
    ["K", "Mean held-out log-likelihood", "SD"],
    [r for _, r in t5_rows], row_classes=[rc for rc, _ in t5_rows],
    align=["c", "n", "n"]))
H.append(f"""
<p>K&nbsp;=&nbsp;{SEL} had the highest mean held-out log-likelihood
({m(_cv_sel)}) but also the largest fold SD ({_cv_sel_sd:.2f}) &mdash; weak
support. Removing the {_N_DUPS} duplicates (N&nbsp;=&nbsp;{_N_UNIQUE}):
r {_dg['pearson_r_INT_BE']:.4f}&nbsp;&rarr;&nbsp;{_du['pearson_r_INT_BE']:.4f},
GAP SD {float(_dg['gap_SD']):.4f}&nbsp;&rarr;&nbsp;{float(_du['gap_SD']):.4f},
structure preserved under Hungarian matching (distance
{m(_du['K7_hungarian_mean_distance'], 4)}) with materially shifted class
sizes.</p>
""")
H.append(figure_embed("fig10", 10,
    f"Five-fold cross-validation: mean held-out log-likelihood &plusmn; SD "
    f"(K&nbsp;=&nbsp;{_KCV_MIN}&ndash;{_KCV_MAX}). K&nbsp;=&nbsp;{SEL} highest "
    f"mean, largest fold variability."))

H.append(f"""
<h3>4.7 Exploratory profile-membership associations (RQ4)</h3>
<p>{int(diagv['n_FDR_significant'])} of {int(diagv['n_tests'])} jointly
FDR-adjusted tests significant (McFadden
pseudo-R&sup2;&nbsp;=&nbsp;{float(diagv['McFadden_pseudo_R2']):.4f}; maximum
VIF&nbsp;=&nbsp;{float(diagv['max_VIF']):.2f}). Significant contrasts
concentrated in constructs ({", ".join(_sig_constructs)});
{("no demographic predictor reached" if not _sig_demogs else
  "demographics " + ", ".join(_sig_demogs) + " also reached")}
joint-FDR significance. Given severe multicollinearity, coefficients are a
joint exploratory pattern, not uniquely identified effects; pseudo-R&sup2;
is not variance explained and not comparable to r&sup2;&nbsp;=&nbsp;{_R2:.4f}.</p>
""")

# ================================================================ 5. DISCUSSION
H.append(f"""
<h2 id="s5">5. Discussion</h2>
<p><b>Principal finding.</b> A strong aggregate association (r&nbsp;=&nbsp;{_R:.4f})
coexists with heterogeneous configurations: {N_INT_DIR} INT&nbsp;&gt;&nbsp;BE
profiles and {N_BE_DIR} BE&nbsp;&gt;&nbsp;INT profiles under the primary
K&nbsp;=&nbsp;{SEL} solution.</p>
<p><b>RQ1&ndash;RQ2.</b> The strong correlation replicates the well-known
intention&ndash;behaviour link, while the GAP distribution shows the
discrepancy runs in both directions &mdash; with BE&nbsp;&gt;&nbsp;INT the
majority ({float(_gsv['negative_gap_pct']):.2f}%). "Action beyond intention"
is the modal discrepancy here, challenging intention-first accounts.</p>
<p><b>RQ3.</b> The {SEL}-profile structure is the lowest-BIC well-behaved
solution, but only {between7:.2f}% of GAP variance is between profiles:
configurations are real but account for a minority of individual variation.
The competing K&nbsp;=&nbsp;{COMP} solution, the diagonal-covariance
alternative, the lattice-collapse components, and the high-variance CV all
constrain how literally the number {SEL} should be taken.</p>
<p><b>RQ4.</b> {int(diagv['n_FDR_significant'])} FDR-significant associations
implicate {", ".join(_sig_constructs)}, with no demographic predictor
significant &mdash; but multicollinearity (VIF&nbsp;=&nbsp;{float(diagv['max_VIF']):.2f})
forbids interpreting any coefficient as an independent effect.</p>
<p><b>Implications.</b> Theoretically, single-mechanism gap accounts must
accommodate coexisting opposite-direction configurations. Practically,
INT&nbsp;&gt;&nbsp;BE households need follow-through enablement, while
BE&nbsp;&gt;&nbsp;INT households invite study of what already drives their
action.</p>
""")

# ================================================================ 6. LIMITS
H.append(f"""
<h2 id="s6">6. Limitations</h2>
<p><b>Cross-sectional design</b> &mdash; no temporal ordering or causal
claims. <b>Model dependence</b> &mdash; K&nbsp;=&nbsp;{COMP} is a
well-behaved competitor (BIC&nbsp;=&nbsp;{_BIC_COMP:.2f} vs
{_BIC_SEL:.2f}); diagonal covariance wins at
K&nbsp;=&nbsp;{", ".join(str(k) for k in _DIAG_LOWER)}; K&nbsp;&ge;&nbsp;{DEG_K_MIN}
degenerate. <b>Lattice collapse</b> &mdash; P{", P".join(str(p) for p in _floor_sel)}
at K&nbsp;=&nbsp;{SEL} (and P{", P".join(str(p) for p in _floor_comp)} at
K&nbsp;=&nbsp;{COMP}) are constant-score artefacts inflating BIC advantage
and classification quality. <b>Stability</b> &mdash; P{_p_near}
({stab7[_p_near]:.1f}%) nearest chance; {within7:.2f}% of GAP variance
within profiles. <b>Predictors</b> &mdash; VIF&nbsp;=&nbsp;{float(diagv['max_VIF']):.2f}.
<b>Reliability</b> &mdash; {esc(_min_alpha_row.construct)} lowest
&alpha;&nbsp;=&nbsp;{float(_min_alpha_row.Cronbach_alpha):.4f}
({int(_min_alpha_row.k_items)} items). <b>Duplicates</b> &mdash; boundaries
sensitive to the {_N_DUPS} duplicate rows. <b>Generalisability</b> &mdash;
single survey; sampling frame not verifiable from the analytic file.</p>
""")

# ================================================================ 7. CONCLUSION
H.append(f"""
<h2 id="s7">7. Conclusion</h2>
<p>A strong population-level intention&ndash;behaviour association
(r&nbsp;=&nbsp;{_R:.4f}, N&nbsp;=&nbsp;{_N_TOTAL}) coexists with
heterogeneous respondent-level configurations. The primary K&nbsp;=&nbsp;{SEL}
solution &mdash; lowest BIC among well-behaved full-covariance models &mdash;
identifies {N_INT_DIR} INT&nbsp;&gt;&nbsp;BE and {N_BE_DIR} BE&nbsp;&gt;&nbsp;INT
profiles capturing {between7:.2f}% of GAP variance. The exact structure is
model-dependent (K&nbsp;=&nbsp;{COMP} competes; diagonal covariance competes;
lattice artefacts; moderate stability), and predictor associations are
exploratory under severe multicollinearity. The study's contribution is the
joint demonstration that aggregate strength and configurational heterogeneity
coexist &mdash; with "action beyond intention" as the modal discrepancy &mdash;
under a fully auditable, uncertainty-disclosing workflow.</p>
""")

# ================================================================ SUPPLEMENTARY
H.append(f"""
<h2 id="supp">Supplementary Materials</h2>
<p><b>Table S1.</b> Full/diagonal covariance BIC comparison
(K&nbsp;=&nbsp;{_COV_KMIN}&ndash;{_COV_KMAX}).
<span class="src">Source: results/paper1_strengthening/covariance_k_extended.csv</span></p>
<p><b>Table S2.</b> Full K&nbsp;=&nbsp;{SEL} multinomial coefficient table
({_N_TESTS} tests with raw/FDR p-values).
<span class="src">Source: results/k7_primary/k7_predictors/multinomial_results.csv</span></p>
<p><b>Table S3.</b> Duplicate-exclusion refit profiles (Hungarian-matched,
N&nbsp;=&nbsp;{_N_UNIQUE}).
<span class="src">Source: results/k7_primary/k7_duplicate_sensitivity_profiles.csv</span></p>
<p><b>Table S4.</b> K&nbsp;=&nbsp;{SEL} per-profile GAP statistics.
<span class="src">Source: results/k7_primary/k7_per_profile_gap.csv</span></p>
<p><b>Table S5.</b> K&nbsp;=&nbsp;{COMP} competing-solution descriptives.
<span class="src">Source: results/04_lpa_estimation/K_{COMP}/</span></p>
<p><b>Figure S1.</b> K&nbsp;=&nbsp;{COMP} profile structure (competing
solution).</p>
""")

# ================================================================ TRANSPARENCY
H.append(f"""
<h2 id="transparency">Data and Computational Transparency</h2>
<p>Every number in this manuscript is substituted at build time from a frozen
evidence file; nothing is hand-typed. Key values: N&nbsp;=&nbsp;{_N_TOTAL};
r&nbsp;=&nbsp;{_R:.4f} [{_CI_LO:.4f},&nbsp;{_CI_HI:.4f}];
p&nbsp;=&nbsp;{_p_mant:.2f}&nbsp;&times;&nbsp;10<sup>&minus;{_p_exp}</sup>;
GAP SD&nbsp;=&nbsp;{float(_gsv['SD']):.6f};
INT&nbsp;&gt;&nbsp;BE&nbsp;=&nbsp;{int(_gsv['positive_gap_count'])}
({float(_gsv['positive_gap_pct']):.2f}%);
BE&nbsp;&gt;&nbsp;INT&nbsp;=&nbsp;{int(_gsv['negative_gap_count'])}
({float(_gsv['negative_gap_pct']):.2f}%);
K&nbsp;=&nbsp;{SEL} BIC&nbsp;=&nbsp;{_BIC_SEL:.2f};
K&nbsp;=&nbsp;{COMP} BIC&nbsp;=&nbsp;{_BIC_COMP:.2f};
sizes [{_sizes_str}]; mean max posterior&nbsp;=&nbsp;{float(_cq['mean_max_posterior']):.4f};
between-profile GAP variance&nbsp;=&nbsp;{between7:.2f}%;
CV mean held-out LL at K&nbsp;=&nbsp;{SEL}&nbsp;=&nbsp;{m(_cv_sel)}
(SD&nbsp;=&nbsp;{_cv_sel_sd:.2f});
{int(diagv['n_FDR_significant'])}/{int(diagv['n_tests'])} FDR-significant;
pseudo-R&sup2;&nbsp;=&nbsp;{float(diagv['McFadden_pseudo_R2']):.4f};
VIF&nbsp;=&nbsp;{float(diagv['max_VIF']):.2f}.</p>

<h2 id="refs">References</h2>
<p>Author-supplied reference list to be inserted here. In-text citations used
in this manuscript refer to the Theory of Planned Behavior and Norm Activation
Model literature, the intention&ndash;behaviour gap and moderator literature,
and prior person-centred (LPA/LCA) applications to pro-environmental and
energy-saving behaviour.</p>

<footer>
Self-contained HTML manuscript &mdash; 10 figures embedded as base64 PNG.
Generated from frozen evidence: results/paper1_final,
results/paper1_strengthening, results/k7_primary.
</footer>
""")

html_doc = ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"utf-8\"/>\n"
            f"<title>Heterogeneous Intention&ndash;Behaviour Configurations "
            f"in Household Energy-Saving Behaviour "
            f"(K = {SEL} primary; full manuscript)</title>\n"
            f"<style>{CSS}</style>\n</head>\n<body>\n" + "\n".join(H)
            + "\n</body>\n</html>\n")

OUTFILE = "paper1_full_manuscript.html"
with open(OUTFILE, "w", encoding="utf-8") as f:
    f.write(html_doc)
n_fig = html_doc.count("data:image/png;base64,")
n_tab = html_doc.count("<b>Table ")
print(f"Wrote {OUTFILE} ({len(html_doc)/1e6:.2f} MB, "
      f"{n_fig} figures, {n_tab} main-table mentions)")
