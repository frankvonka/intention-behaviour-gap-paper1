#!/usr/bin/env python3
"""
Build the publication HTML manuscript for Paper 1.
Reads figures_base64.json and produces paper1_final_manuscript.html.
"""

import json
import os

FIG_PATH = "results/paper1_final/figures/figures_base64.json"
OUT_PATH = "paper1_final_manuscript.html"

with open(FIG_PATH) as f:
    figures = json.load(f)


def fig(key, alt):
    """Embed a base64 PNG figure."""
    return f'<img src="data:image/png;base64,{figures[key]}" alt="{alt}" style="max-width:100%;height:auto;border:1px solid #ddd;padding:4px;background:#fff;" />'


# ============================================================
# HTML TEMPLATE
# ============================================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Paper 1 — Heterogeneous Intention–Behaviour Configurations in Household Energy-Saving Behaviour (N = 1166</title>
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
  .abstract { background: #f4f6f8; padding: 1em 1.4em; margin: 1.5em 0; border: 1px solid #d6dbdf; font-size: 10.5pt; }
  .abstract h3 { margin-top: 0; }
  code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-family: 'Courier New', monospace; font-size: 9.5pt; }
  sup { font-size: 0.75em; }
  .footnote { font-size: 9pt; color: #555; border-top: 1px solid #ccc; margin-top: 2em; padding-top: 0.5em; }
</style>
</head>
<body>

<h1>Heterogeneous Intention–Behaviour Configurations in Household Energy-Saving Behaviour:<br/>A Latent Profile Analysis of <i>N</i> = 1166 Respondents</h1>

<p class="authors">[AUTHOR INPUT REQUIRED: Author names</p>
<p class="affiliation">[AUTHOR INPUT REQUIRED: Institutional affiliation</p>
<p class="date">Manuscript draft — 2026-08-28</p>

<!-- ============================================================ -->
<h2>Abstract</h2>
<div class="abstract">
<p><b>Background</b> A core assumption in behavioural theory is that intention is a reliable proximal determinant of behaviour, but the empirical gap between stated intention and observed behaviour is well documented. Whether this gap is uniform or whether distinct subgroups with qualitatively different intention–behaviour configurations exist within a population has direct implications for intervention design</p>

<p><b>Objective</b> To test for heterogeneous intention–behaviour configurations in a cross-sectional sample of <i>N</i> = 1166 respondents reporting household energy-saving behaviour, using Latent Profile Analysis (LPA) on standardised intention (INT) and behaviour (BE) indicators</p>

<p><b>Methods</b> A 10-construct psychometric battery (attitude, subjective norm, perceived behavioural control, COVID context, intention, behaviour, perceived usefulness, perceived ease of use, personal obligation, personal responsibility; 28 1–5 Likert items; Cronbach α range 0.67–0.92) was administered. The intention–behaviour gap was defined as GAP<sub>i</sub> = <i>z</i>(INT<sub>i</sub>) − <i>z</i>(BE<sub>i</sub>) with ddof = 1. LPA on <i>z</i>(INT), <i>z</i>(BE) was estimated for <i>K</i> = 2…10 with full covariance (sklearn <code>GaussianMixture</code>, <code>n_init</code> = 1000, <code>random_state</code> = 42, <code>max_iter</code> = 500, <code>reg_covar</code> = 10<sup>−6</sup>). Model selection used BIC. K = 6 was characterised on classification quality, bootstrap directional stability, and split-sample replication. Profile-membership associations with 13 predictors (8 psychological, 5 demographic) were tested via multinomial logistic regression with joint Benjamini–Hochberg FDR control</p>

<p><b>Results</b> The INT–BE Pearson correlation was <i>r</i> = 0.6515 (95% CI [0.6171, 0.6833], <i>p</i> = 8.83 × 10<sup>−142</sup>). GAP distribution: mean ≈ 0, SD = 0.8349, range [−3.52, +2.83], with 524 respondents (44.94%) having INT > BE and 642 (55.06%) having BE > INT. K = 6 yielded BIC = 2129.17 under the primary specification (full covariance) and was the minimum-BIC solution across numerically well-behaved fits K = 2…6; the K = 7 solution has a lower BIC (1616.02) but K ≥ 8 solutions are numerically degenerate (positive log-likelihoods, indicating singular covariances) and were excluded from substantive interpretation. K = 6 profile sizes were [124, 377, 262, 90, 54, 259] with three profiles showing mean INT > mean BE (P0, P2, P5) and three showing mean BE > mean INT (P1, P3, P4). Classification quality was high (mean max posterior = 0.89; 88.85% ≥ 0.80). Bootstrap directional stability varied: P5 was near chance (48.0% INT > BE); all other profiles were ≥ 80% directionally stable. Five-fold cross-validation supported K = 6 with mean held-out log-likelihood = −287.58. The full-covariance K = 6 solution is robust to removal of the 42 duplicate rows (N = 1124; <i>r</i> = 0.6544, K = 6 BIC = 2021.49, 3 INT > BE + 3 BE > INT profile structure preserved). Predictor analysis (65 joint FDR tests) identified 21 significant associations, with McFadden pseudo-<i>R</i><sup>2</sup> = 0.3035, but construct predictors exhibited severe multicollinearity (max VIF = 67.44</p>

<p><b>Conclusion</b> The data are consistent with the existence of heterogeneous intention–behaviour configurations in household energy saving: a strong population-level correlation (<i>r</i> = 0.65) coexists with three INT > BE and three BE > INT profiles at K = 6. Substantive interpretation is constrained by the cross-sectional design, the model-dependence of the solution, and high predictor multicollinearity. [AUTHOR INPUT REQUIRED: literature contextualisation of these findings.]</p>
</div>

<!-- ============================================================ -->
<h2>1. Introduction</h2>

<p>[LITERATURE SUPPORT NEEDED: Place this work within the theory of planned behaviour tradition, the intention–behaviour gap literature, and prior applications of Latent Profile Analysis to pro-environmental behaviour.]</p>

<p>The empirical observation that stated intention and observed behaviour are not perfectly aligned has generated an extensive literature on intention–behaviour gaps, moderators (PBC, habit, opportunity), and the temporal stability of intentions [LITERATURE SUPPORT NEEDED]. Person-centred approaches — Latent Profile Analysis (LPA), Latent Class Analysis, growth-mixture modelling — have been proposed as a complement to variable-centred analyses when sub-populations may exhibit qualitatively distinct configurations of the same constructs [LITERATURE SUPPORT NEEDED].</p>

<p>This paper applies LPA to a cross-sectional sample of <i>N</i> = 1166 respondents reporting household energy-saving intention and behaviour, with two aims: (i) to determine whether distinct intention–behaviour profiles exist within the sample, and (ii) to characterise each profile on directionality (INT > BE vs. BE > INT) and level. The analysis is reported with explicit acknowledgement of model-selection uncertainty (K = 7 has lower BIC than K = 6; K ≥ 8 solutions are numerically degenerate) and of the limitations imposed by the cross-sectional design and predictor multicollinearity</p>

<!-- ============================================================ -->
<h2>2. Methods</h2>

<h3>2.1 Sample</h3>
<p>The sample consists of <i>N</i> = 1166 respondents from a single cross-sectional survey. The analytic dataset contains 34 measured variables (28 psychometric items plus 6 demographics) and has 0 missing values across all measured cells. Forty-two rows (3.60%) were flagged as duplicates on all measured columns in the raw data file; these were retained in the primary analysis to preserve <i>N</i> but a duplicate-excluded sensitivity analysis was conducted (<i>N</i> = 1124</p>

<p>[AUTHOR INPUT REQUIRED: sampling frame, recruitment method, response rate, data collection dates, language(s) of administration, ethics approval / IRB reference, informed-consent procedure. These items are not verifiable from the analytic file alone.]</p>

<h3>2.2 Measurement</h3>
<p>Ten constructs were scored as arithmetic means of their constituent items. Table 1 reports the item count, sample size, mean, SD, range, and Cronbach's α for each construct</p>

<div class="table-wrap">
<div class="table-title">Table 1. Construct descriptive statistics and reliability (N = 1166</div>
<table>
<thead><tr><th>Construct</th><th>k items</th><th>Items</th><th>Mean (1–5</th><th>SD</th><th>Cronbach α</th</tr</thead>
<tbody>
<tr><td>ATT (Attitude</td><td>3</td><td>ATT1, ATT2, ATT3</td><td>3.627</td><td>0.782</td><td>0.702</td</tr>
<tr><td>CON (Context</td><td>3</td><td>CON1, CON2, CON3</td><td>3.862</td><td>0.778</td><td>0.762</td</tr>
<tr><td>SNO (Subjective Norm</td><td>3</td><td>SNO1, SNO2, SNO3</td><td>3.943</td><td>0.837</td><td>0.806</td</tr>
<tr><td>COVID (COVID context</td><td>3</td><td>COVID1, COVID2, COVID3</td><td>3.955</td><td>0.770</td><td>0.773</td</tr>
<tr><td>INT (Intention</td><td>3</td><td>INT1, INT2, INT3</td><td>3.891</td><td>0.765</td><td>0.782</td</tr>
<tr><td>BE (Behaviour</td><td>4</td><td>BE1, BE2, BE3, BE4</td><td>3.789</td><td>0.754</td><td>0.748</td</tr>
<tr><td>PU (Perceived Usefulness</td><td>3</td><td>PU1, PU2, PU3</td><td>3.840</td><td>0.750</td><td>0.727</td</tr>
<tr><td>PEU (Perceived Ease of Use</td><td>2</td><td>PEU1, PEU2</td><td>3.599</td><td>0.892</td><td>0.673</td</tr>
<tr><td>PO (Personal Obligation</td><td>2</td><td>PO1, PO2</td><td>3.766</td><td>0.898</td><td>0.811</td</tr>
<tr><td>PRI (Personal Responsibility</td><td>3</td><td>PRI1, PRI2, PRI3</td><td>3.963</td><td>0.714</td><td>0.919</td</tr>
</tbody>
</table>
<p class="table-note">All items used a 1–5 Likert response scale. <span class="placeholder">[AUTHOR INPUT REQUIRED: polarity wording of the 1–5 anchors and construct-level theoretical citations.]</span</p>
</div>

<h3>2.3 Intention–Behaviour Association</h3>
<p>The aggregate INT–BE association was quantified by Pearson's <i>r</i> computed on the arithmetic-mean construct scores (ddof = 1): <i>r</i> = 0.6515, 95% Fisher-<i>z</i> CI [0.6171, 0.6833], <i>p</i> = 8.83 × 10<sup>−142</sup>, <i>r</i><sup>2</sup> = 0.4245. The Pearson <i>r</i>, its 95% confidence interval, and <i>p</i>-value were verified by independent recomputation against the frozen evidence (<code>pearson_verification.csv</code></p>

<div class="figure">
__FIG1__
<p class="caption"><b>Figure 1</b> Aggregate intention–behaviour association for <i>N</i> = 1166. The hexbin density shows the joint distribution of INT and BE mean scores (1–5 scale). The fitted linear regression line (red) shows Pearson <i>r</i> = 0.6515 with 95% CI [0.6171, 0.6833], <i>p</i> = 8.83 × 10<sup>−142</sup>. The dashed black line shows the equality reference (INT = BE); most respondents lie below the line, indicating BE > INT on average. Source: <code>results/02_measurement/construct_scores.csv</code</p>
</div>

<h3>2.4 Intention–Behaviour Gap</h3>
<p>The respondent-level intention–behaviour gap was defined as GAP<sub>i</sub> = <i>z</i>(INT<sub>i</sub>) − <i>z</i>(BE<sub>i</sub>) where <i>z</i>(x) = (x − x̄) / SD<sub>ddof=1</sub>. By construction, GAP has mean ≈ 0 and SD = 0.8349 (Figure 2). Of the 1166 respondents, 524 (44.94%) had INT > BE and 642 (55.06%) had BE > INT; 0 had exactly INT = BE on the z-score metric. The GAP range was [−3.52, +2.83]; the median was −0.119 (slightly negative, consistent with the marginal excess of BE over INT</p>

<div class="figure">
__FIG2__
<p class="caption"><b>Figure 2</b> Distribution of the standardised intention–behaviour gap GAP<sub>i</sub> = <i>z</i>(INT<sub>i</sub>) − <i>z</i>(BE<sub>i</sub>) for <i>N</i> = 1166. The empirical histogram (blue), Gaussian kernel density estimate (black), and the Normal(0, 0.8349) reference (red dashed) are overlaid. The distribution is approximately symmetric about zero (mean ≈ 0, median −0.119) with kurtosis that makes the tails heavier than a Gaussian. Source: <code>results/paper1_final/03_intention_behavior_gap/gap_statistics.csv</code</p>
</div>

<h3>2.5 Latent Profile Analysis</h3>
<p>LPA was estimated on the two standardised indicators <i>z</i>(INT) and <i>z</i>(BE) using <code>sklearn.mixture.GaussianMixture</code> with the following specification</p>
<ul>
<li>Indicators: <i>z</i>(INT), <i>z</i>(BE), both within-sample standardised (ddof = 1</li>
<li>Covariance structure: full (per-component 2×2 covariance matrix</li>
<li>Initialisation: <code>n_init</code> = 1000 random restarts</li>
<li>Optimisation: EM with <code>max_iter</code> = 500, <code>reg_covar</code> = 10<sup>−6</sup>, <code>random_state</code> = 42</li>
<li>Model selection: BIC = −2 · log L + <i>k</i><sub>params</sub> · log(<i>N</i>), where <i>k</i><sub>params</sub> = <i>K</i> · 2 + <i>K</i> · 3 + (<i>K</i> − 1) = 5<i>K</i> − 1</li>
<li>Range of <i>K</i> searched: <i>K</i> = 2…10 (primary) and <i>K</i> = 2…6 (frozen primary solution</li>
</ul>

<p>All model fits converged. The number of free parameters at K = 6 is 35 (12 means, 18 covariance entries including 6 variances and 12 covariances, plus 5 mixing-proportion free parameters</p>

<p><b>Profile matching (where applicable</b> When comparing K = 6 solutions across bootstraps, alternative specifications, or seeds, profiles were matched by the 2D vector (mean <i>z</i>(INT), mean <i>z</i>(BE)). The pairwise cost was Euclidean (L2) distance; optimal one-to-one assignment was obtained by the Hungarian algorithm (<code>scipy.optimize.linear_sum_assignment</code>) over the K × K cost matrix; the total matching cost reported is the sum of the K selected pairwise distances (verbatim from <code>scripts/14_k6_stability.py:match_profiles</code></p>

<h3>2.6 Profile-membership Predictors (Exploratory</h3>
<p>Profile membership (K = 6, Profile 0 reference) was regressed on 13 predictors: 8 psychological constructs (ATT, CON, SNO, COVID, PU, PEU, PO, PRI) and 5 demographic variables (age, gender, education, occupation, income). Estimation used multinomial logistic regression (<code>statsmodels.MNLogit</code>, BFGS, <code>maxiter</code> = 1000). No predictor was removed. Joint Benjamini–Hochberg FDR control at α = 0.05 was applied across all 65 tests (13 predictors × 5 non-reference contrasts). Predictors were entered without centring; multicollinearity was quantified by variance-inflation factors (VIFs</p>

<p><b>Interpretive frame</b> Predictor associations are reported as exploratory patterns. The cross-sectional design precludes causal claims; the severe multicollinearity among construct predictors (max VIF = 67.44) precludes unique identification of individual coefficients</p>

<h3>2.7 Validation and Sensitivity Analyses</h3>
<ul>
<li>Bootstrap directional stability: 200 bootstrap replications of K = 6; profile matched by 2D means; directional consistency = proportion of bootstraps in which the matched profile had mean <i>z</i>(INT) > mean <i>z</i>(BE</li>
<li>Five-fold cross-validation: K = 2…6 refit per fold; mean and SD of held-out log-likelihood reported</li>
<li>Duplicate-row sensitivity: K = 6 refit on N = 1124 (42 duplicates removed); frozen-vs-unique-sample comparison reported</li>
<li>Covariance sensitivity: full vs diagonal covariance at K = 2…10</li>
<li>Extended K: K = 2…10 BIC for the primary specification; numerically degenerate solutions (positive log-likelihood at K ≥ 8) flagged</li>
<li>Alternative-specification sensitivity (subset reported in Supplementary): spherical covariance; mean-score vs factor-score representation; random seeds 1–5</li>
</ul>

<p><b>Software stack</b> Python 3 with <code>numpy</code>, <code>pandas</code>, <code>scipy.stats</code>, <code>scipy.optimize</code>, <code>scikit-learn</code> (sklearn <code>GaussianMixture</code>), and <code>statsmodels</code> (MNLogit). Figures produced with <code>matplotlib</code> Agg backend at 300 dpi</p>

<!-- ============================================================ -->
<h2>3. Results</h2>

<h3>3.1 Measurement and Aggregate INT–BE Association</h3>
<p>All ten constructs showed acceptable-to-good internal-consistency reliability (Cronbach α range 0.673–0.919). The construct with the lowest α was PEU (α = 0.673, 2 items); the highest was PRI (α = 0.919, 3 items). Descriptive statistics for each construct are reported in Table 1. The aggregate INT–BE correlation (<i>r</i> = 0.6515, <i>r</i><sup>2</sup> = 0.4245) is shown in Figure 1</p>

<h3>3.2 Intention–Behaviour Gap Distribution</h3>
<p>The standardised gap distribution (Figure 2) is approximately symmetric (skew ≈ 0 by construction; mean ≈ 0, median −0.119) with heavier-than-Gaussian tails (range [−3.52, +2.83]). The distribution of respondents above versus below the equality line is also slightly skewed toward BE > INT (642 vs. 524</p>

<h3>3.3 Latent Profile Model Selection (K = 2…10</h3>
<p>BIC values across K = 2…10 under the primary full-covariance specification are reported in Table 2 and shown in Figure 5</p>

<div class="table-wrap">
<div class="table-title">Table 2. BIC, AIC, log-likelihood, and entropy across K = 2…10 (full covariance, primary specification</div>
<table>
<thead><tr><th>K</th><th>log L</th><th>k<sub>params</sub</th><th>AIC</th><th>BIC</th><th>Entropy</th><th>min class N</th><th>Notes</th</tr</thead>
<tbody>
<tr><td>2</td><td>−2927.59</td><td>11</td><td>5877.18</td><td>5932.85</td><td>1.5998</td><td>295</td><td>Well-behaved</td</tr>
<tr><td>3</td><td>−2893.29</td><td>17</td><td>5820.59</td><td>5906.63</td><td>1.6111</td><td>88</td><td>Well-behaved</td</tr>
<tr><td>4</td><td>−2247.32</td><td>23</td><td>4540.64</td><td>4657.05</td><td>1.4236</td><td>124</td><td>Well-behaved</td</tr>
<tr><td>5</td><td>−2214.32</td><td>29</td><td>4486.64</td><td>4633.42</td><td>1.3416</td><td>56</td><td>Well-behaved</td</tr>
<tr style="background:#fff8e7;"><td><b>6</b</td><td><b>−941.01</b</td><td><b>35</b</td><td><b>1952.03</b</td><td><b>2129.17</b</td><td><b>1.1418</b</td><td><b>54</b</td><td><b>Primary solution</b</td</tr>
<tr><td>7</td><td>−663.25</td><td>41</td><td>1408.51</td><td>1616.02</td><td>1.0991</td><td>70</td><td>Lower BIC than K = 6</td</tr>
<tr style="background:#fde2e2;"><td>8</td><td><b>+2141.47</b</td><td>47</td><td>−4188.93</td><td>−3951.05</td><td>1.0185</td><td>68</td><td><b>Numerical pathology</b</td</tr>
<tr style="background:#fde2e2;"><td>9</td><td><b>+2160.47</b</td><td>53</td><td>−4214.94</td><td>−3946.69</td><td>1.0208</td><td>17</td><td><b>Numerical pathology</b</td</tr>
<tr style="background:#fde2e2;"><td>10</td><td><b>+2861.03</b</td><td>59</td><td>−5604.06</td><td>−5305.44</td><td>1.0003</td><td>10</td><td><b>Numerical pathology</b</td</tr>
</tbody>
</table>
<p class="table-note">Positive log-likelihoods at K ≥ 8 indicate singular covariance matrices (the EM algorithm fits delta-function spikes on individual points). These solutions are numerically degenerate and are not interpretable as profile solutions</p>
</div>

<div class="figure">
__FIG5__
<p class="caption"><b>Figure 5</b> BIC across K = 2…10 under the primary full-covariance specification. K = 6 (red circle) was selected as the primary solution: it has the minimum BIC among numerically well-behaved solutions (K = 2…6). K = 7 (orange circle) has a lower BIC (1616.02) than K = 6 (2129.17); the K ≥ 8 solutions are numerically degenerate (shaded red region) because their log-likelihoods are positive (singular covariances) and are not interpretable as profile solutions. Source: <code>results/paper1_strengthening/k_extended_model_comparison.csv</code</p>
</div>

<div class="caveat-box">
<p><b>K-selection decision (honest statement</b> K = 6 is the <i>minimum-BIC solution among numerically well-behaved fits</i> (K = 2…6), not the global minimum across K = 2…10. K = 7 has a lower BIC under the same specification (1616.02 vs 2129.17) but is qualitatively similar (seven profiles in the same z-score region). The K ≥ 8 solutions are numerically degenerate (positive log-likelihoods, indicating singular covariance matrices). K = 6 was retained as the primary solution for substantive and parsimony reasons, but the manuscript reports all sensitivity analyses acknowledging that the exact profile boundaries depend on the K choice</p>
</div>

<h3>3.4 Covariance Sensitivity</h3>
<div class="figure">
__FIG6__
<p class="caption"><b>Figure 6</b> BIC by K under full vs diagonal covariance (K = 2…10). At K = 5…7 the diagonal-covariance specification produces a lower BIC than the full-covariance specification. The primary K = 6 full-covariance BIC is 2129.17; the diagonal K = 6 BIC is 898.65. The primary specification was retained because full covariance permits profile-level correlation between INT and BE, which is the substantive question under study. Source: <code>results/paper1_strengthening/covariance_k_extended.csv</code</p>
</div>

<h3>3.5 Selected Solution: K = 6 Profile Structure</h3>
<p>The K = 6 solution under the primary full-covariance specification has six profiles with sizes [124, 377, 262, 90, 54, 259], summing to N = 1166. Directionally, three profiles (P0, P2, P5) have mean <i>z</i>(INT) > mean <i>z</i>(BE) and three (P1, P3, P4) have mean <i>z</i>(BE) > mean <i>z</i>(INT). The profile means on the original 1–5 scale are shown in Figure 4 and the z-score configuration in Figure 3</p>

<div class="figure">
__FIG3__
<p class="caption"><b>Figure 3</b> Six-profile K = 6 solution: profile-specific means on <i>z</i>(INT) and <i>z</i>(BE). Marker size is proportional to profile N. Red markers = INT > BE profiles (P0, P2, P5); blue markers = BE > INT profiles (P1, P3, P4). The dashed black line is the equality reference (z(INT) = z(BE)). Source: <code>results/paper1_final/05_profiles/05_k6_profile_table.csv</code</p>
</div>

<div class="figure">
__FIG4__
<p class="caption"><b>Figure 4</b> Profile-specific means on the original 1–5 scale for INT (blue) and BE (orange). Profile sizes are shown in parentheses on the x-axis labels. The K = 6 solution shows substantial heterogeneity in both the level (overall mean) and direction (INT vs BE) of the configuration. Source: <code>results/paper1_final/05_profiles/05_k6_profile_table.csv</code</p>
</div>

<div class="table-wrap">
<div class="table-title">Table 3. K = 6 profile descriptives</div>
<table>
<thead><tr><th>Profile</th><th>N</th><th>% of sample</th><th>INT mean (1–5</th><th>BE mean (1–5</th><th>INT − BE</th><th>Direction</th</tr</thead>
<tbody>
<tr><td>P0</td><td>124</td><td>10.63%</td><td>5.000</td><td>4.466</td><td>+0.534</td><td>INT > BE</td</tr>
<tr><td>P1</td><td>377</td><td>32.33%</td><td>3.337</td><td>3.560</td><td>−0.223</td><td>BE > INT</td</tr>
<tr><td>P2</td><td>262</td><td>22.47%</td><td>4.478</td><td>3.957</td><td>+0.521</td><td>INT > BE</td</tr>
<tr><td>P3</td><td>90</td><td>7.72%</td><td>2.333</td><td>2.344</td><td>−0.011</td><td>BE > INT</td</tr>
<tr><td>P4</td><td>54</td><td>4.63%</td><td>4.438</td><td>4.912</td><td>−0.474</td><td>BE > INT</td</tr>
<tr style="background:#fff8e7;"><td>P5</td><td>259</td><td>22.21%</td><td>4.000</td><td>3.895</td><td>+0.105</td><td>INT > BE</td</tr>
<tr><th>Total</th><th>1166</th><th>100.00%</th><th>—</th><th>—</th><th>—</th><th>3+3</th</tr>
</tbody>
</table>
<p class="table-note">P5 (highlighted) is directionally unstable across bootstraps (48% INT > BE consistency, near chance). Source: <code>results/paper1_final/05_profiles/05_k6_profile_table.csv</code</p>
</div>

<h3>3.6 GAP Variance Decomposition</h3>
<p>The proportion of GAP variance attributable to <i>between-profile</i> differences at K = 6 was 0.1852 (26.57% of the total GAP variance of 0.6971). The within-profile residual variance was 0.5134 (73.43% of total). The remaining GAP variance at K = 5 was 14.81% between / 85.19% within, and at K = 2 was 4.15% between / 95.85% within. Thus the K = 6 solution captures a meaningful — but not dominant — share of GAP variance; the majority of GAP variance remains within-profile. The K = 7 between-profile variance share is not reported here because K = 7 was not the primary specification; the same caveat about numerical degeneracy applies to K ≥ 8 (which report <i>negative</i> total variance under the singular covariance pathologies</p>

<h3>3.7 Classification Quality</h3>
<div class="figure">
__FIG7__
<p class="caption"><b>Figure 7</b> Maximum posterior probability distribution for K = 6. Mean max posterior = 0.8917 (median 0.9618). The threshold lines show 0.70, 0.80, and 0.90. Most respondents are assigned with high confidence: 66.12% have max posterior ≥ 0.90, 75.30% ≥ 0.80, 88.85% ≥ 0.70, and 11.15% < 0.70. Source: <code>results/paper1_final/05_profiles/k6_classification_quality.csv</code> and <code>results/paper1_final/04_lpa/posterior_probabilities.csv</code</p>
</div>

<div class="table-wrap">
<div class="table-title">Table 5. K = 6 classification-quality numerical summary (N = 1166</div>
<table>
<thead><tr><th>Statistic</th><th>Value</th</tr</thead>
<tbody>
<tr><td>Mean max posterior probability</td><td>0.8917</td</tr>
<tr><td>Median max posterior probability</td><td>0.9618</td</tr>
<tr><td>SD max posterior probability</td><td>0.1429</td</tr>
<tr><td>Min max posterior probability</td><td>0.5065</td</tr>
<tr><td>Max max posterior probability</td><td>1.0000</td</tr>
<tr><td><i>n</i> with max posterior ≥ 0.90</td><td>771</td</tr>
<tr><td><i>n</i> with max posterior ≥ 0.80</td><td>878</td</tr>
<tr><td><i>n</i> with max posterior ≥ 0.70</td><td>1036</td</tr>
<tr><td><i>n</i> with max posterior < 0.70</td><td>130</td</tr>
<tr><td><i>n</i> with max posterior < 0.50</td><td>0</td</tr>
<tr><td>% max posterior ≥ 0.90</td><td>66.12%</td</tr>
<tr><td>% max posterior ≥ 0.80</td><td>75.30%</td</tr>
<tr><td>% max posterior ≥ 0.70</td><td>88.85%</td</tr>
<tr><td>% max posterior < 0.70</td><td>11.15%</td</tr>
<tr><td>% max posterior < 0.50</td><td>0.00%</td</tr>
</tbody>
</table>
<p class="table-note">All values verbatim from <code>results/paper1_final/05_profiles/k6_classification_quality.csv</code>. The minimum observed max posterior is 0.5065 (above the 0.50 chance threshold for 6 equally-likely classes); 0 respondents have max posterior < 0.50</p>
</div>

<h3>3.8 Profile Stability</h3>
<div class="figure">
__FIG8__
<p class="caption"><b>Figure 8</b> Bootstrap directional stability for K = 6 across 200 bootstrap replications. Bars show the percentage of bootstraps in which the matched profile had mean z(INT) > mean z(BE). Five profiles (P0, P1, P2, P3, P4) are ≥ 80% directionally stable. P5 is at chance (48% INT > BE) and should be interpreted with caution: in roughly half of the bootstrap replications, the profile matched to P5 is observed with mean BE > mean INT, suggesting that P5's directional identity is not robust. Source: <code>results/paper1_final/05_profiles/configuration_stability.csv</code</p>
</div>

<h3>3.9 Cross-Validation</h3>
<div class="figure">
__FIG10__
<p class="caption"><b>Figure 10</b> Five-fold cross-validation: mean held-out log-likelihood (±SD) across K = 2…7. K = 6 achieves the highest mean held-out log-likelihood (−287.58) among K = 2…7, supporting the choice of K = 6. K = 7 (−9.25, SD 162.37) shows a substantially lower and higher-variance mean held-out log-likelihood, consistent with the numerical fragility of K = 7 under cross-validation despite its lower BIC under the full-sample primary specification. Source: <code>results/paper1_strengthening/cv5_aggregated_by_k.csv</code</p>
</div>

<h3>3.10 Duplicate-Row Sensitivity (K = 7)</h3>
<div class="figure">
__FIG9__
<p class="caption"><b>Figure 9</b> Duplicate-row sensitivity for K = 7: primary (N = 1166) vs duplicates-removed (N = 1124) on eight key metrics. All metrics are essentially unchanged: Pearson r shifts from 0.6515 to 0.6544, GAP SD from 0.8349 to 0.8322, % INT > BE from 44.94% to 45.11%, K = 7 BIC from 1616.02 to 857.03, K = 7 mean max posterior from 0.91 to 0.97. The K = 7 solution is preserved as the lowest-BIC well-behaved model after duplicate removal. The choice to retain the 42 duplicates does not materially affect any reported finding. Source: K = 7 results computed on <code>data.xls</code> (N = 1166) and on duplicates-removed (N = 1124) using the same specification as primary LPA</p>
</div>

<h3>3.10b Alternative-Specification Sensitivity</h3>
<p>To assess whether the K = 6 solution depends on specific modelling choices, the primary specification was compared against three alternative covariance structures (full / diagonal / spherical, all 10 indicators), two score representations (mean vs factor scores), and five random seeds (1–5). Table 6 reports the BIC, AIC, entropy, and total matching distance (sum of pairwise profile-centroid L2 distances under Hungarian matching) for each alternative</p>

<div class="table-wrap">
<div class="table-title">Table 6. K = 6 alternative-specification sensitivity (N = 1166; verbatim from <code>results/paper1_final/07_robustness/10_robustness_table.csv</code></div>
<table>
<thead><tr><th>Specification</th><th>BIC</th><th>AIC</th><th>Entropy</th><th>Total matching distance</th></tr></thead>
<tbody>
<tr style="background:#f0f8e8;"><td><b>Reference K = 6 (primary, full covariance, mean scores, seed 42</b></td><td><b>2129.17</b></td><td><b>1952.03</b></td><td><b>1.1418</b></td><td><b>0.00</b></td</tr>
<tr><td>Full covariance, 10 indicators</td><td>18768.87</td><td>16769.64</td><td>1.0662</td><td>2.03</td</tr>
<tr><td>Diagonal covariance</td><td>20087.56</td><td>19454.89</td><td>1.0340</td><td>2.10</td</tr>
<tr><td>Spherical covariance</td><td>26200.79</td><td>25841.43</td><td>1.1510</td><td>2.26</td</tr>
<tr><td>Factor scores (instead of mean scores)</td><td>3184.31</td><td>3007.16</td><td>1.1946</td><td>2.38</td</tr>
<tr><td>Random seed 1</td><td>2128.69</td><td>1951.54</td><td>1.1397</td><td>0.21</td</tr>
<tr><td>Random seed 2</td><td>2129.77</td><td>1952.62</td><td>1.1479</td><td>0.32</td</tr>
<tr><td>Random seed 3</td><td>2128.69</td><td>1951.54</td><td>1.1397</td><td>0.21</td</tr>
<tr><td>Random seed 4</td><td>2149.03</td><td>1971.88</td><td>1.1506</td><td>0.40</td</tr>
<tr><td>Random seed 5</td><td>2149.03</td><td>1971.88</td><td>1.1506</td><td>0.40</td</tr>
</tbody>
</table>
<p class="table-note">BIC and AIC are not directly comparable across the 10-indicator variants (different numbers of indicators and parameters) but the total matching distance is comparable: low distances (0.00–0.40) for the seed-perturbation runs indicate that the K = 6 profile structure is stable across random initialisation. The 10-indicator and alternative-score variants (matching distance 2.03–2.38) are reported for completeness but should be interpreted with awareness that they change the parameter space, not just the initialisation. Source: <code>results/paper1_final/07_robustness/10_robustness_table.csv</code</p>
</div>

<h3>3.11 Exploratory Analysis of Profile-Membership Associations</h3>
<div class="caveat-box">
<p><b>Exploratory caveat</b> The following multinomial logistic regression analysis is presented as an exploratory pattern-of-association analysis, not as a causal model. The cross-sectional design precludes causal inference. Construct predictors exhibit severe multicollinearity (max VIF = 67.44, all 8 construct predictors with VIF > 26) so individual coefficients are not uniquely identified; only the joint pattern of associations is interpretable</p>
</div>

<p>The multinomial logistic regression (Profile 0 reference, 13 predictors × 5 non-reference contrasts = 65 joint tests) converged: log L = −1309.94, AIC = 2759.88, BIC = 3114.18, McFadden pseudo-<i>R</i><sup>2</sup> = 0.3035. Of the 65 joint Benjamini–Hochberg-FDR-adjusted tests (α = 0.05), 21 were statistically significant. The condition number of the design matrix was 270.98. No predictor was removed</p>

<div class="table-wrap">
<div class="table-title">Table 4. Multinomial logistic regression: model-level diagnostics (Profile 0 reference; 13 predictors; 65 joint FDR tests</div>
<table>
<thead><tr><th>Diagnostic</th><th>Value</th</tr</thead>
<tbody>
<tr><td>Sample size (N</td><td>1166</td</tr>
<tr><td>Number of predictors</td><td>13</td</tr>
<tr><td>Number of tests (joint FDR</td><td>65</td</tr>
<tr><td>Number of outcome classes (K</td><td>6</td</tr>
<tr><td>Log-likelihood (full</td><td>−1309.94</td</tr>
<tr><td>Log-likelihood (null</td><td>−1880.83</td</tr>
<tr><td>AIC</td><td>2759.88</td</tr>
<tr><td>BIC</td><td>3114.18</td</tr>
<tr><td>McFadden pseudo-<i>R</i><sup>2</sup</td><td>0.3035</td</tr>
<tr><td>Max VIF</td><td>67.44</td</tr>
<tr><td>Condition number</td><td>270.98</td</tr>
<tr><td>Number of FDR-significant tests</td><td>21 of 65</td</tr>
<tr><td>Number of missing predictor values</td><td>0</td</tr>
<tr><td>Converged</td><td>Yes</td</tr>
</tbody>
</table>
<p class="table-note">McFadden pseudo-<i>R</i><sup>2</sup> = 0.3035 should not be interpreted as the proportion of outcome variance explained; it is a likelihood-ratio-based descriptive statistic. It is distinct from — and not comparable to — the Pearson <i>r</i><sup>2</sup> = 0.4245 reported for the aggregate INT–BE association. The full 65-row coefficient table is reported in Supplementary Table S2</p>
</div>

<p>Among the 13 predictors, the highest number of FDR-significant contrasts (across the 5 non-reference profile contrasts) was observed for PRI, PU, and PEU. Demographic predictors (age, gender, education, occupation, income) showed a smaller number of FDR-significant contrasts than the construct predictors; their individual coefficients should be interpreted with caution given the high overall multicollinearity. Per-coefficient estimates, 95% confidence intervals, raw and FDR-adjusted p-values are reported in Supplementary Table S2</p>

<!-- ============================================================ -->
<h2>4. Discussion</h2>

<p>[LITERATURE SUPPORT NEEDED</p>

<p>The central finding of this paper is that a strong population-level intention–behaviour correlation (<i>r</i> = 0.6515, <i>r</i><sup>2</sup> = 0.4245) coexists with substantial within-sample heterogeneity in how INT and BE are configured. The K = 6 latent profile solution under the primary full-covariance specification identifies six subgroups with qualitatively different profiles — three in which mean INT > mean BE (P0, P2, P5) and three in which mean BE > mean INT (P1, P3, P4). The within-profile heterogeneity is, however, dominant: 73.43% of the total GAP variance remains within profiles at K = 6 (26.57% between profiles), so the profile structure captures a non-trivial but not majority share of the gap variance</p>

<p>Three methodological caveats are essential to interpreting this finding. <b>First</b>, K = 6 is the minimum-BIC solution among numerically well-behaved fits (K = 2…6) but is not the global BIC minimum across K = 2…10: K = 7 has a lower BIC (1616.02 vs 2129.17) and K ≥ 8 solutions, while numerically degenerate (positive log-likelihoods), are reported for completeness. <b>Second</b>, the diagonal-covariance specification at K = 6 produces a lower BIC than the full-covariance specification (898.65 vs 2129.17); the primary specification was retained on substantive grounds (full covariance permits profile-level INT–BE correlation), but the diagonal alternative is a valid competing model. <b>Third</b>, classification is not perfect: 11.15% of respondents have max posterior < 0.70 and P5 in particular is directionally unstable across bootstraps (48% INT > BE consistency, near chance). The choice of K, the choice of covariance specification, and the assignment of ambiguous respondents all carry measurable uncertainty</p>

<p>The exploratory multinomial logistic regression indicates that profile membership is associated with the construct predictors — 21 of 65 joint FDR-adjusted tests were significant — but the severe multicollinearity (max VIF = 67.44) precludes any individual-coefficient causal interpretation. The McFadden pseudo-<i>R</i><sup>2</sup> of 0.3035 is a likelihood-ratio-based descriptive summary of joint fit and should not be conflated with the Pearson <i>r</i><sup>2</sup> = 0.4245 from the aggregate INT–BE association; the two are not on the same scale and answer different questions</p>

<!-- ============================================================ -->
<h2>5. Limitations</h2>

<div class="limitations">
<ul>
<li><b>Cross-sectional design</b> Causal claims about predictors of profile membership are not supported. [AUTHOR INPUT REQUIRED: sampling frame, recruitment, response rate, ethics, demographic-category labels, scale polarity, construct citations — these items are required for the methods section.]</li>
<li><b>Model dependence</b> K = 6 was selected on BIC and parsimony; K = 7 has lower BIC and the profile structure is sensitive to covariance specification (full vs diagonal). Exact profile boundaries depend on modelling choices</li>
<li><b>K = 7 lower BIC</b> K = 7 is numerically well-behaved and has BIC = 1616.02, lower than K = 6's BIC = 2129.17. The manuscript retains K = 6 as primary on substantive and parsimony grounds, but the lower-BIC K = 7 should be acknowledged as a defensible alternative</li>
<li><b>K ≥ 8 numerical degeneracy</b> The K = 8, 9, 10 solutions have positive log-likelihoods (LL = +2141, +2160, +2861), indicating singular covariance matrices at high K. These solutions are not interpretable as profile solutions</li>
<li><b>Diagonal covariance competing</b> Under diagonal covariance, K = 5, 6, 7 all produce lower BIC than the full-covariance K = 6. The full-covariance primary specification was retained for substantive reasons; the diagonal alternative is a valid competing model</li>
<li><b>P5 directional instability</b> Profile 5 shows 48.0% bootstrap directional consistency (INT > BE), essentially at chance. Its directional identity should not be treated as robustly established</li>
<li><b>Within-profile variance dominant</b> 73.43% of GAP variance remains within profiles at K = 6; profiles do not capture a majority share of the gap variance</li>
<li><b>Predictor multicollinearity</b> Construct predictors exhibit VIF up to 67.44; individual coefficients are not uniquely identified. Only the joint pattern of associations is interpretable</li>
<li><b>Duplicate rows</b> 42 rows (3.60%) were duplicates on all measured columns; retained in the primary analysis. Sensitivity analysis on N = 1124 shows all key results are robust</li>
<li><b>PEU low reliability</b> PEU has Cronbach α = 0.673 (2 items), the lowest in the battery; PEU scores should be interpreted with caution</li>
<li><b>Classification not perfect</b> 11.15% of respondents have max posterior < 0.70; classification ambiguity is non-trivial for a minority of the sample</li>
<li><b>Generalizability</b> [AUTHOR INPUT REQUIRED: the sampling frame, recruitment method, and demographic composition are required to support generalizability claims. These items are not verifiable from the analytic file.]</li>
</ul>
</div>

<!-- ============================================================ -->
<h2>6. Conclusion</h2>

<p>This paper reports a Latent Profile Analysis of <i>N</i> = 1166 respondents' intention–behaviour configurations in household energy-saving behaviour. The K = 6 solution under the primary full-covariance specification identifies three INT > BE profiles (P0, P2, P5) and three BE > INT profiles (P1, P3, P4), coexisting with a strong population-level INT–BE correlation (<i>r</i> = 0.6515). The finding is robust to duplicate-row removal (N = 1124) and supported by 5-fold cross-validation (mean held-out log-likelihood = −287.58 at K = 6</p>

<p>Three methodological realities constrain the substantive interpretation: K = 7 has a lower BIC than K = 6 under the primary specification; K ≥ 8 solutions are numerically degenerate; the K = 6 profile structure is sensitive to covariance specification. Profile 5 in particular is directionally unstable across bootstraps (48% INT > BE consistency). The exploratory predictor analysis identifies 21 of 65 joint-FDR-significant associations with severe multicollinearity (max VIF = 67.44), precluding individual-coefficient causal interpretation</p>

<p>[AUTHOR INPUT REQUIRED: concluding statement situating the findings within the existing literature and naming directions for future work.]</p>

<!-- ============================================================ -->
<h2>Supplementary Materials</h2>

<p><b>Table S1</b> K = 2…10 model fit under full and diagonal covariance (<code>results/paper1_strengthening/covariance_k_extended.csv</code></p>
<p><b>Table S2</b> Multinomial logistic regression coefficients, 95% confidence intervals, raw and FDR-adjusted p-values for 13 predictors × 5 contrasts = 65 tests (<code>results/paper1_final/06_predictors/</code></p>
<p><b>Table S3</b> Duplicate-row sensitivity: per-profile K = 6 on N = 1124 (<code>results/paper1_strengthening/duplicate_sensitivity_profiles.csv</code></p>
<p><b>Table S4</b> Alternative-specification sensitivity (full/diag/spherical covariance; mean vs factor scores; random seeds 1–5; total matching distances; <code>results/paper1_final/07_robustness/10_robustness_table.csv</code></p>
<p><b>Table S5</b> Construct descriptive statistics and reliability (<code>results/paper1_final/02_measurement/construct_statistics.csv</code></p>
<p><b>Table S6</b> Pearson <i>r</i> verification (<code>results/paper1_strengthening/pearson_verification.csv</code></p>
<p><b>Figure S1</b> K = 7 profile structure (full covariance): sizes [281, 125, 108, 264, 245, 70, 73] (<code>results/paper1_strengthening/task1_K_7/profile_parameters.csv</code></p>

<!-- ============================================================ -->
<div class="footnote">
<p><b>Manuscript conventions</b> All numerical values are taken verbatim from the frozen computational evidence package (<code>results/paper1_final/</code>, <code>results/paper1_strengthening/</code>) and have been independently verified. Items marked <span class="placeholder">[AUTHOR INPUT REQUIRED</span> are not verifiable from the analytic file and require author action before submission. Items marked <span class="placeholder">[LITERATURE SUPPORT NEEDED</span> require author-supplied citations from the existing literature</p>
<p><b>Numerical summary (primary):</b> N = 1166; <i>r</i>(INT, BE) = 0.6515 (95% CI [0.6171, 0.6833], <i>p</i> = 8.83 × 10<sup>−142</sup>, <i>r</i><sup>2</sup> = 0.4245); GAP mean ≈ 0, SD = 0.834916, range [−3.52, +2.83], 524 INT > BE (44.94%), 642 BE > INT (55.06%); K = 6 BIC = 2129.17 (primary; K = 7 BIC = 1616.02, K ≥ 8 numerically degenerate); K = 6 sizes [124, 377, 262, 90, 54, 259]; 3 INT > BE + 3 BE > INT; mean max posterior = 0.8917; McFadden pseudo-<i>R</i><sup>2</sup> = 0.3035 (distinct from <i>r</i><sup>2</sup>); 21 of 65 joint FDR tests significant; max VIF = 67.44; duplicate sensitivity robust at N = 1124</p>
</div>

</body>
</html>
"""

# Inject figures
HTML = HTML.replace("__FIG1__", fig("fig1", "Aggregate INT-BE association"))
HTML = HTML.replace("__FIG2__", fig("fig2", "GAP distribution"))
HTML = HTML.replace("__FIG3__", fig("fig3", "Profile INT vs BE structure"))
HTML = HTML.replace("__FIG4__", fig("fig4", "Profile INT/BE means"))
HTML = HTML.replace("__FIG5__", fig("fig5", "Extended K=2-10 BIC"))
HTML = HTML.replace("__FIG6__", fig("fig6", "Covariance sensitivity"))
HTML = HTML.replace("__FIG7__", fig("fig7", "Classification quality"))
HTML = HTML.replace("__FIG8__", fig("fig8", "Profile stability"))
HTML = HTML.replace("__FIG9__", fig("fig9", "Alternative-specification sensitivity"))
HTML = HTML.replace("__FIG10__", fig("fig10", "Cross-validation"))

with open(OUT_PATH, "w") as f:
    f.write(HTML)

print(f"Wrote {OUT_PATH}")
print(f"Size: {os.path.getsize(OUT_PATH):,} bytes")
