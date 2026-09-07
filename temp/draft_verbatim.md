Heterogeneous Intention–Behaviour Configurations in Household Energy-Saving Behaviour: A Latent Profile Analysis of N = 1166 Respondents


Abstract
Background
A core assumption in behavioural theory is that intention is an important proximal determinant of behaviour, yet substantial discrepancies between stated intention and reported behaviour have been documented. An important question is whether such discrepancies represent a relatively uniform population-level phenomenon or whether distinct subgroups exhibit qualitatively different intention–behaviour configurations.
Objective
To examine heterogeneous intention–behaviour configurations in a cross-sectional sample of N = 1166 respondents reporting household energy-saving behaviour using Latent Profile Analysis (LPA) of standardised intention (INT) and behaviour (BE) indicators.
Methods
A 10-construct psychometric battery comprising 28 Likert-scale items was administered. The constructs were attitude, context, subjective norm, COVID context, intention, behaviour, perceived usefulness, perceived ease of use, personal obligation, and personal responsibility. Construct scores were calculated as arithmetic means of their constituent items; Cronbach's α ranged from 0.673 to 0.919. The intention–behaviour gap was defined as GAPi=z(INTi)−z(BEi)GAP_i=z(INT_i)-z(BE_i), with within-sample standardisation using ddof = 1. LPA was estimated using sklearn GaussianMixture models with full covariance, n_init = 1000, random_state = 42, max_iter = 500, and reg_covar = 10−6. Models with K = 2–10 profiles were examined. Model selection was based primarily on BIC, supplemented by classification quality, bootstrap directional stability, cross-validation, and sensitivity analyses. Exploratory profile-membership associations were examined using multinomial logistic regression with 13 predictors and joint Benjamini–Hochberg false-discovery-rate correction across 65 tests.
Results
The aggregate INT–BE association was strong, r = 0.6515, 95% CI [0.6171, 0.6833], p = 8.83 × 10−142, corresponding to r² = 0.4245. The standardised GAP had mean −4.875 × 10−16, SD = 0.834916, and range [−3.52, +2.83]. Of the 1166 respondents, 524 (44.94%) had INT > BE and 642 (55.06%) had BE > INT. Under the primary full-covariance specification, the K = 6 solution had BIC = 2129.17 and comprised profiles of sizes 124, 377, 262, 90, 54, and 259. Three profiles had mean INT > mean BE and three had mean BE > mean INT. Mean maximum posterior probability was 0.8917, with 88.85% of respondents having maximum posterior probability ≥0.70. Bootstrap directional stability ranged from 2.5% to 87.5% for the proportion of replications in which the matched profile had INT > BE; P5 was near chance at 48.0%. Five-fold cross-validation produced the highest mean held-out log-likelihood for K = 6 among K = 2–6 (−287.58). However, K = 7 had a lower BIC (1616.02) than K = 6, while K ≥ 8 produced numerically degenerate solutions characterised by positive log-likelihoods and singular covariance behaviour. Diagonal covariance also produced lower BIC than full covariance at K = 5–7. Removal of 42 duplicate rows produced negligible changes in the principal findings. Exploratory multinomial logistic regression yielded 21 FDR-significant associations among 65 tests, with McFadden pseudo-R² = 0.3035; severe multicollinearity was present (maximum VIF = 67.44).
Conclusion
The data show that a strong population-level intention–behaviour association coexists with heterogeneous intention–behaviour configurations. The primary K = 6 solution identifies three profiles with INT > BE and three with BE > INT, but the exact profile structure is model-dependent. In particular, K = 7 has a lower BIC, diagonal covariance provides a competing specification, and one profile (P5) is directionally unstable across bootstrap samples. The findings therefore support heterogeneity in intention–behaviour configurations while requiring cautious interpretation of the exact number and boundaries of latent profiles.
Keywords: intention–behaviour gap; energy-saving behaviour; household energy; latent profile analysis; person-centred analysis; pro-environmental behaviour; behavioural heterogeneity

1. Introduction
The relationship between behavioural intention and subsequent behaviour occupies a central position in behavioural research. Intention is generally conceptualised as an important proximal determinant of behaviour, yet empirical research has repeatedly documented incomplete correspondence between what individuals intend to do and what they report doing.
In the context of the Theory of Planned Behavior and the Norm Activation Model, behavioural intention is frequently identified as the primary antecedent of behaviour. While established frameworks posit intention as the proximal determinant, meta-analytic evidence increasingly demonstrates that intention accounts for only a modest share of the variance in actual behaviour (Sheeran & Gemmecke et al., 2025), highlighting a persistent intention–behaviour gap.
The discrepancy between intention and behaviour has commonly been examined through variable-centred approaches. Such approaches estimate the average strength of the intention–behaviour relationship across a population and investigate variables that may strengthen or weaken that relationship.
Research has sought to bridge this gap by examining moderators of the intention–behaviour relationship, such as behavioural control, habit strength, and normative constraints (Fielding & Hornsey, 2026). These moderators, along with situational factors like opportunity and affordability, are consistently linked to the strength of the relationship between intention and enactment (Webb et al., 2022; Webb et al., 2013).
A population-level association, however, does not necessarily imply that the relationship is configured similarly for all individuals. A correlation can be strong while individuals occupy substantially different combinations of intention and behaviour. For example, some respondents may report strong intentions but comparatively lower behaviour, whereas others may report behaviour that exceeds their stated intention. These configurations can be obscured by a single population-level coefficient.
Person-centred methods provide a complementary analytical perspective by examining whether observations can be represented as distinct configurations rather than assuming a single homogeneous population relationship. Latent Profile Analysis (LPA) is one such approach for continuous indicators and has been used to identify heterogeneous behavioural and psychological configurations.
Person-centred methods, such as Latent Profile Analysis (LPA), provide a complementary perspective by identifying distinct subgroups with unique psychological or behavioural configurations (Fielding & Hornsey, 2026). These approaches have gained traction in environmental and energy research for their capacity to disentangle the complexity of pro-environmental behavioural enactment beyond simple average effects.
The present study applies LPA specifically to intention and behaviour in household energy-saving behaviour. The analysis has two principal objectives. First, it examines whether a strong aggregate association between intention and behaviour coexists with heterogeneous respondent-level configurations. Second, it characterises the resulting profiles according to the direction and magnitude of the intention–behaviour configuration.
Because latent profile solutions are inherently model-dependent, the analysis explicitly examines model-selection uncertainty, covariance specification, classification quality, bootstrap directional stability, cross-validation, and sensitivity to duplicate observations. The study does not interpret latent profiles as causal types or claim that a single profile solution represents a universally stable taxonomy of households.

2. Methods
2.1 Sample and Data
The sample consists of N = 1166 respondents from a single cross-sectional survey. The analytic dataset contains 34 measured variables comprising 28 psychometric items and six demographic variables. There were no missing values across the measured cells.
Forty-two rows (3.60%) were flagged as duplicates on all measured columns in the raw data file. These observations were retained in the primary analysis to preserve the original analytic sample. A sensitivity analysis was conducted after removing the 42 duplicate rows, resulting in N = 1124.

2.2 Measurement
Ten constructs were scored as arithmetic means of their constituent items. All items used a 1–5 Likert response scale.
Table 1. Construct descriptive statistics and internal consistency
Construct
Abbreviation
Items
Mean
SD
Cronbach's α
Attitude
ATT
3
3.627
0.782
0.7021
Context
CON
3
3.862
0.778
0.7621
Subjective Norm
SNO
3
3.943
0.837
0.8055
COVID Context
COVID
3
3.955
0.770
0.7729
Intention
INT
3
3.891
0.765
0.7823
Behaviour
BE
4
3.789
0.754
0.7480
Perceived Usefulness
PU
3
3.840
0.750
0.7274
Perceived Ease of Use
PEU
2
3.599
0.892
0.6727
Personal Obligation
PO
2
3.766
0.898
0.8113
Personal Responsibility
PRI
3
3.963
0.714
0.9186

Check these abbrivations and make sure they are correct.

2.3 Aggregate Intention–Behaviour Association
The aggregate association between INT and BE was quantified using Pearson's correlation on the arithmetic-mean construct scores.
The verified correlation was:
r=0.6515r=0.6515
with a 95% confidence interval of [0.6171, 0.6833], p = 8.83 × 10−142, and r² = 0.4245.
The reported Pearson correlation, confidence interval, and p-value were independently recomputed and verified against the frozen computational evidence.
Figure 1 — Aggregate intention–behaviour association
[INSERT FIGURE 1 HERE]
Figure 1. Aggregate intention–behaviour association. Joint distribution of INT and BE mean scores for N = 1166 respondents, with the fitted linear association and the equality reference line (INT = BE). The figure should report r = 0.6515, 95% CI [0.6171, 0.6833], and p = 8.83 × 10−142. Source: frozen construct-score evidence.

2.4 Intention–Behaviour Gap
The respondent-level intention–behaviour gap was defined as:
GAPi=z(INTi)−z(BEi)GAP_i=z(INT_i)-z(BE_i)
where standardisation was performed within the sample using ddof = 1.
The resulting GAP had a mean of −4.875 × 10−16 (effectively zero), SD = 0.834916, and range [−3.52, +2.83].
A total of 524 respondents (44.94%) had INT > BE, while 642 (55.06%) had BE > INT. No respondents had an exact equality under the z-score comparison.
The mean of GAP is approximately zero by construction because both INT and BE were independently standardised before subtraction. This does not imply that the GAP distribution is symmetric.
Figure 2 — Distribution of the intention–behaviour gap
[INSERT FIGURE 2 HERE]
Figure 2. Distribution of the standardised intention–behaviour gap. Distribution of GAP = z(INT) − z(BE) for N = 1166 respondents. The figure should show the empirical distribution and indicate the zero reference point. Reported GAP SD = 0.834916 and range = [−3.52, +2.83]. Source: frozen GAP statistics.

2.5 Latent Profile Analysis
LPA was conducted on two standardised indicators: z(INT) and z(BE).
The primary specification used sklearn.mixture.GaussianMixture with:
indicators: z(INT) and z(BE);
covariance structure: full;
n_init = 1000;
random_state = 42;
max_iter = 500;
reg_covar = 10−6.
Models with K = 2–10 profiles were fitted. BIC was calculated as:
BIC=−2log⁡L+klog⁡(N)BIC=-2\log L+k\log(N)
For the full-covariance two-indicator model, the number of free parameters was:
k=5K−1.k=5K-1.
At K = 6, the model therefore contained 35 free parameters.
Because increasing the number of mixture components can result in numerical covariance collapse, the extended K search was evaluated for numerical behaviour in addition to BIC.

2.6 Profile Matching
When comparing K = 6 solutions across bootstrap samples, alternative specifications, or random seeds, profiles were matched using the two-dimensional vector of profile means:
[z(INT)‾,z(BE)‾].[\overline{z(INT)},\overline{z(BE)}].
Pairwise profile distances were calculated using Euclidean distance. Optimal one-to-one assignment was obtained using the Hungarian algorithm (scipy.optimize.linear_sum_assignment) over the resulting K × K cost matrix.

2.7 Profile-Membership Predictors
Profile membership under the K = 6 solution was examined using multinomial logistic regression with Profile 0 as the reference category.
The model included 13 predictors:
Psychological constructs
ATT
CON
SNO
COVID
PU
PEU
PO
PRI
Demographic variables
age
gender
education
occupation
income
The analysis therefore contained 13 predictors × five non-reference profile contrasts = 65 statistical tests.
Estimation was conducted using statsmodels.MNLogit with BFGS optimisation and maxiter = 1000. Benjamini–Hochberg FDR correction was applied jointly across all 65 tests at α = 0.05.
This analysis is explicitly treated as exploratory and associational. The cross-sectional design does not support causal inference. Furthermore, construct predictors exhibited severe multicollinearity, with a maximum VIF of 67.44. Individual predictor coefficients therefore should not be interpreted as uniquely identified independent effects.

2.8 Validation and Sensitivity Analyses
Several validation and sensitivity analyses were conducted.
Bootstrap directional stability
Two hundred bootstrap replications of the K = 6 model were estimated. Profiles were matched using their two-dimensional profile means, and directional consistency was calculated as the proportion of matched bootstrap profiles for which mean INT exceeded mean BE.
Five-fold cross-validation
Five-fold cross-validation was conducted for K = 2–6. Models were refitted within each training fold and evaluated using held-out log-likelihood.
Duplicate-row sensitivity
The primary K = 6 analysis was repeated after removing the 42 duplicate rows, resulting in N = 1124.
Covariance sensitivity
Full and diagonal covariance structures were compared for K = 2–10.
Extended K search
The primary full-covariance specification was extended from K = 2–6 to K = 2–10.
Alternative specifications
Additional sensitivity analyses examined spherical covariance, alternative score representations, and random seeds.

3. Results
3.1 Measurement and Aggregate INT–BE Association
Internal consistency ranged from α = 0.6727 for PEU to α = 0.9186 for PRI. Descriptive statistics and reliability coefficients are presented in Table 1.
The aggregate INT–BE association was strong, with r = 0.6515, 95% CI [0.6171, 0.6833], p = 8.83 × 10−142, and r² = 0.4245.
Figure 1 should be placed here.

3.2 Intention–Behaviour Gap Distribution
The standardised GAP had mean approximately zero and SD = 0.834916. Its observed range was −3.52 to +2.83.
Of the 1166 respondents, 524 (44.94%) had INT > BE, while 642 (55.06%) had BE > INT. Thus, although the aggregate correlation was strong, the direction of the respondent-level standardised discrepancy was not uniform across the sample.
Figure 2 should be placed here.

3.3 Latent Profile Model Selection
The full-covariance models produced the following fit statistics.
Table 2. Full-covariance Gaussian mixture model comparison
K
Log-likelihood
Parameters
AIC
BIC
Entropy
Smallest class N
Assessment
2
−2927.59
11
5877.18
5932.85
1.5998
295
Well-behaved
3
−2893.29
17
5820.59
5906.63
1.6111
88
Well-behaved
4
−2247.32
23
4540.64
4657.05
1.4236
124
Well-behaved
5
−2214.32
29
4486.64
4633.42
1.3416
56
Well-behaved
6
−941.01
35
1952.03
2129.17
1.1418
54
Primary solution
7
−663.25
41
1408.51
1616.02
1.0991
70
Lower BIC than K = 6
8
+2141.47
47
−4188.93
−3951.05
1.0185
68
Numerical pathology
9
+2160.47
53
−4214.94
−3946.69
1.0208
17
Numerical pathology
10
+2861.03
59
−5604.06
−5305.44
1.0003
10
Numerical pathology

The K = 6 solution had BIC = 2129.17 and was retained as the primary substantive solution. Importantly, this is not the global minimum BIC across K = 2–10. The K = 7 model had a lower BIC of 1616.02.
The K ≥ 8 solutions exhibited positive log-likelihoods and numerical behaviour consistent with singular covariance solutions. These solutions were therefore not treated as interpretable latent profile solutions.
Accordingly, K = 6 is described as the minimum-BIC solution among the numerically well-behaved models through K = 6, rather than as the global BIC optimum.
Figure 3 — Extended K model comparison
[INSERT FIGURE 3 HERE]
Figure 3. Model fit across K = 2–10. BIC under the primary full-covariance specification. The figure should clearly distinguish the retained K = 6 primary solution, the numerically well-behaved K = 7 solution with lower BIC, and the K ≥ 8 solutions showing numerical degeneracy. The figure must not visually imply that K = 6 is the global BIC minimum.

3.4 Covariance Specification Sensitivity
The comparison between full and diagonal covariance structures showed that diagonal covariance produced lower BIC values at K = 5, 6, and 7.
Table 3. Full versus diagonal covariance
K
Full BIC
Diagonal BIC
Lower BIC
2
5932.85
6152.89
Full
3
5906.63
6000.69
Full
4
4657.05
4700.72
Full
5
4633.42
2487.11
Diagonal
6
2129.17
898.65
Diagonal
7
1616.02
477.83
Diagonal

The primary full-covariance specification was retained because it permits profile-specific covariance between INT and BE and therefore corresponds directly to the substantive interest in intention–behaviour configuration. Nevertheless, the diagonal solution represents a legitimate competing specification and is an important limitation on the interpretation of the exact K = 6 structure.
Figure 4 — Covariance sensitivity
[INSERT FIGURE 4 HERE]
Figure 4. BIC under full and diagonal covariance specifications. BIC across K = 2–10 for full and diagonal covariance models. The figure should clearly show that diagonal covariance has lower BIC at K = 5–7, including K = 6.

3.5 K = 6 Profile Structure
The primary K = 6 solution comprised six profiles with sample sizes of 124, 377, 262, 90, 54, and 259.
Table 4. K = 6 profile descriptives
Profile
N
%
INT mean
BE mean
INT − BE
Direction
P0
124
10.63%
5.000
4.466
+0.534
INT > BE
P1
377
32.33%
3.337
3.560
−0.223
BE > INT
P2
262
22.47%
4.478
3.957
+0.521
INT > BE
P3
90
7.72%
2.333
2.344
−0.011
BE > INT
P4
54
4.63%
4.438
4.912
−0.474
BE > INT
P5
259
22.21%
4.000
3.895
+0.105
INT > BE
Total
1166
100.00%
—
—
—
3 + 3

Three profiles had mean INT > BE: P0, P2, and P5.
Three profiles had mean BE > INT: P1, P3, and P4.
The profile structure therefore contains both directions of intention–behaviour discrepancy rather than a single universal direction.
Figure 5 — K = 6 intention–behaviour configuration
[INSERT FIGURE 5 HERE]
Figure 5. K = 6 intention–behaviour profile configuration. Profile-specific means of standardised INT and BE. The equality line represents INT = BE. Profiles should be labelled P0–P5 and scaled according to profile size where appropriate. The figure should show three profiles above the equality line (P0, P2, P5) and three below it (P1, P3, P4).
Figure 6 — Original-scale profile means
[INSERT FIGURE 6 HERE]
Figure 6. Profile-specific intention and behaviour means on the original 1–5 scale. Bars or points should display mean INT and BE for each K = 6 profile, with profile sample sizes included in the labels.

3.6 GAP Variance Decomposition
At K = 6, between-profile variance in GAP was 0.1852, representing 26.57% of the total GAP variance of 0.6971.
Within-profile variance was 0.5134, representing 73.43% of total GAP variance.
Thus, the latent profiles account for a meaningful but minority share of the observed GAP variance. Most GAP variation remains within profiles rather than between them.
For comparison, the between-profile share was 14.81% at K = 5 and 4.15% at K = 2.
Figure 7 — Between- and within-profile GAP variance
[INSERT FIGURE 7 HERE]
Figure 7. GAP variance decomposition at K = 2, K = 5, and K = 6. The figure should show the proportion of GAP variance attributable to between-profile and within-profile differences. For K = 6, 26.57% is between profiles and 73.43% is within profiles.

3.7 Classification Quality
Classification quality was high for most observations.
The mean maximum posterior probability was 0.8917, with a median of 0.9618 and SD of 0.1429.
The distribution was:
66.12% with maximum posterior probability ≥ 0.90;
75.30% with maximum posterior probability ≥ 0.80;
88.85% with maximum posterior probability ≥ 0.70;
11.15% with maximum posterior probability < 0.70.
Thus, although most observations had relatively high classification probabilities, classification was not perfect.
Figure 8 — Classification quality
[INSERT FIGURE 8 HERE]
Figure 8. Maximum posterior probability for the K = 6 solution. Distribution of maximum posterior probabilities, with reference thresholds at 0.70, 0.80, and 0.90. Mean = 0.8917 and median = 0.9618.

3.8 Bootstrap Directional Stability
Bootstrap directional stability was examined over 200 replications.
Table 5. Bootstrap directional stability
Profile
% bootstrap replications with INT > BE
Interpretation
P0
80.5%
Directionally stable
P1
12.5%
Directionally stable as BE > INT
P2
87.5%
Directionally stable
P3
18.0%
Directionally stable as BE > INT
P4
2.5%
Directionally stable as BE > INT
P5
48.0%
Unstable / near chance

P0, P1, P2, P3, and P4 showed directionally consistent behaviour. P5 was near chance, with only 48.0% of bootstrap replications showing INT > BE.
Therefore, P5's direction should not be treated as robustly established.
Figure 9 — Bootstrap directional stability
[INSERT FIGURE 9 HERE]
Figure 9. Bootstrap directional stability of K = 6 profiles. Percentage of 200 bootstrap replications in which each matched profile had mean INT > mean BE. The figure should make clear that P5 is near chance at 48.0%, whereas the other profiles show substantially greater directional consistency.

3.9 Five-Fold Cross-Validation
Five-fold cross-validation was conducted for K = 2–6.
Table 6. Five-fold cross-validation
K
Mean held-out log-likelihood
SD
2
−588.90
18.70
3
−583.17
17.56
4
−473.97
23.54
5
−407.55
101.90
6
−287.58
123.46

K = 6 produced the highest mean held-out log-likelihood among the models evaluated in the five-fold cross-validation.
Figure 10 — Cross-validation
[INSERT FIGURE 10 HERE]
Figure 10. Five-fold cross-validation performance. Mean held-out log-likelihood and standard deviation for K = 2–6. K = 6 has the highest mean held-out log-likelihood (−287.58).

3.10 Duplicate-Row Sensitivity
The primary analysis contained 42 duplicate rows. Removing these observations produced N = 1124.
Table 7. Duplicate-row sensitivity
Metric
Primary N = 1166
Unique N = 1124
Pearson r
0.651458
0.654410
K = 6 BIC
2129.17
2021.49
K = 6 log-likelihood
−941.01
−887.81
K = 6 entropy
1.1418
1.1494
% INT > BE
44.94%
45.11%
% BE > INT
55.06%
54.89%
Profile direction structure
3 + 3
3 + 3
Mean maximum posterior
0.8917
0.8848

Removal of the duplicate rows produced only small changes in the aggregate correlation, GAP direction proportions, classification quality, and profile-direction structure. The three INT > BE and three BE > INT structure was preserved.
Figure 11 — Duplicate sensitivity
[INSERT FIGURE 11 HERE]
Figure 11. Sensitivity of key results to duplicate-row removal. Comparison of primary (N = 1166) and duplicate-excluded (N = 1124) results. The figure should show that the principal findings are materially unchanged.

3.11 Exploratory Profile-Membership Associations
Multinomial logistic regression was used to examine exploratory associations between profile membership and 13 predictors.
Table 8. Multinomial logistic regression model diagnostics
Diagnostic
Value
Sample size
1166
Outcome classes
6
Reference profile
P0
Predictors
13
Joint FDR tests
65
Log-likelihood
−1309.94
AIC
2759.88
BIC
3114.18
McFadden pseudo-R²
0.3035
Maximum VIF
67.44
Condition number
270.98
FDR-significant tests
21 of 65
Missing predictor values
0
Convergence
Yes

Twenty-one of the 65 jointly FDR-adjusted tests were statistically significant.
The construct predictors exhibited severe multicollinearity, with maximum VIF = 67.44 and condition number = 270.98. Consequently, the predictor analysis is interpreted as a joint exploratory pattern rather than as evidence for uniquely identifiable effects of individual predictors.
The McFadden pseudo-R² = 0.3035 is a likelihood-based model-fit statistic and should not be interpreted as the proportion of variance explained. It is not directly comparable with the Pearson r² = 0.4245 from the aggregate INT–BE association.
Full coefficient estimates, confidence intervals, raw p-values, and FDR-adjusted p-values are reported in Supplementary Table S2.

4. Discussion
The principal finding is that a strong aggregate association between intention and behaviour coexists with substantial heterogeneity in respondent-level intention–behaviour configurations. The aggregate correlation was r = 0.6515, yet the K = 6 solution contained three profiles with mean INT > BE and three profiles with mean BE > INT.
Our findings regarding the coexistence of strong aggregate intention-behaviour correlations and substantial respondent-level heterogeneity align with recent literature (Webb et al., 2022; Sheeran / Gemmecke et al., 2025). While previous research often emphasizes variables that moderate the intention-behaviour relationship (Fielding & Hornsey, 2026), our person-centred approach demonstrates that a single motivational explanation (Webb et al., 2013) is often insufficient to capture the complexity of household energy behaviour. Consistent with recent meta-analytic evidence (Carrero et al., 2025; 2025 systematic review), our results suggest that segmenting populations into intention-behaviour profiles captures meaningful heterogeneity, even as the majority of discrepancy remains within-profile. This supports the recent shift toward modelling intention and behaviour as distinct, simultaneously observed outcomes, addressing the over-reliance on intention as a solitary endpoint in current TPB/NAM research (2026).
At the same time, the profile solution should not be interpreted as evidence for six universally distinct behavioural types. At K = 6, only 26.57% of GAP variance was between profiles, whereas 73.43% remained within profiles. Thus, the latent profiles capture a meaningful component of the observed heterogeneity but do not account for the majority of individual-level variation in GAP.
Our findings regarding the co-occurrence of INT > BE and BE > INT configurations are consistent with recent meta-analytic evidence (Carrero et al., 2025; Sheeran & Gemmecke et al., 2025). This bidirectional discrepancy challenges the traditional view that intention typically exceeds behaviour, suggesting instead that "action-beyond-intention" is a significant phenomenon in energy-saving contexts.
Several methodological qualifications are central to interpretation.
First, K = 6 is not the global BIC optimum across the entire extended search. K = 7 had a lower BIC (1616.02) than K = 6 (2129.17). K ≥ 8 produced numerical behaviour consistent with singular covariance solutions and therefore was not substantively interpreted. The K = 6 solution should therefore be described as the primary, interpretable and parsimonious solution rather than as an unequivocally optimal global solution.
Second, covariance specification affects model selection. Diagonal covariance produced lower BIC than full covariance at K = 5–7, including K = 6. The full-covariance specification was retained because it allows covariance between INT and BE within each profile and is therefore directly relevant to the substantive configuration being examined. Nevertheless, the diagonal result demonstrates that the exact latent structure depends on modelling assumptions.
Third, classification quality was high for most respondents but not perfect. Approximately 11.15% of observations had a maximum posterior probability below 0.70. More importantly, P5 was directionally unstable in the bootstrap analysis, with only 48.0% of matched bootstrap profiles showing INT > BE. Its directional classification therefore requires particular caution.
Fourth, the duplicate-row sensitivity analysis provides evidence that the principal findings are not driven by the 42 duplicated observations. After their removal, r changed from 0.6515 to 0.6544, the INT > BE proportion changed from 44.94% to 45.11%, and the three-plus-three directional profile structure was preserved.
Finally, the exploratory predictor analysis should not be used to infer causal mechanisms. Although 21 of 65 FDR-adjusted tests were significant, the construct predictors exhibited severe multicollinearity, with maximum VIF = 67.44. Individual coefficients are therefore not interpreted as uniquely identified independent effects.
[LITERATURE SUPPORT NEEDED: Situate the person-centred finding within prior environmental and energy-saving behaviour research and identify the theoretical contribution of distinguishing aggregate association from heterogeneous configuration.]

5. Limitations
Several limitations should be considered.
5.1 Cross-sectional design
The data are cross-sectional. Consequently, the analysis cannot establish temporal ordering or causal relationships between intention, behaviour, predictors, or profile membership.
5.2 Model dependence
The latent profile structure depends on modelling choices, including the number of profiles, covariance specification, and score representation.
5.3 K = 7 has lower BIC
K = 7 has BIC = 1616.02, lower than the K = 6 BIC of 2129.17. K = 6 is therefore not the global BIC optimum across the full K = 2–10 search.
5.4 Numerical degeneracy at K ≥ 8
The K = 8, 9, and 10 full-covariance solutions produced positive log-likelihoods and behaviour consistent with singular covariance matrices. These solutions were not treated as substantive profile solutions.
5.5 Diagonal covariance is a competing specification
Diagonal covariance produced lower BIC at K = 5–7. This represents an important alternative specification and limits confidence in the exact K = 6 structure.
5.6 P5 directional instability
P5 showed 48.0% bootstrap consistency for the INT > BE direction, approximately equivalent to chance. Its directional interpretation is therefore not robust.
5.7 Within-profile variance dominates
At K = 6, 73.43% of GAP variance remained within profiles. The profiles therefore capture only a minority of total GAP variance.
5.8 Predictor multicollinearity
The exploratory predictor model had a maximum VIF of 67.44 and condition number of 270.98. Individual predictor coefficients should therefore not be interpreted as uniquely identified independent effects.
5.9 Duplicate observations
Forty-two duplicate rows were retained in the primary analysis. However, the duplicate-exclusion sensitivity analysis produced materially similar results.
5.10 PEU reliability
PEU had the lowest internal-consistency coefficient, α = 0.6727, based on two items. Findings involving PEU should therefore be interpreted cautiously.
5.11 Classification uncertainty
Although the mean maximum posterior probability was 0.8917, 11.15% of respondents had maximum posterior probability below 0.70.
5.12 Generalisability
The generalisability of the findings depends on the sampling frame, recruitment procedure, demographic composition, and survey context.
[AUTHOR INPUT REQUIRED: sampling frame, recruitment method, demographic-category definitions, and other information needed to evaluate generalisability.]

6. Conclusion
This study examined intention–behaviour configurations in household energy-saving behaviour using a person-centred latent profile approach. The aggregate association between intention and behaviour was strong (r = 0.6515), but the population-level association coexisted with heterogeneous respondent-level configurations.
Under the primary full-covariance specification, the K = 6 solution identified six profiles comprising three profiles with mean INT > BE and three with mean BE > INT. The profiles captured 26.57% of GAP variance, while 73.43% remained within profiles, indicating that the latent structure captures a meaningful but incomplete component of intention–behaviour heterogeneity.
The evidence supports the central proposition that a strong aggregate intention–behaviour association can coexist with heterogeneous configurations. However, the exact profile solution should not be regarded as definitive. K = 7 had a lower BIC than K = 6, diagonal covariance provided a competing lower-BIC specification at K = 6, and P5 was directionally unstable across bootstrap replications.
The duplicate-row sensitivity analysis preserved the principal findings, and five-fold cross-validation supported K = 6 within the range K = 2–6. The exploratory predictor analysis identified 21 FDR-significant associations among 65 tests, but severe multicollinearity prevents interpretation of individual coefficients as uniquely identified effects.
[AUTHOR INPUT REQUIRED: Add a final literature-grounded statement explaining the theoretical contribution of the findings and specific directions for future research.]

Supplementary Materials
Supplementary Table S1
Full and diagonal covariance model comparison across K = 2–10.
Source: results/paper1_strengthening/covariance_k_extended.csv
Supplementary Table S2
Full multinomial logistic regression coefficient table containing the 13 predictors × five non-reference contrasts = 65 tests, including coefficient estimates, confidence intervals, raw p-values, and FDR-adjusted p-values.
Source: results/paper1_final/06_predictors/
Supplementary Table S3
Duplicate-row sensitivity analysis at the profile level for N = 1124.
Source: results/paper1_strengthening/duplicate_sensitivity_profiles.csv
Supplementary Table S4
Alternative-specification sensitivity analyses, including full, diagonal, and spherical covariance, alternative score representations, and random seeds.
Source: results/paper1_final/07_robustness/10_robustness_table.csv
Supplementary Table S5
Construct descriptive statistics and reliability.
Source: results/paper1_final/02_measurement/construct_statistics.csv
Supplementary Table S6
Independent Pearson correlation verification.
Source: results/paper1_strengthening/pearson_verification.csv
Supplementary Figure S1
K = 7 profile structure under the full-covariance specification.
The K = 7 solution contains profile sizes [281, 125, 108, 264, 245, 70, 73].
Source: results/paper1_strengthening/task1_K_7/profile_parameters.csv

Data and Computational Transparency
All numerical values reported in this manuscript are taken from the frozen computational evidence package supplied for the study. No new calculations are introduced in this manuscript.
The primary numerical results are:
N = 1166;
Pearson r = 0.6515;
95% CI [0.6171, 0.6833];
p = 8.83 × 10−142;
r² = 0.4245;
GAP mean = −4.875 × 10−16;
GAP SD = 0.834916;
GAP range = [−3.52, +2.83];
INT > BE = 524 (44.94%);
BE > INT = 642 (55.06%);
K = 6 full-covariance BIC = 2129.17;
K = 7 full-covariance BIC = 1616.02;
K ≥ 8 = numerically degenerate;
K = 6 profile sizes = [124, 377, 262, 90, 54, 259];
three INT > BE profiles and three BE > INT profiles;
mean maximum posterior probability = 0.8917;
88.85% with maximum posterior ≥0.70;
K = 6 between-profile GAP variance = 26.57%;
K = 6 within-profile GAP variance = 73.43%;
K = 6 five-fold CV mean held-out log-likelihood = −287.58;
duplicate-excluded N = 1124;
duplicate-excluded Pearson r = 0.654410;
21 of 65 joint FDR tests significant;
McFadden pseudo-R² = 0.3035;
maximum VIF = 67.44.
[AUTHOR INPUT REQUIRED: complete references, author information, institutional information, sampling information, ethics information, demographic definitions, item wording, and theoretical construct citations.]

Recommended Final Figure Order
Figure 1: Aggregate INT–BE association
→ Establishes the strong population-level relationship.
Figure 2: GAP distribution
→ Shows respondent-level discrepancy and both directions.
Figure 3: Extended K = 2–10 BIC
→ Establishes model-selection uncertainty and the K = 7 issue.
Figure 4: Full vs diagonal covariance
→ Shows covariance-specification sensitivity.
Figure 5: K = 6 profile configuration in z-space
→ Main conceptual figure. Shows the six configurations and INT > BE / BE > INT directions.
Figure 6: K = 6 profile means on the original 1–5 scale
→ Makes the substantive magnitude of each profile easy to understand.
Figure 7: Between- vs within-profile GAP variance
→ Prevents overclaiming that profiles explain most individual heterogeneity.
Figure 8: Posterior classification quality
→ Establishes assignment quality.
Figure 9: Bootstrap directional stability
→ Explicitly exposes P5 instability.
Figure 10: Five-fold cross-validation
→ Additional support for K = 6 within K = 2–6.
Figure 11: Duplicate-row sensitivity
→ Demonstrates robustness to the 42 duplicate rows.
Supplementary Figure S1: K = 7 profile structure
→ Keeps the competing K = 7 solution visible without making it the main substantive figure.

Final Production Rules
The manuscript should be generated from the frozen results without changing numerical values.
The following must not be added unless explicitly supplied by the authors:
No invented references.
No invented sampling procedure.
No invented recruitment procedure.
No invented response rate.
No invented ethics approval.
No invented demographic definitions.
No invented questionnaire wording.
No invented theoretical citations.
No new statistical calculations.
No new analyses.
No new interpretation of individual multinomial coefficients.
No causal language.
No statement that K = 6 is the global BIC minimum.
No statement that P5 is a stable INT > BE profile.
No statement that the six profiles explain most of the intention–behaviour gap.
No statement that the predictor analysis identifies causal determinants.
No replacement of the frozen results with newly calculated values.
The manuscript should preserve the distinction between:
aggregate association → GAP distribution → latent configuration → model uncertainty → exploratory predictor associations.
This distinction is central to the interpretation of the study.


 ok i wrote this draft, check its integrity with the final verified results, make  aifnal html file and embed the figures in it where it needed. befor any changes in this fdraft tell me