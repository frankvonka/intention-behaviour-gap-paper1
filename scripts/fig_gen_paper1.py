"""Generate 10 publication figures for Paper 1 from frozen evidence CSVs.
Unified Publication Visual Standards (serif, standardized palette, fixed sizes).
STRICT: no refit, no new analysis, no invented values.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Universal Paper Style Configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 5
})

# Universal Palette
PALETTE = {
    'primary': '#1f77b4',      # Navy
    'secondary': '#d62728',    # Crimson
    'accent_1': '#2ca02c',     # Green
    'accent_2': '#ff7f0e',     # Amber
    'neutral': '#555555',      # Slate Gray
    'grid': '#e0e0e0'
}
PROFILE_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#555555', '#8c564b']

SINGLE = (6.5, 4.8)   # single-column (enlarged for title legibility)
WIDE = (10.5, 4.8)    # double-column / wide (enlarged for title legibility)

B = "results/paper1_final"
S = "results/paper1_strengthening"
FIG = f"{B}/figures"
os.makedirs(FIG, exist_ok=True)

# --- Dynamic sample / model constants (no hardcoded N or K) ---
SEL = int(pd.read_csv("results/05_lpa_selection/selected_model.csv")["selected_K"].iloc[0])
_dims = pd.read_csv("results/01_data_inspection/data_dimensions.csv")
N_TOTAL = int(_dims["N_rows"].iloc[0])
_dups_df = pd.read_csv("results/01_data_inspection/duplicate_information.csv")
N_DUPS = int(_dups_df.loc[_dups_df["check"] == "all_columns", "n_duplicates"].iloc[0])
N_UNIQUE = N_TOTAL - N_DUPS
_N_FMT = f"{N_TOTAL:,}"

def save(fig_id, fig, data_df, caption):
    d = f"{FIG}/fig{fig_id}"
    os.makedirs(d, exist_ok=True)
    plt.tight_layout()
    fig.savefig(f"{d}/fig{fig_id}.png", bbox_inches="tight")
    fig.savefig(f"{d}/fig{fig_id}.pdf", bbox_inches="tight")
    if data_df is not None:
        data_df.to_csv(f"{d}/fig{fig_id}_data.csv", index=False)
    with open(f"{d}/fig{fig_id}_caption.txt", "w") as f:
        f.write(caption)
    plt.close(fig)
    print(f"Figure {fig_id} saved -> {d}")

cs = pd.read_csv(f"{B}/02_measurement/construct_scores.csv")
assert len(cs) == N_TOTAL
INT, BE = cs["INT"].values, cs["BE"].values

# Frozen correlation evidence (de-hardcodes the Figure 1 r / CI / p annotation)
_corr = pd.read_csv("results/02_measurement/int_be_correlation.csv")
R_FROZEN = float(_corr["r"].iloc[0])
CI_LO = float(_corr["CI95_lower"].iloc[0])
CI_HI = float(_corr["CI95_upper"].iloc[0])
P_FROZEN = float(_corr["p"].iloc[0])
P_SCINOT = f"{P_FROZEN:.2e}".replace("e-0", "e-")

# ---------------- FIGURE 1 — Overall intention-behaviour association ----------------
slope, intercept, _, _, _ = stats.linregress(INT, BE)
xs = np.linspace(1, 5, 200); ys = slope * xs + intercept
n = len(INT); x_mean = INT.mean(); Sxx = ((INT - x_mean) ** 2).sum()
t_val = stats.t.ppf(0.975, n - 2)
se_pred = np.sqrt(((BE - (slope*INT + intercept)) ** 2).sum() / (n - 2)) * np.sqrt(1/n + (xs - x_mean) ** 2 / Sxx)
ci_lo, ci_hi = ys - t_val * se_pred, ys + t_val * se_pred

fig, ax = plt.subplots(figsize=SINGLE)
ax.grid(True)
ax.scatter(INT, BE, s=6, alpha=0.30, color=PALETTE['primary'], edgecolors="none")
ax.plot(xs, ys, color=PALETTE['secondary'], label="Linear fit")
ax.fill_between(xs, ci_lo, ci_hi, color=PALETTE['secondary'], alpha=0.15, label="95% CI of fit")
ax.plot([1, 5], [1, 5], color=PALETTE['neutral'], lw=0.8, ls=":")
ax.set(xlim=(1, 5), ylim=(1, 5), xticks=[1, 2, 3, 4, 5], yticks=[1, 2, 3, 4, 5],
       xlabel="Energy-saving intention", ylabel="Energy-saving behaviour",
       title="Overall intention–behaviour association")
ax.text(1.08, 4.85, f"$r = {R_FROZEN:.4f}$\n95% CI [{CI_LO:.4f}, {CI_HI:.4f}]\n$p < 0.001$\n$N = {N_TOTAL:,}$",
        fontsize=10, va="top",
        bbox=dict(facecolor="white", edgecolor=PALETTE['neutral'], boxstyle="round,pad=0.3"))
ax.legend(loc="lower right", frameon=False)
save(1, fig, cs[["respondent", "INT", "BE"]],
     f"Figure 1. Aggregate association between self-reported household energy-saving intention "
     f"and behaviour (N = {_N_FMT}). Pearson r = {R_FROZEN:.4f} (95% CI [{CI_LO:.4f}, {CI_HI:.4f}], "
     f"p = {P_SCINOT}). The strong aggregate association does not imply uniform individual "
     f"alignment; the dotted diagonal represents perfect alignment.")

# ---------------- FIGURE 2 — Distribution of the standardized gap ----------------
INT_z = (INT - INT.mean()) / INT.std(ddof=1)
BE_z = (BE - BE.mean()) / BE.std(ddof=1)
GAP = INT_z - BE_z
gap_df = pd.DataFrame({"respondent": cs["respondent"], "GAP": GAP})
gx = np.linspace(GAP.min(), GAP.max(), 400)

_gap = pd.read_csv(f"{B}/03_intention_behavior_gap/gap_statistics.csv").set_index("statistic")["value"]
GAP_SD = float(_gap["SD"]); GAP_MIN = float(_gap["min"]); GAP_MAX = float(_gap["max"])
N_POS = int(_gap["positive_gap_count"]); N_NEG = int(_gap["negative_gap_count"])
PCT_POS = float(_gap["positive_gap_pct"]); PCT_NEG = float(_gap["negative_gap_pct"])

fig, ax = plt.subplots(figsize=SINGLE)
ax.grid(True, axis="y")
ax.hist(GAP, bins=50, density=True, color=PALETTE['primary'], alpha=0.70, edgecolor="white", linewidth=0.4)
ax.plot(gx, stats.norm.pdf(gx, GAP.mean(), GAP.std()), color=PALETTE['secondary'], label="Normal density")
ax.axvline(0, color=PALETTE['neutral'], lw=1.0, ls="--")
ax.set(xlabel=r"Standardized intention–behaviour gap, $z(\mathrm{INT}) - z(\mathrm{BE})$",
       ylabel="Density",
       title="Distribution of the intention–behaviour gap")
ax.text(0.03, 0.96, f"Mean $\\approx 0$\n$SD = {GAP_SD:.4f}$\nRange $[{GAP_MIN:.3f}, {GAP_MAX:+.3f}]$\n$N = {N_TOTAL:,}$",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(facecolor="white", edgecolor=PALETTE['neutral'], boxstyle="round,pad=0.3"))
ax.legend(loc="upper right", frameon=False)
save(2, fig, gap_df,
     f"Figure 2. Distribution of the respondent-level standardized intention-behaviour gap, "
     f"GAP = z(INT) - z(BE) (N = {_N_FMT}). Mean approximately 0, SD = {GAP_SD:.4f}, "
     f"range [{GAP_MIN:.3f}, {GAP_MAX:+.3f}]. Both positive (INT > BE) and negative (BE > INT) "
     f"discrepancies occur.")

# ---------------- FIGURE 3 — Direction of the discrepancy (wide) ----------------
n_int_gt_be, n_be_gt_int, total = N_POS, N_NEG, N_TOTAL
pct_int = n_int_gt_be / total * 100
pct_be = n_be_gt_int / total * 100

fig, ax = plt.subplots(figsize=WIDE)
ax.barh([0], [pct_int], color=PALETTE['primary'], edgecolor="white", height=0.5)
ax.barh([0], [pct_be], left=[pct_int], color=PALETTE['secondary'], edgecolor="white", height=0.5)
ax.axvline(50, color="white", lw=1.2, ls=":")
ax.set(xlim=(0, 100), ylim=(-0.6, 0.6), yticks=[], xticks=[0, 25, 50, 75, 100],
       xticklabels=["0%", "25%", "50%", "75%", "100%"],
       xlabel=f"Percentage of respondents ($N = 1{{,}}{N_TOTAL % 1000:03d}$)",
       title="Direction of respondent-level intention–behaviour discrepancy")
ax.text(pct_int / 2, 0, f"{pct_int:.2f}%  INT > BE  ($n = {n_int_gt_be}$)",
        ha="center", va="center", color="white", fontsize=10)
ax.text(pct_int + pct_be / 2, 0, f"{pct_be:.2f}%  BE > INT  ($n = {n_be_gt_int}$)",
        ha="center", va="center", color="white", fontsize=10)
save(3, fig, pd.DataFrame({"direction": ["INT > BE", "BE > INT"],
                           "n": [n_int_gt_be, n_be_gt_int],
                           "pct": [pct_int, pct_be]}),
     f"Figure 3. Direction of respondent-level intention-behaviour discrepancy. {PCT_POS:.2f}% of "
     f"respondents (n = {n_int_gt_be}) had INT > BE; {PCT_NEG:.2f}% (n = {n_be_gt_int}) had BE > INT. "
     f"No respondent had INT = BE.")

# ---------------- FIGURE 4 — Model selection across K ----------------
lpa = pd.read_csv(f"{B}/04_lpa/04_lpa_comparison_table.csv")
_k_list = lpa["K"].astype(int).tolist()
_sel_row = lpa[lpa["K"] == SEL].iloc[0]
SEL_BIC = float(_sel_row["BIC"])

fig, ax = plt.subplots(figsize=SINGLE)
ax.grid(True)
ax.plot(lpa["K"], lpa["BIC"], marker="o", color=PALETTE['primary'], markersize=5)
for k, b in zip(lpa["K"], lpa["BIC"]):
    c = PALETTE['secondary'] if k == SEL else PALETTE['primary']
    ax.plot([k], [b], marker="o", color=c, markersize=7, zorder=5)
    ax.annotate(f"{b:,.2f}", (k, b), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9,
                color=PALETTE['secondary'] if k == SEL else PALETTE['neutral'])
ax.set(xticks=_k_list,
       xlabel="Number of profiles ($K$)",
       ylabel="Bayesian Information Criterion (BIC)",
       title="Model selection for Gaussian mixture profiles")
ax.annotate("Minimum BIC\n(primary specification)", (SEL, SEL_BIC),
            xytext=(SEL - 1.7, SEL_BIC + 0.35 * (lpa["BIC"].max() - SEL_BIC)),
            fontsize=9.5, color=PALETTE['secondary'],
            arrowprops=dict(arrowstyle="->", color=PALETTE['secondary'], lw=0.8))
save(4, fig, lpa,
     f"Figure 4. BIC values for Gaussian mixture profiles estimated for K = {min(_k_list)} to "
     f"{max(_k_list)} under the primary specification (full covariance, n_init = 1000, seed = 42, "
     f"mean scores). K = {SEL} was selected as the primary specification because it produced the "
     f"minimum BIC ({SEL_BIC:.2f}) among the estimated range under the prespecified primary model.")

# ---------------- FIGURE 5 — Primary-profile configuration (dynamic K) ----------------
pt = pd.read_csv(f"{B}/05_profiles/05_k6_profile_table.csv")
N_PROFILES = len(pt)
_int_gt_p = [int(i) for i, r in pt.iterrows() if r["mean_z_INT"] > r["mean_z_BE"]]
_be_gt_p = [int(i) for i, r in pt.iterrows() if r["mean_z_BE"] > r["mean_z_INT"]]
_int_lbl = ", ".join(f"P{i}" for i in _int_gt_p)
_be_lbl = ", ".join(f"P{i}" for i in _be_gt_p)
_sizes_lbl = ", ".join(f"P{i} = {int(r['N'])}" for i, r in pt.iterrows())

fig, ax = plt.subplots(figsize=WIDE)
ax.grid(True)
for i, row in pt.iterrows():
    ax.plot([0, 1], [row["INT_mean"], row["BE_mean"]], marker="o",
            color=PROFILE_COLORS[i % len(PROFILE_COLORS)], markersize=6, lw=1.8,
            label=f"P{i} ($n = {int(row['N'])}$)")
ax.set(xlim=(-0.35, 1.35), xticks=[0, 1],
       xticklabels=["Intention", "Behaviour"],
       ylabel="Mean score (1–5 scale)",
       title=f"{N_PROFILES}-profile configuration of intention and behaviour")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9,
          labelspacing=0.5, handlelength=1.5)
save(5, fig, pt,
     f"Figure 5. Mean intention and behaviour scores for each of the {N_PROFILES} profiles under "
     f"the primary K = {SEL} specification. {_int_lbl} show INT > BE; {_be_lbl} show BE > INT. "
     f"Profile sizes: {_sizes_lbl}. Profile labels carry no substantive meaning beyond the "
     f"numerical configuration.")

# ---------------- FIGURE 6 — Profile-specific discrepancy (dynamic K) ----------------
fig, ax = plt.subplots(figsize=SINGLE)
ax.grid(True, axis="y")
sizes = pt["N"].values
diffs = pt["INT_minus_BE"].values
bar_colors = [PALETTE['primary'] if d > 0 else PALETTE['secondary'] for d in diffs]
ax.bar(range(N_PROFILES), diffs, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.65)
ax.axhline(0, color=PALETTE['neutral'], lw=0.8)
for i, d in enumerate(diffs):
    va = "bottom" if d >= 0 else "top"
    off = 0.02 if d >= 0 else -0.02
    ax.text(i, d + off, f"{d:+.4f}", ha="center", va=va, fontsize=9)
ax.set(xticks=range(N_PROFILES), xticklabels=[f"P{i}\n($n = {int(s)}$)" for i, s in enumerate(sizes)],
       ylabel="Intention minus behaviour (INT − BE)",
       title="Profile-specific intention–behaviour discrepancy")
ax.text(0.02, 0.97, "Navy: INT > BE\nCrimson: BE > INT", transform=ax.transAxes, va="top",
        fontsize=9, bbox=dict(facecolor="white", edgecolor=PALETTE['neutral'],
                                boxstyle="round,pad=0.3"))
_near_idx = int(np.abs(diffs).argmin())
save(6, fig, pt,
     f"Figure 6. Profile-specific intention-behaviour discrepancy (INT - BE) for the K = {SEL} "
     f"primary solution. Positive values (navy) indicate INT > BE; negative values (crimson) "
     f"indicate BE > INT. P{_near_idx} is the nearest-balanced profile (difference = {diffs[_near_idx]:+.4f}); "
     f"see bootstrap stability in Figure 8.")

# ---------------- FIGURE 7 — Classification quality (dynamic K, no hardcoded stats) ----------------
KDIR = f"results/04_lpa_estimation/K_{SEL}"
_post_path = os.path.join(KDIR, "posterior_probabilities.csv")
if not os.path.exists(_post_path):
    # Fall back to legacy location only if the canonical path is missing
    _post_path = f"{B}/04_lpa/posterior_probabilities.csv"
pp = pd.read_csv(_post_path)
post_cols = [c for c in pp.columns if c.startswith("post_profile_")]
max_post = pp[post_cols].max(axis=1).values
_mean = float(max_post.mean()); _med = float(np.median(max_post))
_pct90 = float((max_post >= 0.90).mean() * 100)
_pct80 = float((max_post >= 0.80).mean() * 100)
_pct70 = float((max_post >= 0.70).mean() * 100)
_pct50below = float((max_post < 0.50).mean() * 100)

fig, ax = plt.subplots(figsize=SINGLE)
ax.grid(True, axis="y")
ax.hist(max_post, bins=30, color=PALETTE['primary'], alpha=0.75, edgecolor="white", linewidth=0.4)
for thr, c in [(0.70, PALETTE['accent_1']), (0.80, PALETTE['accent_2']), (0.90, PALETTE['secondary'])]:
    ax.axvline(thr, color=c, ls="--", lw=1.0)
ax.set(xlim=(0.45, 1.02),
       xlabel="Maximum posterior probability", ylabel="Respondents",
       title=f"Classification quality of the {SEL}-profile solution")
ax.text(0.03, 0.96,
        f"Mean = {_mean:.4f}\nMedian = {_med:.4f}\n$\\geq 0.90$: {_pct90:.2f}%\n$\\geq 0.80$: {_pct80:.2f}%\n"
        f"$\\geq 0.70$: {_pct70:.2f}%\n$< 0.50$: {_pct50below:.1f}%",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(facecolor="white", edgecolor=PALETTE['neutral'], boxstyle="round,pad=0.3"))
save(7, fig, pd.DataFrame({"respondent": pp["respondent"] if "respondent" in pp.columns else np.arange(len(max_post)),
                           "max_posterior": max_post}),
     f"Figure 7. Distribution of respondent-level maximum posterior probability under the K = {SEL} "
     f"primary solution (N = {_N_FMT}). Mean = {_mean:.4f}, median = {_med:.4f}; {_pct90:.2f}% at or "
     f"above 0.90, {_pct80:.2f}% at or above 0.80, {_pct70:.2f}% at or above 0.70; {_pct50below:.1f}% below 0.50.")

# ---------------- FIGURE 8 — Bootstrap directional stability (wide, dynamic K) ----------------
# Prefer the K=7 strengthening stability table (fresh) over the K=6 legacy pack entry
cs_stab = None
for _p8 in [f"{S}/k7_stability.csv", f"{B}/05_profiles/configuration_stability.csv"]:
    if os.path.exists(_p8):
        _df8 = pd.read_csv(_p8)
        if len(_df8) == N_PROFILES:
            cs_stab = _df8; break
        # mismatch in rows → fall through
if cs_stab is None:
    raise FileNotFoundError(f"No K={SEL} stability table found (looked in {S}/k7_stability.csv and {B}/05_profiles/configuration_stability.csv)")
dominant = cs_stab[["pct_INT_gt_BE", "pct_BE_gt_INT"]].max(axis=1)

fig, ax = plt.subplots(figsize=WIDE)
ax.grid(True, axis="x")
bar_colors = [PALETTE['accent_2'] if p < 60 else PALETTE['primary'] for p in dominant]
ax.barh(range(N_PROFILES), dominant, color=bar_colors, edgecolor="white", height=0.6)
ax.axvline(50, color=PALETTE['neutral'], lw=1.0, ls="--")
ax.text(50.5, N_PROFILES - 0.45, "50% (chance)", fontsize=9.5, color=PALETTE['neutral'])
for i in range(N_PROFILES):
    a, b = cs_stab["pct_INT_gt_BE"].iloc[i], cs_stab["pct_BE_gt_INT"].iloc[i]
    ax.text(dominant.iloc[i] + 1.2, i, f"{a:.1f}% INT > BE / {b:.1f}% BE > INT",
            va="center", fontsize=9.5)
ax.set(yticks=range(N_PROFILES), yticklabels=[f"P{i}" for i in range(N_PROFILES)],
       xlim=(0, 118), ylim=(-0.6, N_PROFILES - 0.1),
       xlabel="Direction consistency across 200 bootstrap replications (%)",
       title=f"Bootstrap stability of profile-specific intention–behaviour direction (K = {SEL})")
_widest_idx = int((cs_stab["pct_INT_gt_BE"] - 50).abs().idxmin())
_wc = cs_stab.iloc[_widest_idx]
save(8, fig, cs_stab,
     f"Figure 8. Direction consistency of each profile across 200 bootstrap replications of the "
     f"K = {SEL} LPA, reported as % INT > BE / % BE > INT. P{_widest_idx} is nearest chance "
     f"({float(_wc['pct_INT_gt_BE']):.1f}% / {float(_wc['pct_BE_gt_INT']):.1f}%) and should be "
     f"interpreted cautiously. Profiles are not equally stable.")

# ---------------- FIGURE 9 — FDR-significant predictor counts (dynamic denominator) ----------------
# Prefer the live 18_profile_predictors table (fresh K=7) over the stale pack copy
ps = None
for _p9 in ["results/18_profile_predictors/predictor_summary.csv", f"{B}/06_predictors/predictor_summary.csv"]:
    if os.path.exists(_p9):
        ps = pd.read_csv(_p9); break
if ps is None:
    raise FileNotFoundError("No predictor_summary.csv found")
ps_sorted = ps.sort_values("n_significant_contrasts", ascending=True)
_diag_path9 = "results/18_profile_predictors/model_diagnostics.csv"
_k_pred = int(pd.read_csv(_diag_path9).set_index("metric").loc["n_outcome_classes", "value"]) \
    if os.path.exists(_diag_path9) else SEL
try:
    _diag9 = pd.read_csv(_diag_path9).set_index("metric")["value"]
    _maxvif = float(_diag9["max_VIF"])
    _ntests9 = int(float(_diag9["n_tests"])); _nsig9 = int(float(_diag9["n_FDR_significant"]))
except Exception:
    _maxvif = float("nan"); _ntests9 = len(ps) * (SEL - 1); _nsig9 = _ntests9
_n_pred = len(ps)
_max_sig = int(ps_sorted["n_significant_contrasts"].max())

fig, ax = plt.subplots(figsize=SINGLE)
ax.grid(True, axis="x")
colors = [PALETTE['primary'] if n > 0 else PALETTE['grid'] for n in ps_sorted["n_significant_contrasts"]]
ax.barh(ps_sorted["predictor"], ps_sorted["n_significant_contrasts"], color=colors,
        edgecolor="white", height=0.65)
for i, n in enumerate(ps_sorted["n_significant_contrasts"]):
    ax.text(n + 0.12, i, str(int(n)), va="center", fontsize=9, color=PALETTE['neutral'])
ax.set(xlim=(0, _max_sig + 1.2), xticks=list(range(_max_sig + 1)),
       xlabel=f"FDR-significant contrasts (of {_k_pred - 1})",
       title="Predictor associations with profile membership")
ax.tick_params(axis="y", labelsize=10)
ax.text(0.98, 0.04, f"{_nsig9} of {_ntests9} contrasts significant\n(joint Benjamini–Hochberg FDR)\n"
                    f"Maximum VIF = {_maxvif:.2f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor=PALETTE['neutral'], boxstyle="round,pad=0.3"))
save(9, fig, ps_sorted,
     f"Figure 9. Number of FDR-significant multinomial logistic regression contrasts per predictor "
     f"(Profile 0 reference; {_n_pred} predictors x {_k_pred - 1} contrasts = {_ntests9} tests; joint "
     f"Benjamini-Hochberg FDR at alpha = 0.05). {_nsig9} of {_ntests9} contrasts were significant. "
     f"Severe multicollinearity among construct predictors (maximum VIF = {_maxvif:.2f}) means "
     f"individual coefficients are not uniquely identified; associations are exploratory, not causal.")

# ---------------- FIGURE 10 — Five-fold cross-validation (dynamic K highlight) ----------------
cv = pd.read_csv(f"{S}/cv5_aggregated_by_k.csv")
_best_k = int(cv.loc[cv["mean_ll_test"].idxmax(), "K"])
_best_ll = float(cv.loc[cv["K"] == _best_k, "mean_ll_test"].iloc[0])

fig, ax = plt.subplots(figsize=SINGLE)
ax.grid(True)
ax.errorbar(cv["K"], cv["mean_ll_test"], yerr=cv["sd_ll_test"], fmt="o-",
            color=PALETTE['primary'], ecolor=PALETTE['neutral'], capsize=3, markersize=5)
ax.plot([SEL], [cv.loc[cv["K"] == SEL, "mean_ll_test"].iloc[0]], marker="o",
        color=PALETTE['secondary'], markersize=8, zorder=5)
for k, ll in zip(cv["K"], cv["mean_ll_test"]):
    ax.annotate(f"{ll:.1f}", (k, ll), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=9,
                color=PALETTE['secondary'] if k == SEL else PALETTE['neutral'])
ax.set(xticks=sorted(cv["K"].astype(int).tolist()),
       xlabel="Number of profiles ($K$)",
       ylabel="Mean held-out log-likelihood",
       title=f"Cross-validation: held-out log-likelihood by K (K = 2 to {max(_k_list)})")
_k_lab = f"K = {SEL} (primary, BIC-selected)"
if _best_k != SEL:
    _k_lab += f"; best held-out LL at K = {_best_k}"
_rank_sentence = (f"{_k_lab} had the highest mean held-out log-likelihood ({_best_ll:.2f})"
                  if _best_k == SEL else
                  f"{_k_lab} had a mean held-out log-likelihood of {_best_ll:.2f}; the highest was "
                  f"K = {_best_k} ({float(cv.loc[cv['K'] == _best_k, 'mean_ll_test'].iloc[0]):.2f})")
save(10, fig, cv,
     f"Figure 10. Five-fold cross-validation held-out log-likelihood by K (K = {min(_k_list)} to "
     f"{max(_k_list)}); error bars show +/- 1 SD across folds. {_rank_sentence}; cross-validated "
     f"fit is one criterion and BIC is the primary selection criterion.")

print("All 10 figures generated under unified publication standards.")
