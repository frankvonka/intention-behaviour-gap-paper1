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
assert len(cs) == 1166
INT, BE = cs["INT"].values, cs["BE"].values

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
ax.text(1.08, 4.85, "$r = 0.6515$\n95% CI [0.6171, 0.6833]\n$p < 0.001$\n$N = 1{,}166$",
        fontsize=10, va="top",
        bbox=dict(facecolor="white", edgecolor=PALETTE['neutral'], boxstyle="round,pad=0.3"))
ax.legend(loc="lower right", frameon=False)
save(1, fig, cs[["respondent", "INT", "BE"]],
     "Figure 1. Aggregate association between self-reported household energy-saving intention "
     "and behaviour (N = 1,166). Pearson r = 0.6515 (95% CI [0.6171, 0.6833], p = 8.83e-142). "
     "The strong aggregate association does not imply uniform individual alignment; the dotted "
     "diagonal represents perfect alignment.")

# ---------------- FIGURE 2 — Distribution of the standardized gap ----------------
INT_z = (INT - INT.mean()) / INT.std(ddof=1)
BE_z = (BE - BE.mean()) / BE.std(ddof=1)
GAP = INT_z - BE_z
gap_df = pd.DataFrame({"respondent": cs["respondent"], "GAP": GAP})
gx = np.linspace(GAP.min(), GAP.max(), 400)

fig, ax = plt.subplots(figsize=SINGLE)
ax.grid(True, axis="y")
ax.hist(GAP, bins=50, density=True, color=PALETTE['primary'], alpha=0.70, edgecolor="white", linewidth=0.4)
ax.plot(gx, stats.norm.pdf(gx, GAP.mean(), GAP.std()), color=PALETTE['secondary'], label="Normal density")
ax.axvline(0, color=PALETTE['neutral'], lw=1.0, ls="--")
ax.set(xlabel=r"Standardized intention–behaviour gap, $z(\mathrm{INT}) - z(\mathrm{BE})$",
       ylabel="Density",
       title="Distribution of the intention–behaviour gap")
ax.text(0.03, 0.96, "Mean $\\approx 0$\n$SD = 0.8349$\nRange $[-3.520, +2.827]$\n$N = 1{,}166$",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(facecolor="white", edgecolor=PALETTE['neutral'], boxstyle="round,pad=0.3"))
ax.legend(loc="upper right", frameon=False)
save(2, fig, gap_df,
     "Figure 2. Distribution of the respondent-level standardized intention-behaviour gap, "
     "GAP = z(INT) - z(BE) (N = 1,166). Mean approximately 0, SD = 0.8349, range [-3.520, +2.827]. "
     "Both positive (INT > BE) and negative (BE > INT) discrepancies occur.")

# ---------------- FIGURE 3 — Direction of the discrepancy (wide) ----------------
n_int_gt_be, n_be_gt_int, total = 524, 642, 1166
pct_int = n_int_gt_be / total * 100
pct_be = n_be_gt_int / total * 100

fig, ax = plt.subplots(figsize=WIDE)
ax.barh([0], [pct_int], color=PALETTE['primary'], edgecolor="white", height=0.5)
ax.barh([0], [pct_be], left=[pct_int], color=PALETTE['secondary'], edgecolor="white", height=0.5)
ax.axvline(50, color="white", lw=1.2, ls=":")
ax.set(xlim=(0, 100), ylim=(-0.6, 0.6), yticks=[], xticks=[0, 25, 50, 75, 100],
       xticklabels=["0%", "25%", "50%", "75%", "100%"],
       xlabel="Percentage of respondents ($N = 1{,}166$)",
       title="Direction of respondent-level intention–behaviour discrepancy")
ax.text(pct_int / 2, 0, f"{pct_int:.2f}%  INT > BE  ($n = 524$)",
        ha="center", va="center", color="white", fontsize=10)
ax.text(pct_int + pct_be / 2, 0, f"{pct_be:.2f}%  BE > INT  ($n = 642$)",
        ha="center", va="center", color="white", fontsize=10)
save(3, fig, pd.DataFrame({"direction": ["INT > BE", "BE > INT"],
                           "n": [n_int_gt_be, n_be_gt_int],
                           "pct": [pct_int, pct_be]}),
     "Figure 3. Direction of respondent-level intention-behaviour discrepancy. 44.94% of "
     "respondents (n = 524) had INT > BE; 55.06% (n = 642) had BE > INT. No respondent had "
     "INT = BE.")

# ---------------- FIGURE 4 — Model selection across K ----------------
lpa = pd.read_csv(f"{B}/04_lpa/04_lpa_comparison_table.csv")

fig, ax = plt.subplots(figsize=SINGLE)
ax.grid(True)
ax.plot(lpa["K"], lpa["BIC"], marker="o", color=PALETTE['primary'], markersize=5)
for k, b in zip(lpa["K"], lpa["BIC"]):
    c = PALETTE['secondary'] if k == 6 else PALETTE['primary']
    ax.plot([k], [b], marker="o", color=c, markersize=7, zorder=5)
    ax.annotate(f"{b:,.2f}", (k, b), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9,
                color=PALETTE['secondary'] if k == 6 else PALETTE['neutral'])
ax.set(xticks=[2, 3, 4, 5, 6], ylim=(1500, 6400),
       xlabel="Number of profiles ($K$)",
       ylabel="Bayesian Information Criterion (BIC)",
       title="Model selection for Gaussian mixture profiles")
ax.annotate("Minimum BIC\n(primary specification)", (6, 2129.17), xytext=(4.3, 2700),
            fontsize=9.5, color=PALETTE['secondary'],
            arrowprops=dict(arrowstyle="->", color=PALETTE['secondary'], lw=0.8))
save(4, fig, lpa,
     "Figure 4. BIC values for Gaussian mixture profiles estimated for K = 2 to 6 under the "
     "primary specification (full covariance, n_init = 1000, seed = 42, mean scores). K = 6 was "
     "selected as the primary specification because it produced the minimum BIC (2129.17) among "
     "K = 2-6 under the prespecified primary model.")

# ---------------- FIGURE 5 — Six-profile configuration ----------------
pt = pd.read_csv(f"{B}/05_profiles/05_k6_profile_table.csv")

fig, ax = plt.subplots(figsize=WIDE)
ax.grid(True)
for i, row in pt.iterrows():
    ax.plot([0, 1], [row["INT_mean"], row["BE_mean"]], marker="o",
            color=PROFILE_COLORS[i], markersize=6, lw=1.8, label=f"P{i} ($n = {int(row['N'])}$)")
ax.set(xlim=(-0.35, 1.35), ylim=(1.8, 5.3), xticks=[0, 1],
       xticklabels=["Intention", "Behaviour"],
       ylabel="Mean score (1–5 scale)",
       title="Six-profile configuration of intention and behaviour")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9,
          labelspacing=0.5, handlelength=1.5)
save(5, fig, pt,
     "Figure 5. Mean intention and behaviour scores for each of the six profiles under the "
     "primary K = 6 specification. P0, P2, P5 show INT > BE; P1, P3, P4 show BE > INT. Profile "
     "sizes: P0 = 124, P1 = 377, P2 = 262, P3 = 90, P4 = 54, P5 = 259. Profile labels carry no "
     "substantive meaning beyond the numerical configuration.")

# ---------------- FIGURE 6 — Profile-specific discrepancy ----------------
fig, ax = plt.subplots(figsize=SINGLE)
ax.grid(True, axis="y")
sizes = pt["N"].values
diffs = pt["INT_minus_BE"].values
bar_colors = [PALETTE['primary'] if d > 0 else PALETTE['secondary'] for d in diffs]
ax.bar(range(6), diffs, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.65)
ax.axhline(0, color=PALETTE['neutral'], lw=0.8)
for i, d in enumerate(diffs):
    va = "bottom" if d >= 0 else "top"
    off = 0.02 if d >= 0 else -0.02
    ax.text(i, d + off, f"{d:+.4f}", ha="center", va=va, fontsize=9)
ax.set(xticks=range(6), xticklabels=[f"P{i}\n($n = {int(s)}$)" for i, s in enumerate(sizes)],
       ylim=(-0.65, 0.72),
       ylabel="Intention minus behaviour (INT − BE)",
       title="Profile-specific intention–behaviour discrepancy")
ax.text(0.02, 0.97, "Navy: INT > BE\nCrimson: BE > INT", transform=ax.transAxes, va="top",
        fontsize=9, bbox=dict(facecolor="white", edgecolor=PALETTE['neutral'],
                                boxstyle="round,pad=0.3"))
save(6, fig, pt,
     "Figure 6. Profile-specific intention-behaviour discrepancy (INT - BE) for the K = 6 primary "
     "solution. Positive values (navy) indicate INT > BE; negative values (crimson) indicate "
     "BE > INT. P5 (n = 259) is near balance (+0.1052) and exhibited only 48% direction "
     "consistency across 200 bootstrap replications.")

# ---------------- FIGURE 7 — Classification quality ----------------
pp = pd.read_csv(f"{B}/04_lpa/posterior_probabilities.csv")
post_cols = [c for c in pp.columns if c.startswith("post_profile_")]
max_post = pp[post_cols].max(axis=1).values

fig, ax = plt.subplots(figsize=SINGLE)
ax.grid(True, axis="y")
ax.hist(max_post, bins=30, color=PALETTE['primary'], alpha=0.75, edgecolor="white", linewidth=0.4)
for thr, c in [(0.70, PALETTE['accent_1']), (0.80, PALETTE['accent_2']), (0.90, PALETTE['secondary'])]:
    ax.axvline(thr, color=c, ls="--", lw=1.0)
ax.set(xlim=(0.45, 1.02),
       xlabel="Maximum posterior probability", ylabel="Respondents",
       title="Classification quality of the six-profile solution")
ax.text(0.03, 0.96,
        "Mean = 0.8917\nMedian = 0.9618\n$\\geq 0.90$: 66.12%\n$\\geq 0.80$: 75.30%\n"
        "$\\geq 0.70$: 88.85%\n$< 0.50$: 0%",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(facecolor="white", edgecolor=PALETTE['neutral'], boxstyle="round,pad=0.3"))
save(7, fig, pd.DataFrame({"respondent": pp["respondent"], "max_posterior": max_post}),
     "Figure 7. Distribution of respondent-level maximum posterior probability under the K = 6 "
     "primary solution (N = 1,166). Mean = 0.8917, median = 0.9618; 66.12% at or above 0.90, "
     "75.30% at or above 0.80, 88.85% at or above 0.70; 0% below 0.50.")

# ---------------- FIGURE 8 — Bootstrap directional stability (wide) ----------------
cs_stab = pd.read_csv(f"{B}/05_profiles/configuration_stability.csv")
dominant = cs_stab[["pct_INT_gt_BE", "pct_BE_gt_INT"]].max(axis=1)

fig, ax = plt.subplots(figsize=WIDE)
ax.grid(True, axis="x")
bar_colors = [PALETTE['accent_2'] if p < 60 else PALETTE['primary'] for p in dominant]
ax.barh(range(6), dominant, color=bar_colors, edgecolor="white", height=0.6)
ax.axvline(50, color=PALETTE['neutral'], lw=1.0, ls="--")
ax.text(50.5, 5.55, "50% (chance)", fontsize=9.5, color=PALETTE['neutral'])
for i in range(6):
    a, b = cs_stab["pct_INT_gt_BE"].iloc[i], cs_stab["pct_BE_gt_INT"].iloc[i]
    ax.text(dominant.iloc[i] + 1.2, i, f"{a:.1f}% INT > BE / {b:.1f}% BE > INT",
            va="center", fontsize=9.5)
ax.set(yticks=range(6), yticklabels=[f"P{i}" for i in range(6)], xlim=(0, 118), ylim=(-0.6, 5.9),
       xlabel="Direction consistency across 200 bootstrap replications (%)",
       title="Bootstrap stability of profile-specific intention–behaviour direction")
save(8, fig, cs_stab,
     "Figure 8. Direction consistency of each profile across 200 bootstrap replications of the "
     "K = 6 latent profile analysis, reported as % INT > BE / % BE > INT. P0-P4 are directionally "
     "stable (80.5%-97.5%); P5 is near chance (48.0% / 52.0%) and should be interpreted "
     "cautiously. Profiles are not equally stable.")

# ---------------- FIGURE 9 — FDR-significant predictor counts ----------------
ps = pd.read_csv(f"{B}/06_predictors/predictor_summary.csv")
ps_sorted = ps.sort_values("n_significant_contrasts", ascending=True)

fig, ax = plt.subplots(figsize=SINGLE)
ax.grid(True, axis="x")
colors = [PALETTE['primary'] if n > 0 else PALETTE['grid'] for n in ps_sorted["n_significant_contrasts"]]
ax.barh(ps_sorted["predictor"], ps_sorted["n_significant_contrasts"], color=colors,
        edgecolor="white", height=0.65)
for i, n in enumerate(ps_sorted["n_significant_contrasts"]):
    ax.text(n + 0.12, i, str(int(n)), va="center", fontsize=9, color=PALETTE['neutral'])
ax.set(xlim=(0, 6.2), xticks=[0, 1, 2, 3, 4, 5],
       xlabel="FDR-significant contrasts (of 5)",
       title="Predictor associations with profile membership")
ax.tick_params(axis="y", labelsize=10)
ax.text(0.98, 0.04, "21 of 65 contrasts significant\n(joint Benjamini–Hochberg FDR)\n"
                    "Maximum VIF = 67.44",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor=PALETTE['neutral'], boxstyle="round,pad=0.3"))
save(9, fig, ps_sorted,
     "Figure 9. Number of FDR-significant multinomial logistic regression contrasts per predictor "
     "(Profile 0 reference; 13 predictors x 5 contrasts = 65 tests; joint Benjamini-Hochberg FDR "
     "at alpha = 0.05). 21 of 65 contrasts were significant. Severe multicollinearity among "
     "construct predictors (maximum VIF = 67.44) means individual coefficients are not uniquely "
     "identified; associations are exploratory, not causal.")

# ---------------- FIGURE 10 — Five-fold cross-validation ----------------
cv = pd.read_csv(f"{S}/cv5_aggregated_by_k.csv")

fig, ax = plt.subplots(figsize=SINGLE)
ax.grid(True)
ax.errorbar(cv["K"], cv["mean_ll_test"], yerr=cv["sd_ll_test"], fmt="o-",
            color=PALETTE['primary'], ecolor=PALETTE['neutral'], capsize=3, markersize=5)
ax.plot([6], [cv.loc[cv["K"] == 6, "mean_ll_test"].iloc[0]], marker="o",
        color=PALETTE['secondary'], markersize=8, zorder=5)
for k, ll in zip(cv["K"], cv["mean_ll_test"]):
    pass
    ax.annotate(f"{ll:.1f}", (k, ll), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=9,
                color=PALETTE['secondary'] if k == 6 else PALETTE['neutral'])
ax.set(xticks=[2, 3, 4, 5, 6],
       xlabel="Number of profiles ($K$)",
       ylabel="Mean held-out log-likelihood",
       title="Cross-validation support for the six-profile solution")
save(10, fig, cv,
     "Figure 10. Five-fold cross-validation held-out log-likelihood by K (K = 2 to 6); error "
     "bars show +/- 1 SD across folds. K = 6 had the highest mean held-out log-likelihood, "
     "providing additional support for the six-profile primary specification.")

print("All 10 figures generated under unified publication standards.")
