#!/usr/bin/env python3
"""Generate the 11-figure + S1 set for the draft manuscript — K=7 PRIMARY.

The primary K is read dynamically from results/05_lpa_selection/selected_model.csv
(currently K=7, lowest BIC among well-behaved models); the leading competing
solution is the lowest-BIC well-behaved K != primary. All data from frozen CSVs
plus results/k7_primary/ (validated primary-K analyses). No annotations, no
'Figure N.' prefixes.
"""
import base64
import io
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

SINGLE = (6.5, 4.8)
WIDE = (10.5, 4.8)
OUT = "results/draft_figures"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/png", exist_ok=True)

PF = "results/paper1_final"
S = "results/paper1_strengthening"
K7 = "results/k7_primary"

# --- dynamic solution constants (no hardcoded K / N / bootstrap count) ---
SEL = int(pd.read_csv(
    "results/05_lpa_selection/selected_model.csv")["selected_K"].iloc[0])
_kext_all = pd.read_csv(f"{S}/k_extended_model_comparison.csv")
_kext_full = _kext_all[_kext_all.covariance_type == 'full']
_KMIN_EXT, _KMAX_EXT = int(_kext_full.K.min()), int(_kext_full.K.max())
_wb = _kext_full[_kext_full.log_likelihood < 0]   # numerically well-behaved
COMP = int(_wb[_wb.K != SEL].sort_values("BIC").K.iloc[0])
_deg_ks = sorted(int(k) for k in _kext_full.loc[_kext_full.log_likelihood > 0, "K"])
DEG_K_MIN = min(_deg_ks) if _deg_ks else None
N_TOTAL = int(pd.read_csv(
    "results/01_data_inspection/data_dimensions.csv")["N_rows"].iloc[0])
_bootm = pd.read_csv("results/14_k6_stability/k6_stability_master.csv")
N_BOOT = int(dict(zip(_bootm.metric, _bootm.value))["n_bootstrap"])

figures = {}


def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b


# ---------- Fig 1: aggregate INT-BE association ----------
def fig1():
    scores = pd.read_csv(f"{PF}/02_measurement/construct_scores.csv", index_col="respondent")
    INT, BE = scores["INT"].values, scores["BE"].values
    fig, ax = plt.subplots(figsize=SINGLE)
    hb = ax.hexbin(INT, BE, gridsize=30, cmap='Blues', mincnt=1, alpha=0.85)
    fig.colorbar(hb, ax=ax, label='Count')
    slope, intercept, r_val, p_val, se = stats.linregress(INT, BE)
    xs = np.linspace(INT.min(), INT.max(), 100)
    ax.plot(xs, intercept + slope * xs, 'r-', lw=2, label=f'Linear fit (r = {r_val:.4f})')
    lim = (1, 5)
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.5, label='Equality (INT = BE)')
    ax.set(xlim=lim, ylim=lim,
           xlabel='Intention (INT) mean score (1-5)', ylabel='Behaviour (BE) mean score (1-5)',
           title='Aggregate Intention-Behaviour Association')
    ax.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    return to_b64(fig)


# ---------- Fig 2: GAP distribution ----------
def fig2():
    gs = pd.read_csv(f"{PF}/03_intention_behavior_gap/gap_statistics.csv")
    gs_v = dict(zip(gs.statistic, gs.value))
    gap_sd = gs_v['SD']
    scores = pd.read_csv(f"{PF}/02_measurement/construct_scores.csv", index_col="respondent")
    z_INT = (scores["INT"] - scores["INT"].mean()) / scores["INT"].std(ddof=1)
    z_BE = (scores["BE"] - scores["BE"].mean()) / scores["BE"].std(ddof=1)
    GAP = z_INT - z_BE
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.hist(GAP, bins=40, density=True, alpha=0.6, color='#4C72B0',
            edgecolor='white', linewidth=0.5)
    from scipy.stats import gaussian_kde, norm
    kde = gaussian_kde(GAP)
    xs = np.linspace(GAP.min(), GAP.max(), 200)
    ax.plot(xs, kde(xs), 'k-', lw=1.5, label='KDE')
    ax.plot(xs, norm.pdf(xs, 0, gap_sd), 'r--', lw=1.5, label=f'Normal(0, {gap_sd:.4f})')
    ax.axvline(0, color='k', linestyle=':', lw=1)
    ax.set(xlabel='GAP = z(INT) - z(BE)', ylabel='Density',
           title='Distribution of the Standardised Intention-Behaviour Gap')
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    return to_b64(fig)


# ---------- Fig 3: extended K BIC (full covariance) ----------
def fig3():
    k_ext = pd.read_csv(f"{S}/k_extended_model_comparison.csv")
    full = k_ext[k_ext.covariance_type == 'full']
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(full.K, full.BIC, 'o-', color='#4C72B0', lw=2, markersize=8)
    for ksel, col, tag in [(SEL, '#27AE60', f'K={SEL} (primary, lowest well-behaved BIC)'),
                           (COMP, '#2E86C1', f'K={COMP} (competing, well-behaved)')]:
        krow = full[full.K == ksel]
        ax.scatter(krow.K, krow.BIC, s=200, facecolors='none', edgecolors=col,
                   linewidths=3, zorder=5, label=tag)
    if DEG_K_MIN is not None and (full.K >= DEG_K_MIN).any():
        ax.axvspan(DEG_K_MIN - 0.5, _KMAX_EXT + 0.5, alpha=0.1, color='red',
                   label=f'K>={DEG_K_MIN}: numerical degeneracy')
    ax.set(xlabel='Number of profiles (K)', ylabel='BIC',
           title=f'Model Fit Across K = {_KMIN_EXT}-{_KMAX_EXT} (Full Covariance)')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_xticks(range(_KMIN_EXT, _KMAX_EXT + 1))
    plt.tight_layout()
    return to_b64(fig)


# ---------- Fig 4: full vs diagonal covariance ----------
def fig4():
    cov = pd.read_csv(f"{S}/covariance_k_extended.csv")
    fig, ax = plt.subplots(figsize=SINGLE)
    f = cov[cov.covariance_type == 'full']
    d = cov[cov.covariance_type == 'diag']
    ax.plot(f.K, f.BIC, 'o-', color='#4C72B0', lw=2, markersize=8, label='Full covariance')
    ax.plot(d.K, d.BIC, 's--', color='#DD8452', lw=2, markersize=8, label='Diagonal covariance')
    ax.set(xlabel='Number of profiles (K)', ylabel='BIC',
           title='Covariance Specification Sensitivity: BIC by K')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xticks(range(int(cov.K.min()), int(cov.K.max()) + 1))
    plt.tight_layout()
    return to_b64(fig)


# ---------- Fig 5: K=7 profile configuration in z-space ----------
def fig5():
    pt = pd.read_csv(f"{K7}/k7_profile_table_full_sample.csv")
    fig, ax = plt.subplots(figsize=SINGLE)
    lim_min = min(pt.mean_z_INT.min(), pt.mean_z_BE.min()) - 0.2
    lim_max = max(pt.mean_z_INT.max(), pt.mean_z_BE.max()) + 0.2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1, alpha=0.5,
            label='Equality (z(INT) = z(BE))')
    colors = ['#E74C3C' if r.mean_z_INT > r.mean_z_BE else '#3498DB' for _, r in pt.iterrows()]
    for i, r in pt.iterrows():
        ax.scatter(r.mean_z_INT, r.mean_z_BE, s=r.N * 2, c=colors[i], alpha=0.8,
                   edgecolors='k', linewidth=0.5, zorder=5)
        ax.annotate(f"P{i}", (r.mean_z_INT, r.mean_z_BE), xytext=(5, 5),
                    textcoords='offset points', fontsize=10, fontweight='bold')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=10, label='z(INT) > z(BE)', alpha=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',
               markersize=10, label='z(BE) > z(INT)', alpha=0.8),
        Line2D([0], [0], color='k', linestyle='--', lw=1, alpha=0.5,
               label='Equality (z(INT) = z(BE))'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    ax.set(xlabel='Mean z(Intention)', ylabel='Mean z(Behaviour)',
           title=f'K = {SEL} Profile Configuration in z-Space',
           xlim=(lim_min, lim_max), ylim=(lim_min, lim_max))
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return to_b64(fig)


# ---------- Fig 6: K=7 profile means on original 1-5 scale ----------
def fig6():
    pt = pd.read_csv(f"{K7}/k7_profile_table_full_sample.csv")
    K_N = len(pt)
    fig, ax = plt.subplots(figsize=WIDE)
    x = np.arange(K_N)
    w = 0.35
    ax.bar(x - w/2, pt.INT_mean, w, label='Intention', color='#4C72B0',
           edgecolor='k', linewidth=0.5)
    ax.bar(x + w/2, pt.BE_mean, w, label='Behaviour', color='#DD8452',
           edgecolor='k', linewidth=0.5)
    ax.set(xlabel='Profile', ylabel='Mean score (1-5 scale)',
           title=f'K = {SEL} Profile Means on the Original 1-5 Scale',
           xticks=x, xticklabels=[f"P{i} (n={int(n)})" for i, n in zip(pt.profile, pt.N)],
           ylim=(1, 5.4))
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    return to_b64(fig)


# ---------- Fig 7: GAP variance decomposition ----------
def fig7():
    gv = pd.read_csv(f"{K7}/gap_variance_decomposition_K2_K7.csv")
    sub = gv[gv.K.isin(sorted({2, COMP - 1, COMP, SEL}))].copy()
    fig, ax = plt.subplots(figsize=SINGLE)
    x = np.arange(len(sub))
    w = 0.35
    ax.bar(x - w/2, sub.between_over_total * 100, w, label='Between-profile',
           color='#4C72B0', edgecolor='k', linewidth=0.5)
    ax.bar(x + w/2, (1 - sub.between_over_total) * 100, w, label='Within-profile',
           color='#DD8452', edgecolor='k', linewidth=0.5)
    for xi, v in zip(x - w/2, sub.between_over_total * 100):
        ax.text(xi, v + 1.2, f"{v:.2f}%", ha='center', fontsize=9)
    ax.set(xlabel='K', ylabel='% of total GAP variance',
           title='GAP Variance: Between- vs Within-Profile',
           xticks=x, xticklabels=[f"K = {k}" for k in sub.K], ylim=(0, 105))
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    return to_b64(fig)


# ---------- Fig 8: K=7 classification quality ----------
def fig8():
    cq = pd.read_csv(f"{S}/k7_classification_quality.csv")
    v = dict(zip(cq.statistic, cq.value))
    pp = pd.read_csv(f"{S}/k7_posterior_probabilities.csv")
    max_post = pp[[c for c in pp.columns if c.startswith('post_profile_')]].max(axis=1)
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.hist(max_post, bins=30, alpha=0.7, color='#4C72B0', edgecolor='white', linewidth=0.5)
    for thr, c in [(0.70, '#27AE60'), (0.80, '#F39C12'), (0.90, '#E74C3C')]:
        ax.axvline(thr, color=c, linestyle='--', lw=1.5, label=f'Threshold {thr:.2f}')
    m = v['mean_max_posterior']
    ax.axvline(m, color='k', lw=1.5, label=f'Mean = {m:.4f}')
    ax.set(xlabel='Maximum posterior probability', ylabel='Number of respondents',
           title=f'Classification Quality: Maximum Posterior Distribution (K = {SEL})')
    ax.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    return to_b64(fig)


# ---------- Fig 9: K=7 bootstrap directional stability ----------
def fig9():
    st = pd.read_csv(f"{S}/k7_stability.csv")
    K_N = len(st)
    fig, ax = plt.subplots(figsize=SINGLE)
    x = np.arange(K_N)
    # directionally stable if >=80 (INT>BE) or <=20 (BE>INT); near chance 40-60
    colors = ['#27AE60' if (p >= 80 or p <= 20) else ('#E74C3C' if 40 < p < 60 else '#F39C12')
              for p in st.pct_INT_gt_BE]
    ax.bar(x, st.pct_INT_gt_BE, 0.6, color=colors, edgecolor='k', linewidth=0.5)
    ax.axhline(50, color='k', linestyle=':', lw=1.5, label='Chance (50%)')
    ax.axhline(80, color='gray', linestyle=':', lw=1, label='Stability threshold (80%)')
    ax.axhline(20, color='gray', linestyle=':', lw=1, label='Stability threshold (20%)')
    ax.set(xlabel='Profile', ylabel='% INT > BE across bootstraps',
           title=f'Bootstrap Directional Stability ({N_BOOT} Replications, K = {SEL})',
           xticks=x, xticklabels=[f"P{i}" for i in range(K_N)], ylim=(0, 105))
    legend_elements = [
        Patch(facecolor='#27AE60', edgecolor='k', label='Directionally stable (>=80% or <=20%)'),
        Patch(facecolor='#F39C12', edgecolor='k', label='Moderate (20-80%)'),
        Patch(facecolor='#E74C3C', edgecolor='k', label='Near chance (40-60%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    return to_b64(fig)


# ---------- Fig 10: CV K=2-7 ----------
def fig10():
    cv = pd.read_csv(f"{S}/cv5_aggregated_by_k.csv")
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.errorbar(cv.K, cv.mean_ll_test, yerr=cv.sd_ll_test, fmt='o-',
                color='#4C72B0', lw=2, markersize=8, capsize=5,
                label='Mean held-out log-likelihood (+/-SD)')
    k7 = cv[cv.K == SEL]
    ax.scatter(k7.K, k7.mean_ll_test, s=200, facecolors='none', edgecolors='#27AE60',
               linewidths=3, zorder=5, label=f'K={SEL} (primary; highest mean LL, large fold SD)')
    k6 = cv[cv.K == COMP]
    ax.scatter(k6.K, k6.mean_ll_test, s=200, facecolors='none', edgecolors='#2E86C1',
               linewidths=3, zorder=5, label=f'K={COMP} (leading competitor among K=2-{COMP})')
    ax.set(xlabel='Number of profiles (K)', ylabel='Held-out log-likelihood',
           title='Five-Fold Cross-Validation: Held-Out Log-Likelihood by K')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xticks(range(int(cv.K.min()), int(cv.K.max()) + 1))
    plt.tight_layout()
    return to_b64(fig)


# ---------- Fig 11: duplicate sensitivity (selected K) ----------
def fig11():
    ds = pd.read_csv(f"{K7}/k7_duplicate_sensitivity.csv")
    g = dict(zip(ds.metric, pd.to_numeric(ds.frozen, errors='coerce')))
    u = dict(zip(ds.metric, pd.to_numeric(ds.unique_sample, errors='coerce')))
    N_FULL = int(g['N_full_sample'])
    N_UNIQ = int(g['N_unique_sample'])
    K_SEL = SEL
    prof = pd.read_csv(f"{K7}/k7_duplicate_sensitivity_profiles.csv")
    pp = pd.read_csv(f"{S}/task1_K_{SEL}/profile_parameters.csv")
    frozen_sizes = pp["size"].values
    # inverse Hungarian map: frozen profile -> unique-sample N
    uniq_size_by_frozen = {}
    for _, r in prof.iterrows():
        uniq_size_by_frozen[int(r.matched_frozen_profile)] = int(r.N)
    u_sizes = [uniq_size_by_frozen[j] for j in range(K_SEL)]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    w = 0.35
    c1, c2 = '#4C72B0', '#DD8452'

    def paired_bars(ax, labels, fv, uv, title, ylim, fmt, ylab=None):
        x = np.arange(len(labels))
        ax.bar(x - w/2, fv, w, color=c1, edgecolor='k', linewidth=0.5)
        ax.bar(x + w/2, uv, w, color=c2, edgecolor='k', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim)
        off = (ylim[1] - ylim[0]) * 0.02
        for xi, v in zip(x - w/2, fv):
            ax.text(xi, v + off, fmt % v, ha='center', fontsize=7)
        for xi, v in zip(x + w/2, uv):
            ax.text(xi, v + off, fmt % v, ha='center', fontsize=7)
        if ylab:
            ax.set_ylabel(ylab, fontsize=9)

    paired_bars(axes[0], ['Pearson r', 'GAP SD'],
                [g['pearson_r_INT_BE'], g['gap_SD']],
                [u['pearson_r_INT_BE'], u['gap_SD']],
                'Aggregate metrics', (0, 1.0), '%.3f', 'Value')

    paired_bars(axes[1], ['Entropy', 'Mean max post.', 'Min class N\n(shown / 1000)'],
                [g['K7_entropy'], g['K7_mean_max_posterior'], g['K7_min_class_N'] / 1000.0],
                [u['K7_entropy'], u['K7_mean_max_posterior'], u['K7_min_class_N'] / 1000.0],
                f'K = {SEL} fit and classification', (0, 1.25), '%.3f', 'Value (min class N / 1000)')

    paired_bars(axes[2], ['% INT > BE', '% BE > INT'],
                [g['pct_INT_gt_BE'], g['pct_BE_gt_INT']],
                [u['pct_INT_gt_BE'], u['pct_BE_gt_INT']],
                'Gap direction', (0, 70), '%.2f', '% of respondents')

    x = np.arange(K_SEL)
    axes[3].bar(x - w/2, frozen_sizes, w, color=c1, edgecolor='k', linewidth=0.5)
    axes[3].bar(x + w/2, u_sizes, w, color=c2, edgecolor='k', linewidth=0.5)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels([f"P{i}" for i in range(K_SEL)], fontsize=8)
    axes[3].set_title('Profile sizes (Hungarian-matched)', fontsize=10)
    axes[3].set_ylabel('N', fontsize=9)
    axes[3].set_ylim(0, max(max(frozen_sizes), max(u_sizes)) * 1.2)

    legend_elements = [
        Patch(facecolor=c1, edgecolor='k', label=f'Primary (N = {N_FULL})'),
        Patch(facecolor=c2, edgecolor='k', label=f'Duplicates removed (N = {N_UNIQ})'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=8)
    fig.suptitle(f'Duplicate-Row Sensitivity (K = {SEL})', fontsize=12, y=1.02)
    plt.tight_layout()
    return to_b64(fig)


# ---------- Fig S1: K=6 profile structure (supplementary) ----------
def figS1():
    # Competing-solution (K=6) structure from the genuine per-K LPA artifacts.
    # The legacy results/paper1_final/05_profiles/05_k6_profile_table.csv now
    # holds the selected-solution data, so build the frame here instead.
    comp_k = int(pd.read_csv(
        "results/05_lpa_selection/selected_model.csv")["selected_K"].iloc[0])
    _wb = pd.read_csv("results/paper1_strengthening/k_extended_model_comparison.csv")
    _wb = _wb[(_wb.covariance_type == 'full') & (_wb.log_likelihood < 0)]
    comp_k = int(_wb[_wb.K != comp_k].sort_values("BIC").K.iloc[0])
    means = pd.read_csv(f"results/04_lpa_estimation/K_{comp_k}/profile_means.csv")
    sizes = pd.read_csv(f"results/04_lpa_estimation/K_{comp_k}/profile_sizes.csv")
    size_col = "size" if "size" in sizes.columns else "N"
    n_total = int(pd.read_csv("results/01_data_inspection/data_dimensions.csv")["N_rows"].iloc[0])
    pt = pd.DataFrame({
        "profile": range(len(means)),
        "mean_z_INT": means["z_INT"].values,
        "mean_z_BE": means["z_BE"].values,
        "N": [int(sizes.loc[sizes["profile"] == p, size_col].iloc[0])
              for p in range(len(means))],
    })
    pt["percentage"] = pt["N"] / n_total * 100.0
    fig, ax = plt.subplots(figsize=SINGLE)
    lim_min = min(pt.mean_z_INT.min(), pt.mean_z_BE.min()) - 0.2
    lim_max = max(pt.mean_z_INT.max(), pt.mean_z_BE.max()) + 0.2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1, alpha=0.5,
            label='Equality (z(INT) = z(BE))')
    colors = ['#E74C3C' if r.mean_z_INT > r.mean_z_BE else '#3498DB' for _, r in pt.iterrows()]
    for i, r in pt.iterrows():
        ax.scatter(r.mean_z_INT, r.mean_z_BE, s=r.N * 2, c=colors[i], alpha=0.8,
                   edgecolors='k', linewidth=0.5, zorder=5)
        ax.annotate(f"P{i}", (r.mean_z_INT, r.mean_z_BE), xytext=(5, 5),
                    textcoords='offset points', fontsize=10, fontweight='bold')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=10, label='z(INT) > z(BE)', alpha=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',
               markersize=10, label='z(BE) > z(INT)', alpha=0.8),
        Line2D([0], [0], color='k', linestyle='--', lw=1, alpha=0.5,
               label='Equality (z(INT) = z(BE))'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    ax.set(xlabel='Mean z(Intention)', ylabel='Mean z(Behaviour)',
           title=f'K = {comp_k} Profile Structure (Supplementary)',
           xlim=(lim_min, lim_max), ylim=(lim_min, lim_max))
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return to_b64(fig)


builders = [
    ('fig1', fig1), ('fig2', fig2), ('fig3', fig3), ('fig4', fig4),
    ('fig5', fig5), ('fig6', fig6), ('fig7', fig7), ('fig8', fig8),
    ('fig9', fig9), ('fig10', fig10), ('fig11', fig11), ('figS1', figS1),
]

for name, fn in builders:
    figures[name] = fn()
    with open(f"{OUT}/png/{name}.png", "wb") as f:
        f.write(base64.b64decode(figures[name]))
    print(f"  {name} OK")

with open(f"{OUT}/draft_figures_base64.json", "w") as f:
    json.dump(figures, f)
print(f"Saved {len(figures)} figures to {OUT}")
