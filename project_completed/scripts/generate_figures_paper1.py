#!/usr/bin/env python3
"""
Generate all 10 publication figures for Paper 1 from frozen evidence CSVs.
No 'Figure N.' in titles; no text annotations on figures.
Figures embedded as base64 in the final HTML manuscript.
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

# --- Style ---
SINGLE = (6.5, 4.8)
WIDE = (10.5, 4.8)
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

OUTPUT_DIR = "results/paper1_final/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


# ============================================================
# FIGURE 1: Aggregate INT–BE association
# ============================================================
def generate_figure1():
    scores = pd.read_csv("results/02_measurement/construct_scores.csv", index_col="respondent")
    INT = scores["INT"].values
    BE = scores["BE"].values

    fig, ax = plt.subplots(figsize=SINGLE)

    hb = ax.hexbin(INT, BE, gridsize=30, cmap='Blues', mincnt=1, alpha=0.8)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count')

    slope, intercept, r_val, p_val, se = stats.linregress(INT, BE)
    x_line = np.linspace(INT.min(), INT.max(), 100)
    ax.plot(x_line, intercept + slope * x_line, 'r-', lw=2, label=f'Linear fit: r = {r_val:.4f}')

    lim_min = min(INT.min(), BE.min())
    lim_max = max(INT.max(), BE.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1, alpha=0.5, label='Equality (INT = BE)')

    ax.set_xlabel('Intention (INT) mean score (1–5)')
    ax.set_ylabel('Behaviour (BE) mean score (1–5)')
    ax.set_title('Aggregate Intention–Behaviour Association')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(lim_min - 0.1, lim_max + 0.1)
    ax.set_ylim(lim_min - 0.1, lim_max + 0.1)

    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 2: GAP distribution
# ============================================================
def generate_figure2():
    gap_stats = pd.read_csv("results/paper1_final/03_intention_behavior_gap/gap_statistics.csv")
    gap_sd = gap_stats.loc[gap_stats['statistic'] == 'SD', 'value'].values[0]

    scores = pd.read_csv("results/02_measurement/construct_scores.csv", index_col="respondent")
    z_INT = (scores["INT"] - scores["INT"].mean()) / scores["INT"].std(ddof=1)
    z_BE = (scores["BE"] - scores["BE"].mean()) / scores["BE"].std(ddof=1)
    GAP = z_INT - z_BE

    fig, ax = plt.subplots(figsize=SINGLE)

    counts, bins, patches = ax.hist(GAP, bins=40, density=True, alpha=0.6,
                                     color='#4C72B0', edgecolor='white', linewidth=0.5)

    from scipy.stats import gaussian_kde, norm
    kde = gaussian_kde(GAP)
    x_grid = np.linspace(GAP.min(), GAP.max(), 200)
    ax.plot(x_grid, kde(x_grid), 'k-', lw=1.5, label='KDE')
    ax.plot(x_grid, norm.pdf(x_grid, 0, gap_sd), 'r--', lw=1.5,
            label=f'Normal(0, {gap_sd:.4f})')

    ax.axvline(0, color='k', linestyle=':', lw=1)

    ax.set_xlabel('GAP = z(INT) − z(BE)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of the Standardized Intention–Behaviour Gap')
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 3: INT vs BE profile structure
# ============================================================
def generate_figure3():
    profiles = pd.read_csv("results/paper1_final/05_profiles/05_k6_profile_table.csv")

    fig, ax = plt.subplots(figsize=SINGLE)

    lim_min = min(profiles['INT_mean'].min(), profiles['BE_mean'].min()) - 0.1
    lim_max = max(profiles['INT_mean'].max(), profiles['BE_mean'].max()) + 0.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1, alpha=0.5, label='INT = BE')

    sizes = profiles['N'] * 2
    colors = ['#E74C3C' if row['INT_minus_BE'] > 0 else '#3498DB' for _, row in profiles.iterrows()]

    for idx, row in profiles.iterrows():
        ax.scatter(row['INT_mean'], row['BE_mean'],
                   s=sizes[idx], c=colors[idx], alpha=0.8, edgecolors='k', linewidth=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', markersize=10,
               label='INT > BE', alpha=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', markersize=10,
               label='BE > INT', alpha=0.8),
        Line2D([0], [0], color='k', linestyle='--', lw=1, alpha=0.5, label='Equality (INT = BE)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    ax.set_xlabel('Mean Intention (INT)')
    ax.set_ylabel('Mean Behaviour (BE)')
    ax.set_title('Six-Profile Solution: INT vs BE Means')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 4: Profile INT and BE means
# ============================================================
def generate_figure4():
    profiles = pd.read_csv("results/paper1_final/05_profiles/05_k6_profile_table.csv")

    fig, ax = plt.subplots(figsize=WIDE)

    x = np.arange(6)
    width = 0.35

    ax.bar(x - width/2, profiles['INT_mean'], width, label='Intention (INT)',
           color='#4C72B0', edgecolor='k', linewidth=0.5)
    ax.bar(x + width/2, profiles['BE_mean'], width, label='Behaviour (BE)',
           color='#DD8452', edgecolor='k', linewidth=0.5)

    ax.set_xlabel('Profile')
    ax.set_ylabel('Mean score (1–5)')
    ax.set_title('Profile-Specific Intention and Behaviour Means')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i}' for i in range(6)])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 5.5)

    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 5: BIC across K = 2–10
# ============================================================
def generate_figure5():
    k_ext = pd.read_csv("results/paper1_strengthening/k_extended_model_comparison.csv")
    cov_ext = pd.read_csv("results/paper1_strengthening/covariance_k_extended.csv")

    fig, ax = plt.subplots(figsize=SINGLE)

    full = k_ext[k_ext['covariance_type'] == 'full']
    ax.plot(full['K'], full['BIC'], 'o-', color='#4C72B0', lw=2, markersize=8, label='Full covariance')

    diag = cov_ext[cov_ext['covariance_type'] == 'diag']
    if len(diag) > 0:
        ax.plot(diag['K'], diag['BIC'], 's--', color='#DD8452', lw=2, markersize=8, label='Diagonal covariance')

    k7 = full[full['K'] == 7]
    ax.scatter(k7['K'], k7['BIC'], s=200, facecolors='none', edgecolors='#F39C12',
               linewidths=3, zorder=5, label='K=7 (lowest BIC)')

    k_degen = full[full['K'] >= 8]
    if len(k_degen) > 0:
        ax.axvspan(7.5, 10.5, alpha=0.1, color='red', label='K≥8: numerical degeneracy')

    ax.set_xlabel('Number of profiles (K)')
    ax.set_ylabel('BIC')
    ax.set_title('BIC Across K = 2–10 (Full Covariance)')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_xticks(range(2, 11))

    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 6: Covariance sensitivity
# ============================================================
def generate_figure6():
    cov_ext = pd.read_csv("results/paper1_strengthening/covariance_k_extended.csv")

    fig, ax = plt.subplots(figsize=SINGLE)

    full = cov_ext[cov_ext['covariance_type'] == 'full']
    diag = cov_ext[cov_ext['covariance_type'] == 'diag']

    ax.plot(full['K'], full['BIC'], 'o-', color='#4C72B0', lw=2, markersize=8, label='Full covariance')
    ax.plot(diag['K'], diag['BIC'], 's--', color='#DD8452', lw=2, markersize=8, label='Diagonal covariance')

    ax.set_xlabel('Number of profiles (K)')
    ax.set_ylabel('BIC')
    ax.set_title('Covariance Specification Sensitivity: BIC by K')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xticks(range(2, 11))

    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 7: Classification quality
# ============================================================
def generate_figure7():
    class_qual = pd.read_csv("results/paper1_final/05_profiles/k6_classification_quality.csv")

    def cq(name):
        return float(class_qual.loc[class_qual['statistic'] == name, 'value'].values[0])

    post = pd.read_csv("results/paper1_final/04_lpa/posterior_probabilities.csv")
    max_post = post[[c for c in post.columns if c.startswith('post_profile_')]].max(axis=1)

    fig, ax = plt.subplots(figsize=SINGLE)

    counts, bins, patches = ax.hist(max_post, bins=30, alpha=0.7, color='#4C72B0',
                                     edgecolor='white', linewidth=0.5)

    ax.axvline(0.7, color='#E74C3C', linestyle='--', lw=1.5, label='Threshold 0.70')
    ax.axvline(0.8, color='#F39C12', linestyle='--', lw=1.5, label='Threshold 0.80')
    ax.axvline(0.9, color='#27AE60', linestyle='--', lw=1.5, label='Threshold 0.90')

    mean_post = cq('mean_max_posterior')
    ax.axvline(mean_post, color='k', linestyle='-', lw=1.5, label=f'Mean = {mean_post:.4f}')

    ax.set_xlabel('Maximum posterior probability')
    ax.set_ylabel('Number of respondents')
    ax.set_title('Classification Quality: Maximum Posterior Distribution')
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 8: Profile stability
# ============================================================
def generate_figure8():
    config_stab = pd.read_csv("results/paper1_final/05_profiles/configuration_stability.csv")

    fig, ax = plt.subplots(figsize=SINGLE)

    x = np.arange(6)
    width = 0.35

    colors = ['#E74C3C' if p == 5 else ('#27AE60' if p >= 80 else '#F39C12')
              for p in config_stab['pct_INT_gt_BE']]

    bars = ax.bar(x, config_stab['pct_INT_gt_BE'], width,
                  color=colors, edgecolor='k', linewidth=0.5)

    ax.axhline(50, color='k', linestyle=':', lw=1.5, label='Chance (50%)')
    ax.axhline(80, color='gray', linestyle=':', lw=1, label='Stability threshold (80%)')

    ax.set_xlabel('Profile')
    ax.set_ylabel('% INT > BE across bootstraps')
    ax.set_title('Bootstrap Directional Stability (200 Replications)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i}' for i in range(6)])
    ax.set_ylim(0, 105)

    legend_elements = [
        Patch(facecolor='#27AE60', edgecolor='k', label='Stable (≥80%)'),
        Patch(facecolor='#F39C12', edgecolor='k', label='Moderate (50–80%)'),
        Patch(facecolor='#E74C3C', edgecolor='k', label='Near chance (<50%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 9: Duplicate-row sensitivity for K = 7 (N = 1166 vs N = 1124)
# ============================================================
def generate_figure9():
    # K=7 duplicate sensitivity from verification
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.8))

    width = 0.35

    # Panel A: Pearson r + GAP SD (both ~0.8)
    labels_a = ['Pearson r', 'GAP SD']
    frozen_a = np.array([0.651458, 0.834916])
    unique_a = np.array([0.654410, 0.832201])
    x = np.arange(len(labels_a))
    ax1.bar(x - width/2, frozen_a, width, label='Primary (N=1166)',
            color='#4C72B0', edgecolor='k', linewidth=0.5)
    ax1.bar(x + width/2, unique_a, width, label='Duplicates removed (N=1124)',
            color='#DD8452', edgecolor='k', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_a)

    # Panel B: Entropy + Mean max posterior
    labels_b = ['Entropy', 'Mean max post.']
    frozen_b = np.array([0.0991, 0.9105])
    unique_b = np.array([0.0436, 0.9708])
    x = np.arange(len(labels_b))
    ax2.bar(x - width/2, frozen_b, width, label='Primary (N=1166)',
            color='#4C72B0', edgecolor='k', linewidth=0.5)
    ax2.bar(x + width/2, unique_b, width, label='Duplicates removed (N=1124)',
            color='#DD8452', edgecolor='k', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_b)

    # Panel C: BIC + LL (large magnitude)
    labels_c = ['K=7 BIC', 'K=7 LL']
    frozen_c = np.array([1616.02, -663.25])
    unique_c = np.array([857.03, -284.51])
    x = np.arange(len(labels_c))
    ax3.bar(x - width/2, frozen_c, width, label='Primary (N=1166)',
            color='#4C72B0', edgecolor='k', linewidth=0.5)
    ax3.bar(x + width/2, unique_c, width, label='Duplicates removed (N=1124)',
            color='#DD8452', edgecolor='k', linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_c)
    ax3.legend(loc='upper right', framealpha=0.9)

    fig.suptitle('Model Fit Metrics K = 7', fontsize=12, fontweight='600', y=1.02)
    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 10: Cross-validation (K = 2…7)
# ============================================================
def generate_figure10():
    cv_agg = pd.read_csv("results/paper1_strengthening/cv5_aggregated_by_k.csv")

    fig, ax = plt.subplots(figsize=SINGLE)

    ax.errorbar(cv_agg['K'], cv_agg['mean_ll_test'], yerr=cv_agg['sd_ll_test'],
                fmt='o-', color='#4C72B0', lw=2, markersize=8, capsize=5,
                label='Mean held-out log-likelihood (±SD)')

    k6 = cv_agg[cv_agg['K'] == 6]
    ax.scatter(k6['K'], k6['mean_ll_test'], s=200, facecolors='none', edgecolors='#E74C3C',
               linewidths=3, zorder=5, label='K=6 (primary, highest mean LL)')

    k7 = cv_agg[cv_agg['K'] == 7]
    if len(k7) > 0:
        ax.scatter(k7['K'], k7['mean_ll_test'], s=200, facecolors='none',
                   edgecolors='#F39C12', linewidths=3, zorder=5,
                   label='K=7 (lower held-out LL, high variance)')

    ax.set_xlabel('Number of profiles (K)')
    ax.set_ylabel('Held-out log-likelihood')
    ax.set_title('Five-Fold Cross-Validation: Held-Out Log-Likelihood by K = 2…7')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xticks(range(2, 8))

    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    print("Generating all 10 figures...")

    figures = {}

    print("  Figure 1: Aggregate INT-BE association...")
    figures['fig1'] = generate_figure1()

    print("  Figure 2: GAP distribution...")
    figures['fig2'] = generate_figure2()

    print("  Figure 3: Profile INT vs BE structure...")
    figures['fig3'] = generate_figure3()

    print("  Figure 4: Profile INT/BE means...")
    figures['fig4'] = generate_figure4()

    print("  Figure 5: Extended K=2–10 BIC...")
    figures['fig5'] = generate_figure5()

    print("  Figure 6: Covariance sensitivity...")
    figures['fig6'] = generate_figure6()

    print("  Figure 7: Classification quality...")
    figures['fig7'] = generate_figure7()

    print("  Figure 8: Profile stability...")
    figures['fig8'] = generate_figure8()

    print("  Figure 9: Duplicate sensitivity...")
    figures['fig9'] = generate_figure9()

    print("  Figure 10: Cross-validation...")
    figures['fig10'] = generate_figure10()

    with open(os.path.join(OUTPUT_DIR, "figures_base64.json"), "w") as f:
        json.dump(figures, f)

    print(f"\nAll figures saved to {OUTPUT_DIR}/figures_base64.json")
    print("Done!")


if __name__ == "__main__":
    main()
