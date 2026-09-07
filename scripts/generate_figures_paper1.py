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

# --- Dynamic solution constants (read from frozen evidence, no hardcoded K/N) ---
SEL = int(pd.read_csv(
    "results/05_lpa_selection/selected_model.csv")["selected_K"].iloc[0])
_kext_all = pd.read_csv(
    "results/paper1_strengthening/k_extended_model_comparison.csv")
_kext_full = _kext_all[_kext_all.covariance_type == 'full']
_KMIN_EXT, _KMAX_EXT = int(_kext_full.K.min()), int(_kext_full.K.max())
_wb = _kext_full[_kext_full.log_likelihood < 0]   # numerically well-behaved fits
_deg_ks = sorted(int(k) for k in _kext_full.loc[_kext_full.log_likelihood > 0, "K"])
DEG_K_MIN = min(_deg_ks) if _deg_ks else None
N_TOTAL = int(pd.read_csv(
    "results/01_data_inspection/data_dimensions.csv")["N_rows"].iloc[0])
_dupinfo = pd.read_csv("results/01_data_inspection/duplicate_information.csv")
N_DUPS = int(_dupinfo.loc[_dupinfo["check"] == "all_columns",
                          "n_duplicates"].iloc[0])
N_UNIQ = N_TOTAL - N_DUPS
_bootm = pd.read_csv("results/14_k6_stability/k6_stability_master.csv")
N_BOOT = int(dict(zip(_bootm.metric, _bootm.value))["n_bootstrap"])


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
# FIGURE 3: INT vs BE profile structure (primary K)
# ============================================================
def generate_figure3():
    profiles = pd.read_csv(
        f"results/paper1_strengthening/task1_K_{SEL}/profile_parameters.csv")

    fig, ax = plt.subplots(figsize=SINGLE)

    lim_min = min(profiles['mean_z_INT'].min(), profiles['mean_z_BE'].min()) - 0.1
    lim_max = max(profiles['mean_z_INT'].max(), profiles['mean_z_BE'].max()) + 0.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1, alpha=0.5, label='z(INT) = z(BE)')

    sizes = profiles['size'] * 2
    colors = ['#E74C3C' if row['mean_z_INT'] > row['mean_z_BE'] else '#3498DB' for _, row in profiles.iterrows()]

    for idx, row in profiles.iterrows():
        ax.scatter(row['mean_z_INT'], row['mean_z_BE'],
                   s=sizes[idx], c=colors[idx], alpha=0.8, edgecolors='k', linewidth=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', markersize=10,
               label='z(INT) > z(BE)', alpha=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', markersize=10,
               label='z(BE) > z(INT)', alpha=0.8),
        Line2D([0], [0], color='k', linestyle='--', lw=1, alpha=0.5, label='Equality (z(INT) = z(BE))'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    ax.set_xlabel('Mean z(Intention)')
    ax.set_ylabel('Mean z(Behaviour)')
    ax.set_title(f'{SEL}-Profile Solution (K = {SEL}): INT vs BE Means')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 4: Profile INT and BE means (primary K)
# ============================================================
def generate_figure4():
    profiles = pd.read_csv(
        f"results/paper1_strengthening/task1_K_{SEL}/profile_parameters.csv")

    fig, ax = plt.subplots(figsize=WIDE)

    x = np.arange(SEL)
    width = 0.35

    ax.bar(x - width/2, profiles['mean_z_INT'], width, label='z(Intention)',
           color='#4C72B0', edgecolor='k', linewidth=0.5)
    ax.bar(x + width/2, profiles['mean_z_BE'], width, label='z(Behaviour)',
           color='#DD8452', edgecolor='k', linewidth=0.5)

    ax.set_xlabel('Profile')
    ax.set_ylabel('Mean z-score')
    ax.set_title(f'Profile-Specific z(Intention) and z(Behaviour) Means (K = {SEL})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i}' for i in range(SEL)])
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 5: BIC across the extended K range
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

    k_sel_row = full[full['K'] == SEL]
    ax.scatter(k_sel_row['K'], k_sel_row['BIC'], s=200, facecolors='none', edgecolors='#F39C12',
               linewidths=3, zorder=5, label=f'K={SEL} (lowest well-behaved BIC)')

    if DEG_K_MIN is not None:
        ax.axvspan(DEG_K_MIN - 0.5, _KMAX_EXT + 0.5, alpha=0.1, color='red',
                   label=f'K>={DEG_K_MIN}: numerical degeneracy')

    ax.set_xlabel('Number of profiles (K)')
    ax.set_ylabel('BIC')
    ax.set_title(f'BIC Across K = {_KMIN_EXT}–{_KMAX_EXT} (Full Covariance)')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_xticks(range(_KMIN_EXT, _KMAX_EXT + 1))

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
    ax.set_xticks(range(int(cov_ext['K'].min()), int(cov_ext['K'].max()) + 1))

    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 7: Classification quality (primary K)
# ============================================================
def generate_figure7():
    class_qual = pd.read_csv(
        f"results/paper1_strengthening/k{SEL}_classification_quality.csv")

    def cq(name):
        return float(class_qual.loc[class_qual['statistic'] == name, 'value'].values[0])

    post = pd.read_csv(
        f"results/paper1_strengthening/k{SEL}_posterior_probabilities.csv")
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
    ax.set_title(f'Classification Quality: Maximum Posterior Distribution (K = {SEL})')
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 8: Profile stability (primary K)
# ============================================================
def generate_figure8():
    config_stab = pd.read_csv(
        f"results/paper1_strengthening/k{SEL}_stability.csv")

    fig, ax = plt.subplots(figsize=SINGLE)

    x = np.arange(SEL)
    width = 0.35

    # Colour by the observed directional-consistency value (legend thresholds)
    colors = ['#27AE60' if p >= 80 else ('#E74C3C' if p < 50 else '#F39C12')
              for p in config_stab['pct_INT_gt_BE']]

    bars = ax.bar(x, config_stab['pct_INT_gt_BE'], width,
                  color=colors, edgecolor='k', linewidth=0.5)

    ax.axhline(50, color='k', linestyle=':', lw=1.5, label='Chance (50%)')
    ax.axhline(80, color='gray', linestyle=':', lw=1, label='Stability threshold (80%)')

    ax.set_xlabel('Profile')
    ax.set_ylabel('% INT > BE across bootstraps')
    ax.set_title(f'Bootstrap Directional Stability ({N_BOOT} Replications, K = {SEL})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i}' for i in range(SEL)])
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
# FIGURE 9: Duplicate-row sensitivity for the primary K
# ============================================================
def generate_figure9():
    # Duplicate sensitivity — read both columns from the primary-K sensitivity file
    ct = pd.read_csv(
        f"results/k{SEL}_primary/k{SEL}_duplicate_sensitivity.csv").set_index("metric")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.8))

    width = 0.35
    lab_prim = f'Primary (N={N_TOTAL})'
    lab_uniq = f'Duplicates removed (N={N_UNIQ})'

    # Panel A: Pearson r + GAP SD
    labels_a = ['Pearson r', 'GAP SD']
    frozen_a = np.array([float(ct.loc["pearson_r_INT_BE", "frozen"]),
                         float(ct.loc["gap_SD", "frozen"])])
    unique_a = np.array([float(ct.loc["pearson_r_INT_BE", "unique_sample"]),
                         float(ct.loc["gap_SD", "unique_sample"])])
    x = np.arange(len(labels_a))
    ax1.bar(x - width/2, frozen_a, width, label=lab_prim,
            color='#4C72B0', edgecolor='k', linewidth=0.5)
    ax1.bar(x + width/2, unique_a, width, label=lab_uniq,
            color='#DD8452', edgecolor='k', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_a)

    # Panel B: Entropy + Mean max posterior
    labels_b = ['Entropy', 'Mean max post.']
    frozen_b = np.array([float(ct.loc[f"K{SEL}_entropy", "frozen"]),
                         float(ct.loc[f"K{SEL}_mean_max_posterior", "frozen"])])
    unique_b = np.array([float(ct.loc[f"K{SEL}_entropy", "unique_sample"]),
                         float(ct.loc[f"K{SEL}_mean_max_posterior", "unique_sample"])])
    x = np.arange(len(labels_b))
    ax2.bar(x - width/2, frozen_b, width, label=lab_prim,
            color='#4C72B0', edgecolor='k', linewidth=0.5)
    ax2.bar(x + width/2, unique_b, width, label=lab_uniq,
            color='#DD8452', edgecolor='k', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_b)

    # Panel C: BIC + LL (large magnitude)
    labels_c = [f'K={SEL} BIC', f'K={SEL} LL']
    frozen_c = np.array([float(ct.loc[f"K{SEL}_BIC", "frozen"]),
                         float(ct.loc[f"K{SEL}_log_likelihood", "frozen"])])
    unique_c = np.array([float(ct.loc[f"K{SEL}_BIC", "unique_sample"]),
                         float(ct.loc[f"K{SEL}_log_likelihood", "unique_sample"])])
    x = np.arange(len(labels_c))
    ax3.bar(x - width/2, frozen_c, width, label=lab_prim,
            color='#4C72B0', edgecolor='k', linewidth=0.5)
    ax3.bar(x + width/2, unique_c, width, label=lab_uniq,
            color='#DD8452', edgecolor='k', linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_c)
    ax3.legend(loc='upper right', framealpha=0.9)

    fig.suptitle(f'Model Fit Metrics K = {SEL}', fontsize=12, fontweight='600', y=1.02)
    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# FIGURE 10: Cross-validation (K = 2…7)
# ============================================================
def generate_figure10():
    cv_agg = pd.read_csv("results/paper1_strengthening/cv5_aggregated_by_k.csv")
    cv_agg['K'] = cv_agg['K'].astype(int)
    sel_k = int(pd.read_csv("results/05_lpa_selection/selected_model.csv")["selected_K"].iloc[0])
    kmin = int(cv_agg['K'].min())
    kmax = int(cv_agg['K'].max())

    fig, ax = plt.subplots(figsize=SINGLE)

    ax.errorbar(cv_agg['K'], cv_agg['mean_ll_test'], yerr=cv_agg['sd_ll_test'],
                fmt='o-', color='#4C72B0', lw=2, markersize=8, capsize=5,
                label='Mean held-out log-likelihood (±SD)')

    # Circle the BIC-selected primary K and, if different, the best held-out LL K
    k_sel = cv_agg[cv_agg['K'] == sel_k]
    ax.scatter(k_sel['K'], k_sel['mean_ll_test'], s=200, facecolors='none', edgecolors='#E74C3C',
               linewidths=3, zorder=5, label=f'K={sel_k} (primary, BIC-selected)')

    best_k = int(cv_agg.loc[cv_agg['mean_ll_test'].idxmax(), 'K'])
    if best_k != sel_k:
        k_best = cv_agg[cv_agg['K'] == best_k]
        ax.scatter(k_best['K'], k_best['mean_ll_test'], s=200, facecolors='none',
                   edgecolors='#F39C12', linewidths=3, zorder=5,
                   label=f'K={best_k} (highest mean held-out LL)')
    else:
        ax.scatter([], [], s=0, label=f'K={sel_k} also has the highest mean held-out LL')

    ax.set_xlabel('Number of profiles (K)')
    ax.set_ylabel('Held-out log-likelihood')
    ax.set_title(f'Five-Fold Cross-Validation: Held-Out Log-Likelihood by K = {kmin}…{kmax}')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xticks(range(int(cv_agg['K'].min()), kmax + 1))

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

    print(f"  Figure 5: Extended K={_KMIN_EXT}–{_KMAX_EXT} BIC...")
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
