"""
Phase 06 — Profile Characterization and Gap-Profile Test
===========================================================
Project: Intention–Behaviour Gap in Household Energy-Saving Behaviour

Uses the selected LPA solution (K=6).
For each profile: N, percentage, mean/SD of INT, BE, and all other
constructs (ATT, CON, SNO, COVID, PU, PEU, PO, PRI).

Pairwise comparisons between profiles with:
- test statistic, p-value, effect size (Cohen's d), 95% CI.

No arbitrary "high"/"low" thresholds. Investigates whether profiles
naturally produce HIGH-INT/LOW-BE or LOW-INT/HIGH-BE patterns.
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORES_PATH = "results/02_measurement/construct_scores.csv"
PHASE04_DIR = "results/04_lpa_estimation"
PHASE05_DIR = "results/05_lpa_selection"
RESULTS_DIR = "results/06_profile_analysis"
SEED = 42

# All constructs for profile characterization
ALL_CONSTRUCTS = ["INT", "BE", "ATT", "CON", "SNO", "COVID", "PU", "PEU", "PO", "PRI"]


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def cohens_d(group1, group2):
    """
    Cohen's d = (M1 - M2) / pooled SD
    pooled SD = sqrt[((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2)]
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = group1.mean(), group2.mean()
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return float((m1 - m2) / pooled)


def mean_diff_ci(g1, g2, alpha=0.05):
    """95% CI for difference in means (Welch)."""
    n1, n2 = len(g1), len(g2)
    m1, m2 = g1.mean(), g2.mean()
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    if se == 0:
        return (0.0, 0.0)
    # Welch-Satterthwaite df
    num = (s1**2 / n1 + s2**2 / n2) ** 2
    den = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
    df = num / den if den > 0 else n1 + n2 - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    diff = m1 - m2
    return (float(diff - t_crit * se), float(diff + t_crit * se))


def main():
    ensure_dir(RESULTS_DIR)
    np.random.seed(SEED)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    N = scores.shape[0]

    # Read selected K
    selected = pd.read_csv(os.path.join(PHASE05_DIR, "selected_model.csv"))
    K = int(selected["selected_K"].values[0])

    # Read class assignments from Phase 04
    kdir = os.path.join(PHASE04_DIR, f"K_{K}")
    post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
    labels = post_df["assigned_class"].values

    # Attach labels to scores
    df = scores.copy()
    df["profile"] = labels

    # ------------------------------------------------------------------
    # 1. Profile sizes
    # ------------------------------------------------------------------
    sizes = df["profile"].value_counts().sort_index()
    profile_sizes = pd.DataFrame(
        {
            "profile": sizes.index,
            "N": sizes.values,
            "percentage": sizes.values / N * 100,
        }
    )
    profile_sizes.to_csv(
        os.path.join(RESULTS_DIR, "profile_sizes.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 2. Profile means and SDs for all constructs
    # ------------------------------------------------------------------
    profile_means_rows = []
    for p in range(K):
        sub = df[df["profile"] == p]
        row = {"profile": p, "N": len(sub)}
        for c in ALL_CONSTRUCTS:
            row[f"{c}_mean"] = float(sub[c].mean())
            row[f"{c}_SD"] = float(sub[c].std(ddof=1))
        profile_means_rows.append(row)
    profile_means_df = pd.DataFrame(profile_means_rows)
    profile_means_df.to_csv(
        os.path.join(RESULTS_DIR, "profile_means.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 3. Pairwise comparisons (Welch t-test + Cohen's d + 95% CI)
    # ------------------------------------------------------------------
    comp_rows = []
    effect_rows = []
    for c in ALL_CONSTRUCTS:
        for p1, p2 in itertools.combinations(range(K), 2):
            g1 = df[df["profile"] == p1][c]
            g2 = df[df["profile"] == p2][c]
            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
            d = cohens_d(g1, g2)
            ci_lo, ci_hi = mean_diff_ci(g1, g2)
            comp_rows.append(
                {
                    "construct": c,
                    "profile_1": p1,
                    "profile_2": p2,
                    "mean_1": float(g1.mean()),
                    "mean_2": float(g2.mean()),
                    "t_statistic": float(t_stat),
                    "df": float(len(g1) + len(g2) - 2),
                    "p_value": float(p_val),
                    "cohens_d": d,
                    "CI95_lower": ci_lo,
                    "CI95_upper": ci_hi,
                }
            )
            effect_rows.append(
                {
                    "construct": c,
                    "profile_1": p1,
                    "profile_2": p2,
                    "cohens_d": d,
                    "abs_cohens_d": abs(d) if not np.isnan(d) else np.nan,
                }
            )
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(
        os.path.join(RESULTS_DIR, "profile_comparisons.csv"), index=False
    )
    effect_df = pd.DataFrame(effect_rows)
    effect_df.to_csv(
        os.path.join(RESULTS_DIR, "profile_effect_sizes.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 4. Gap-profile check
    # Investigate whether any profile naturally shows HIGH INT / LOW BE
    # or LOW INT / HIGH BE using relative profile means.
    # ------------------------------------------------------------------
    gap_rows = []
    for p in range(K):
        sub = df[df["profile"] == p]
        int_mean = sub["INT"].mean()
        be_mean = sub["BE"].mean()
        gap = int_mean - be_mean
        # Standardized gap within profile
        z_int = (int_mean - df["INT"].mean()) / df["INT"].std(ddof=1)
        z_be = (be_mean - df["BE"].mean()) / df["BE"].std(ddof=1)
        std_gap = z_int - z_be
        gap_rows.append(
            {
                "profile": p,
                "N": len(sub),
                "INT_mean": float(int_mean),
                "BE_mean": float(be_mean),
                "raw_gap_INT_minus_BE": float(gap),
                "z_INT": float(z_int),
                "z_BE": float(z_be),
                "standardized_gap": float(std_gap),
                "pattern": (
                    "HIGH_INT_LOW_BE" if std_gap > 0.2
                    else ("LOW_INT_HIGH_BE" if std_gap < -0.2
                          else "BALANCED")
                ),
            }
        )
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(
        os.path.join(RESULTS_DIR, "gap_profile_check.csv"), index=False
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 06 — PROFILE CHARACTERIZATION COMPLETE")
    print("=" * 60)
    print(f"Selected K = {K}")
    print(f"\nProfile sizes:")
    print(profile_sizes.to_string(index=False))
    print(f"\nProfile means (INT, BE):")
    print(profile_means_df[["profile", "N", "INT_mean", "INT_SD", "BE_mean", "BE_SD"]].to_string(index=False))
    print(f"\nGap-profile check:")
    print(gap_df.to_string(index=False))

    print("\nKey outputs:")
    for fn in [
        "profile_sizes.csv",
        "profile_means.csv",
        "profile_comparisons.csv",
        "profile_effect_sizes.csv",
        "gap_profile_check.csv",
    ]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
