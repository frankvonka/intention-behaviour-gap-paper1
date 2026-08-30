"""
Phase 13 — K=6 Profile Validation
====================================
Validates the K=6 profile solution using existing posterior assignments
and construct scores. No refitting, no literature, no interpretation.
"""
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORES_PATH = "results/02_measurement/construct_scores.csv"
PHASE04_DIR = "results/04_lpa_estimation"
PHASE03_DIR = "results/03_gap_analysis"
RESULTS_DIR = "results/13_k6_profile_validation"
SEED = 42
K = 6

CONSTRUCTS = ["ATT", "CON", "SNO", "COVID", "INT", "BE", "PU", "PEU", "PO", "PRI"]


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = g1.mean(), g2.mean()
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return float((m1 - m2) / pooled)


def main():
    ensure_dir(RESULTS_DIR)
    np.random.seed(SEED)

    # ------------------------------------------------------------------
    # Load existing data
    # ------------------------------------------------------------------
    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    N = scores.shape[0]

    # K=6 posterior assignments
    kdir = os.path.join(PHASE04_DIR, f"K_{K}")
    post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
    labels = post_df["assigned_class"].values
    post_cols = [c for c in post_df.columns if c.startswith("post_profile_")]
    posteriors = post_df[post_cols].values

    # Gap scores
    gap_scores = np.load(os.path.join(PHASE03_DIR, "gap_scores.npy"))

    # Attach to dataframe
    df = scores.copy()
    df["profile"] = labels
    df["gap"] = gap_scores

    # ------------------------------------------------------------------
    # 1. K=6 PROFILE DESCRIPTIVES
    # ------------------------------------------------------------------
    desc_rows = []
    for p in range(K):
        sub = df[df["profile"] == p]
        row = {"profile": p, "N": int(len(sub)), "percentage": float(len(sub) / N * 100)}
        for c in CONSTRUCTS:
            s = sub[c]
            q75, q25 = np.nanpercentile(s, [75, 25])
            row[f"{c}_mean"] = float(s.mean())
            row[f"{c}_SD"] = float(s.std(ddof=1))
            row[f"{c}_median"] = float(s.median())
            row[f"{c}_IQR"] = float(q75 - q25)
        # INT, BE, GAP summary
        row["INT_mean"] = float(sub["INT"].mean())
        row["BE_mean"] = float(sub["BE"].mean())
        row["INT_minus_BE"] = float(sub["INT"].mean() - sub["BE"].mean())
        row["GAP_mean"] = float(sub["gap"].mean())
        desc_rows.append(row)
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(
        os.path.join(RESULTS_DIR, "k6_profile_descriptives.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 2. PROFILE SEPARATION (Cohen's d)
    # ------------------------------------------------------------------
    sep_rows = []
    for metric in ["INT", "BE", "gap"]:
        for p1, p2 in [(i, j) for i in range(K) for j in range(i + 1, K)]:
            g1 = df[df["profile"] == p1][metric]
            g2 = df[df["profile"] == p2][metric]
            d = cohens_d(g1, g2)
            sep_rows.append({
                "metric": metric,
                "profile_1": p1,
                "profile_2": p2,
                "n_1": int(len(g1)),
                "n_2": int(len(g2)),
                "mean_1": float(g1.mean()),
                "mean_2": float(g2.mean()),
                "SD_1": float(g1.std(ddof=1)),
                "SD_2": float(g2.std(ddof=1)),
                "Cohens_d": d,
            })
    pd.DataFrame(sep_rows).to_csv(
        os.path.join(RESULTS_DIR, "pairwise_profile_separation.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 3. INT-BE CONFIGURATION
    # ------------------------------------------------------------------
    cfg_rows = []
    for p in range(K):
        sub = df[df["profile"] == p]
        int_mean = float(sub["INT"].mean())
        be_mean = float(sub["BE"].mean())
        diff = int_mean - be_mean
        # Determine direction numerically (no arbitrary threshold for "approximate")
        direction = "INT_gt_BE" if diff > 0 else ("BE_gt_INT" if diff < 0 else "INT_eq_BE")
        cfg_rows.append({
            "profile": p,
            "N": int(len(sub)),
            "mean_INT": int_mean,
            "mean_BE": be_mean,
            "INT_minus_BE": diff,
            "mean_GAP": float(sub["gap"].mean()),
            "INT_gt_BE": bool(diff > 0),
            "BE_gt_INT": bool(diff < 0),
            "INT_eq_BE": bool(diff == 0),
            "direction": direction,
        })
    pd.DataFrame(cfg_rows).to_csv(
        os.path.join(RESULTS_DIR, "k6_int_be_configuration.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 4. POSTERIOR CLASSIFICATION QUALITY
    # ------------------------------------------------------------------
    max_post = posteriors.max(axis=1)
    cls_rows = [
        {"statistic": "mean_max_posterior", "value": float(max_post.mean())},
        {"statistic": "median_max_posterior", "value": float(np.median(max_post))},
        {"statistic": "SD_max_posterior", "value": float(max_post.std(ddof=1))},
        {"statistic": "min_max_posterior", "value": float(max_post.min())},
        {"statistic": "max_max_posterior", "value": float(max_post.max())},
        {"statistic": "n_maxpost_ge_0.90", "value": int((max_post >= 0.90).sum())},
        {"statistic": "n_maxpost_ge_0.80", "value": int((max_post >= 0.80).sum())},
        {"statistic": "n_maxpost_ge_0.70", "value": int((max_post >= 0.70).sum())},
        {"statistic": "n_maxpost_lt_0.70", "value": int((max_post < 0.70).sum())},
        {"statistic": "n_maxpost_lt_0.50", "value": int((max_post < 0.50).sum())},
        {"statistic": "pct_maxpost_ge_0.90", "value": float((max_post >= 0.90).mean() * 100)},
        {"statistic": "pct_maxpost_ge_0.80", "value": float((max_post >= 0.80).mean() * 100)},
        {"statistic": "pct_maxpost_ge_0.70", "value": float((max_post >= 0.70).mean() * 100)},
        {"statistic": "pct_maxpost_lt_0.70", "value": float((max_post < 0.70).mean() * 100)},
        {"statistic": "pct_maxpost_lt_0.50", "value": float((max_post < 0.50).mean() * 100)},
    ]
    pd.DataFrame(cls_rows).to_csv(
        os.path.join(RESULTS_DIR, "k6_classification_quality.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 5. PROFILE ASSIGNMENT UNCERTAINTY
    # ------------------------------------------------------------------
    unc_rows = []
    max_post_series = pd.Series(max_post, index=df.index)
    for p in range(K):
        mask = df["profile"] == p
        mp_profile = max_post_series[mask.values]
        unc_rows.append({
            "profile": p,
            "N": int(mask.sum()),
            "mean_max_posterior": float(mp_profile.mean()),
            "median_max_posterior": float(mp_profile.median()),
            "SD_max_posterior": float(mp_profile.std(ddof=1)),
            "pct_maxpost_lt_0.70": float((mp_profile < 0.70).mean() * 100),
            "pct_maxpost_lt_0.50": float((mp_profile < 0.50).mean() * 100),
        })
    pd.DataFrame(unc_rows).to_csv(
        os.path.join(RESULTS_DIR, "k6_profile_uncertainty.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 6. PROFILE SIZE
    # ------------------------------------------------------------------
    sizes = df["profile"].value_counts().sort_index()
    sizes_df = pd.DataFrame({
        "profile": sizes.index,
        "N": sizes.values,
        "percentage": (sizes.values / N * 100),
    })
    sizes_df.to_csv(
        os.path.join(RESULTS_DIR, "k6_profile_sizes.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 7. MASTER PACKAGE
    # ------------------------------------------------------------------
    master_rows = []
    for p in range(K):
        d = desc_df[desc_df["profile"] == p].iloc[0]
        master_rows.append({
            "profile": p,
            "N": int(d["N"]),
            "percentage": float(d["percentage"]),
            "mean_INT": float(d["INT_mean"]),
            "mean_BE": float(d["BE_mean"]),
            "INT_minus_BE": float(d["INT_minus_BE"]),
            "mean_GAP": float(d["GAP_mean"]),
        })
    pd.DataFrame(master_rows).to_csv(
        os.path.join(RESULTS_DIR, "k6_validation_master.csv"), index=False
    )

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = """# Phase 13 — K=6 Profile Validation

## Purpose
Produce numerical evidence for later evaluation of the K=6 profile solution.
No refitting. No literature. No interpretation. No plots.

## Input Files
- results/02_measurement/construct_scores.csv
- results/04_lpa_estimation/K_6/posterior_probabilities.csv
- results/03_gap_analysis/gap_scores.npy

## Calculations
1. **k6_profile_descriptives.csv** — per-profile N, %, mean/SD/median/IQR for all 10 constructs + INT, BE, INT-BE, GAP
2. **pairwise_profile_separation.csv** — Cohen's d for all profile pairs on INT, BE, GAP
3. **k6_int_be_configuration.csv** — signed INT-BE differences per profile
4. **k6_classification_quality.csv** — distribution of max posterior probabilities
5. **k6_profile_uncertainty.csv** — uncertainty statistics per profile
6. **k6_profile_sizes.csv** — simple N and % table
7. **k6_validation_master.csv** — master summary

## Methods
- Construct scores: arithmetic mean of items (from Phase 02)
- GAP: z_INT - z_BE (from Phase 03)
- Cohen's d: (M1-M2) / pooled SD with pooled SD = sqrt[((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2)]
- IQR: Q75 - Q25

## No Data Modifications
All calculations read existing files. No new fitting. No dataset changes.
"""
    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write(readme)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 13 — K=6 PROFILE VALIDATION COMPLETE")
    print("=" * 60)
    print(f"\nInputs: N={N}, K={K}")
    for fn in [
        "k6_profile_descriptives.csv",
        "pairwise_profile_separation.csv",
        "k6_int_be_configuration.csv",
        "k6_classification_quality.csv",
        "k6_profile_uncertainty.csv",
        "k6_profile_sizes.csv",
        "k6_validation_master.csv",
        "README.md",
    ]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
