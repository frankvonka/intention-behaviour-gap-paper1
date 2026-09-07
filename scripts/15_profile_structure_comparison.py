"""
Phase 15 — Profile Structure Comparison (K=2 .. selected K)
=============================================================
Numerically compares the existing K solutions from Phase 04 (dynamically
read from model_fit.csv; previously hardcoded K=2..K=6).
Descriptive/computational only. No K selection, no labels, no
interpretation, no plots, no literature.
"""
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORES_PATH = "results/02_measurement/construct_scores.csv"
PHASE04_DIR = "results/04_lpa_estimation"
RESULTS_DIR = "results/15_profile_structure_comparison"

# K values read dynamically from the Phase 04 fit summary
# (previously hardcoded [2, 3, 4, 5, 6])
K_RANGE = sorted(
    pd.read_csv(os.path.join(PHASE04_DIR, "model_fit.csv"))["K"].unique().tolist()
)
CONSTRUCTS = ["ATT", "CON", "SNO", "COVID", "INT", "BE", "PU", "PEU", "PO", "PRI"]


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def load_labels(K):
    kdir = os.path.join(PHASE04_DIR, f"K_{K}")
    post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
    return post_df["assigned_class"].values


def profile_table(df, labels, K, N):
    """Per-profile N, pct, mean INT, mean BE, INT-BE."""
    rows = []
    for p in range(K):
        sub = df[labels == p]
        rows.append({
            "K": K,
            "profile": p,
            "N": int(len(sub)),
            "percentage": float(len(sub) / N * 100),
            "mean_INT": float(sub["INT"].mean()),
            "mean_BE": float(sub["BE"].mean()),
            "INT_minus_BE": float(sub["INT"].mean() - sub["BE"].mean()),
        })
    return rows


def main():
    ensure_dir(RESULTS_DIR)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    N = scores.shape[0]

    labels_by_k = {K: load_labels(K) for K in K_RANGE}

    # ==================================================================
    # 1. PROFILE STRUCTURE TABLE
    # ==================================================================
    struct_rows = []
    for K in K_RANGE:
        struct_rows.extend(profile_table(scores, labels_by_k[K], K, N))
    struct_df = pd.DataFrame(struct_rows)
    struct_df.to_csv(
        os.path.join(RESULTS_DIR, "all_k_profile_structure.csv"), index=False
    )

    # ==================================================================
    # 2. GAP CONFIGURATION
    # ==================================================================
    gap_cfg = struct_df[["K", "profile", "N", "mean_INT", "mean_BE", "INT_minus_BE"]].copy()
    gap_cfg["INT_gt_BE"] = gap_cfg["INT_minus_BE"] > 0
    gap_cfg["BE_gt_INT"] = gap_cfg["INT_minus_BE"] < 0
    gap_cfg.to_csv(
        os.path.join(RESULTS_DIR, "all_k_gap_configuration.csv"), index=False
    )

    # ==================================================================
    # 3. PROFILE SPLITTING (consecutive K comparison)
    # ==================================================================
    # Standardized profile mean vectors across the 10 constructs.
    # Standardization: across respondents (grand mean/SD per construct),
    # so profile means are comparable across K solutions.
    grand_mean = scores[CONSTRUCTS].mean()
    grand_sd = scores[CONSTRUCTS].std(ddof=1)

    def std_profile_means(labels, K):
        rows = []
        for p in range(K):
            sub = scores[labels == p][CONSTRUCTS]
            rows.append(((sub.mean() - grand_mean) / grand_sd).values)
        return np.array(rows)  # (K, 10)

    split_rows = []
    # Consecutive-K pairs derived dynamically from K_RANGE
    # (previously hardcoded [(2, 3), (3, 4), (4, 5), (5, 6)])
    consecutive_pairs = list(zip(K_RANGE[:-1], K_RANGE[1:]))
    for K_low, K_high in consecutive_pairs:
        means_low = std_profile_means(labels_by_k[K_low], K_low)
        means_high = std_profile_means(labels_by_k[K_high], K_high)
        # For each high-K profile, nearest low-K profile (Euclidean)
        nearest = []
        dists = []
        for j in range(K_high):
            d = np.linalg.norm(means_high[j] - means_low, axis=1)
            nearest.append(int(np.argmin(d)))
            dists.append(float(d.min()))
        # Group high-K profiles by their nearest low-K profile
        for i in range(K_low):
            children = [j for j in range(K_high) if nearest[j] == i]
            split_rows.append({
                "K_low": K_low,
                "K_high": K_high,
                "low_profile": i,
                "low_profile_N": int((labels_by_k[K_low] == i).sum()),
                "n_high_profiles_matched": len(children),
                "high_profiles": str(children),
                "split": bool(len(children) > 1),
                "mean_euclidean_distance_to_children": float(np.mean([dists[j] for j in children])) if children else np.nan,
            })
    pd.DataFrame(split_rows).to_csv(
        os.path.join(RESULTS_DIR, "profile_split_comparison.csv"), index=False
    )

    # ==================================================================
    # 4. INT-BE INFORMATION ADDED BY EACH K
    # ==================================================================
    info_rows = []
    for K in K_RANGE:
        sub = struct_df[struct_df["K"] == K]
        diffs = sub["INT_minus_BE"].values
        info_rows.append({
            "K": K,
            "n_profiles": K,
            "n_INT_gt_BE": int((diffs > 0).sum()),
            "n_BE_gt_INT": int((diffs < 0).sum()),
            "min_INT_minus_BE": float(diffs.min()),
            "max_INT_minus_BE": float(diffs.max()),
            "range_INT_minus_BE": float(diffs.max() - diffs.min()),
            "variance_INT_minus_BE_means": float(diffs.var(ddof=1)),
        })
    pd.DataFrame(info_rows).to_csv(
        os.path.join(RESULTS_DIR, "gap_information_by_k.csv"), index=False
    )

    # ==================================================================
    # 5. PROFILE SIZE DIAGNOSTICS
    # ==================================================================
    size_rows = []
    for K in K_RANGE:
        sub = struct_df[struct_df["K"] == K]
        size_rows.append({
            "K": K,
            "min_N": int(sub["N"].min()),
            "max_N": int(sub["N"].max()),
            "min_percentage": float(sub["percentage"].min()),
            "max_percentage": float(sub["percentage"].max()),
        })
    pd.DataFrame(size_rows).to_csv(
        os.path.join(RESULTS_DIR, "profile_size_by_k.csv"), index=False
    )

    # ==================================================================
    # 6. BOUNDARY DIAGNOSTICS
    # ==================================================================
    bound_rows = []
    for K in K_RANGE:
        labels = labels_by_k[K]
        for p in range(K):
            sub = scores[labels == p]
            int_p = sub["INT"].values
            be_p = sub["BE"].values
            bound_rows.append({
                "K": K,
                "profile": p,
                "N": int(len(sub)),
                "variance_INT": float(int_p.var(ddof=1)),
                "variance_BE": float(be_p.var(ddof=1)),
                "min_INT": float(int_p.min()),
                "max_INT": float(int_p.max()),
                "min_BE": float(be_p.min()),
                "max_BE": float(be_p.max()),
                "zero_variance_INT": bool(int_p.var(ddof=1) == 0.0),
                "zero_variance_BE": bool(be_p.var(ddof=1) == 0.0),
            })
    pd.DataFrame(bound_rows).to_csv(
        os.path.join(RESULTS_DIR, "boundary_by_k.csv"), index=False
    )

    # ==================================================================
    # 7. MODEL FIT SUMMARY (existing verified results)
    # ==================================================================
    fit = pd.read_csv(os.path.join(PHASE04_DIR, "model_fit.csv"))
    fit_summary = fit[["K", "log_likelihood", "AIC", "BIC", "entropy",
                       "n_params", "smallest_class_pct", "largest_class_pct",
                       "converged", "n_iter"]].copy()
    # Attach profile sizes
    sizes_str = []
    for K in K_RANGE:
        sizes = pd.read_csv(os.path.join(PHASE04_DIR, f"K_{K}", "profile_sizes.csv"))
        sizes_str.append(str(sizes["size"].tolist()))
    fit_summary["profile_sizes"] = sizes_str
    fit_summary.to_csv(
        os.path.join(RESULTS_DIR, "model_fit_summary.csv"), index=False
    )

    # ==================================================================
    # 8. MASTER FILE
    # ==================================================================
    master_rows = []
    for K in K_RANGE:
        sub = struct_df[struct_df["K"] == K]
        frow = fit[fit["K"] == K].iloc[0]
        master_rows.append({
            "K": K,
            "n_profiles": K,
            "BIC": float(frow["BIC"]),
            "AIC": float(frow["AIC"]),
            "entropy": float(frow["entropy"]),
            "min_profile_N": int(sub["N"].min()),
            "max_profile_N": int(sub["N"].max()),
            "n_INT_gt_BE": int((sub["INT_minus_BE"] > 0).sum()),
            "n_BE_gt_INT": int((sub["INT_minus_BE"] < 0).sum()),
            "range_INT_minus_BE": float(sub["INT_minus_BE"].max() - sub["INT_minus_BE"].min()),
        })
    pd.DataFrame(master_rows).to_csv(
        os.path.join(RESULTS_DIR, "phase15_master.csv"), index=False
    )

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = f"""# Phase 15 — Profile Structure Comparison (K={K_RANGE[0]} to K={K_RANGE[-1]})

## Purpose
Numerically compare the existing K={K_RANGE[0]}..K={K_RANGE[-1]} solutions
(dynamically read from Phase 04 model_fit.csv). Descriptive and
computational only. No K selection, no profile labels, no interpretation.

## Input Result Files
- results/02_measurement/construct_scores.csv (construct scores)
- results/04_lpa_estimation/K_{K}/posterior_probabilities.csv (class assignments)
- results/04_lpa_estimation/K_{K}/profile_sizes.csv (profile sizes)
- results/04_lpa_estimation/model_fit.csv (fit statistics)

## Calculations
1. **all_k_profile_structure.csv** — per-K, per-profile N, %, mean INT, mean BE, INT-BE
2. **all_k_gap_configuration.csv** — signed INT-BE differences with direction flags
3. **profile_split_comparison.csv** — consecutive-K splitting via nearest
   standardized profile mean vector (Euclidean distance, 10 constructs)
4. **gap_information_by_k.csv** — counts of INT>BE / BE>INT profiles, range
   and variance of profile-level INT-BE means
5. **profile_size_by_k.csv** — min/max N and percentage per K
6. **boundary_by_k.csv** — per-profile INT/BE variance, min, max; zero-variance flags
7. **model_fit_summary.csv** — LL, AIC, BIC, entropy, n_params, sizes (existing results)
8. **phase15_master.csv** — key per-K summary

## Distance Metric and Standardization
- Profile mean vectors computed across the 10 constructs
  (ATT, CON, SNO, COVID, INT, BE, PU, PEU, PO, PRI)
- Standardized using respondent-level grand mean and SD per construct:
  z = (profile_mean - grand_mean) / grand_SD
- Euclidean distance between standardized vectors
- Nearest-neighbor assignment (argmin distance); no manual assignment

## No Data Modification / No Model-Selection Change
All numbers read from existing verified result files. No refitting.
No dataset changes. No plots. No literature.
"""
    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write(readme)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 15 — PROFILE STRUCTURE COMPARISON COMPLETE")
    print("=" * 60)
    print("\nProfile structure (all K):")
    print(struct_df.to_string(index=False))
    print("\nGap information by K:")
    print(pd.DataFrame(info_rows).to_string(index=False))
    print("\nProfile splitting:")
    print(pd.DataFrame(split_rows)[["K_low", "K_high", "low_profile", "n_high_profiles_matched", "high_profiles", "split"]].to_string(index=False))
    print("\nOutputs:")
    for fn in ["all_k_profile_structure.csv", "all_k_gap_configuration.csv",
               "profile_split_comparison.csv", "gap_information_by_k.csv",
               "profile_size_by_k.csv", "boundary_by_k.csv",
               "model_fit_summary.csv", "phase15_master.csv", "README.md"]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
