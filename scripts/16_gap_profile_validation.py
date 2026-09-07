"""
Phase 16 — Intention–Behavior Gap Profile Validation
=====================================================
Determines whether the observed LPA profiles represent meaningful
INT–BE configurations or mainly reflect overall response level.

Uses existing K results from Phase 04 (K=2..selected K, read dynamically
from model_fit.csv; previously hardcoded K=2..K=6). Performs a diagnostic
gap-only Gaussian mixture model comparison. No interpretation.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORES_PATH = "results/02_measurement/construct_scores.csv"
PHASE04_DIR = "results/04_lpa_estimation"
PHASE05_DIR = "results/05_lpa_selection"
PHASE03_DIR = "results/03_gap_analysis"
RESULTS_DIR = "results/16_gap_profile_validation"

# K values read dynamically from the Phase 04 fit summary
# (previously hardcoded [2, 3, 4, 5, 6])
K_RANGE = sorted(
    pd.read_csv(os.path.join(PHASE04_DIR, "model_fit.csv"))["K"].unique().tolist()
)
# Selected primary K from the Phase 05 model selection
K_SEL = int(
    pd.read_csv(os.path.join(PHASE05_DIR, "selected_model.csv"))["selected_K"].iloc[0]
)
SEED = 42
N_INIT = 1000
COVARIANCE_TYPE = "full"  # for multivariate (same as primary); gap-only uses appropriate 1D spec


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def count_params(K, n_features, cov_type="full"):
    means = K * n_features
    if cov_type == "full":
        covs = K * n_features * (n_features + 1) / 2.0
    elif cov_type == "diag":
        covs = K * n_features
    elif cov_type == "spherical":
        covs = K
    else:
        covs = K * n_features
    weights = K - 1
    return int(means + covs + weights)


def safe_entropy(posterior):
    p = np.clip(posterior, 1e-15, 1.0)
    n, K = posterior.shape
    log_k = np.log(K)
    if log_k == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)) / (n * log_k))


def load_labels(K):
    kdir = os.path.join(PHASE04_DIR, f"K_{K}")
    post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
    return post_df["assigned_class"].values


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

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    N = scores.shape[0]

    # Standardized INT and BE (Phase 03 procedure)
    z_INT = (scores["INT"] - scores["INT"].mean()) / scores["INT"].std(ddof=1)
    z_BE = (scores["BE"] - scores["BE"].mean()) / scores["BE"].std(ddof=1)

    # Respondent-level GAP = z(INT) - z(BE)
    gap_scores = np.load(os.path.join(PHASE03_DIR, "gap_scores.npy"))
    GAP = gap_scores

    # LEVEL = (z_INT + z_BE) / 2
    LEVEL = (z_INT + z_BE) / 2.0

    # Labels for each K
    labels_by_k = {K: load_labels(K) for K in K_RANGE}

    # ==================================================================
    # 1. PROFILE MEANS ON INT AND BE
    # ==================================================================
    struct_rows = []
    for K in K_RANGE:
        labels = labels_by_k[K]
        for p in range(K):
            mask = labels == p
            sub = scores[mask]
            struct_rows.append({
                "K": K,
                "profile": p,
                "N": int(mask.sum()),
                "percentage": float(mask.sum() / N * 100),
                "INT_mean": float(sub["INT"].mean()),
                "BE_mean": float(sub["BE"].mean()),
                "INT_minus_BE": float(sub["INT"].mean() - sub["BE"].mean()),
            })
    pd.DataFrame(struct_rows).to_csv(
        os.path.join(RESULTS_DIR, "profile_int_be_by_k.csv"), index=False
    )

    # ==================================================================
    # 2. WITHIN-PROFILE GAP VARIABILITY
    # ==================================================================
    gap_var_rows = []
    for K in K_RANGE:
        labels = labels_by_k[K]
        for p in range(K):
            mask = labels == p
            gap_p = GAP[mask]
            gap_var_rows.append({
                "K": K,
                "profile": p,
                "N": int(mask.sum()),
                "mean_GAP": float(gap_p.mean()),
                "SD_GAP": float(gap_p.std(ddof=1)),
                "median_GAP": float(np.median(gap_p)),
                "min_GAP": float(gap_p.min()),
                "max_GAP": float(gap_p.max()),
            })
    pd.DataFrame(gap_var_rows).to_csv(
        os.path.join(RESULTS_DIR, "gap_variability_by_profile.csv"), index=False
    )

    # ==================================================================
    # 3. GAP VARIANCE DECOMPOSITION (per K)
    # ==================================================================
    decomp_rows = []
    total_var = float(GAP.var(ddof=1))
    for K in K_RANGE:
        labels = labels_by_k[K]
        profile_means = []
        within_variances = []
        weights = []
        for p in range(K):
            mask = labels == p
            gap_p = GAP[mask]
            profile_means.append(float(gap_p.mean()))
            within_variances.append(float(gap_p.var(ddof=1)))
            weights.append(int(mask.sum()) / N)

        weights_arr = np.array(weights)
        # Between-profile variance (using weighted mean)
        grand_mean = np.sum(weights_arr * np.array(profile_means))
        between_var = np.sum(weights_arr * (np.array(profile_means) - grand_mean)**2)
        # Within-profile variance (weighted mean of within-group variances)
        if K > 1:
            within_var = np.sum(weights_arr * np.array(within_variances))
        else:
            within_var = 0.0
        decomp_rows.append({
            "K": K,
            "between_profile_variance": float(between_var),
            "within_profile_variance": float(within_var),
            "total_variance": total_var,
            "between_over_total": float(between_var / total_var) if total_var > 0 else np.nan,
        })
    pd.DataFrame(decomp_rows).to_csv(
        os.path.join(RESULTS_DIR, "gap_variance_decomposition.csv"), index=False
    )

    # ==================================================================
    # 4. PROFILE SEPARATION ON GAP (pairwise Cohen's d)
    # ==================================================================
    sep_rows = []
    for K in K_RANGE:
        labels = labels_by_k[K]
        for p1, p2 in [(i, j) for i in range(K) for j in range(i + 1, K)]:
            g1 = GAP[labels == p1]
            g2 = GAP[labels == p2]
            d = cohens_d(g1, g2)
            sep_rows.append({
                "K": K,
                "profile_1": p1,
                "profile_2": p2,
                "mean_GAP_p1": float(g1.mean()),
                "mean_GAP_p2": float(g2.mean()),
                "GAP_difference": float(g1.mean() - g2.mean()),
                "Cohens_d": d,
            })
    pd.DataFrame(sep_rows).to_csv(
        os.path.join(RESULTS_DIR, "pairwise_gap_separation.csv"), index=False
    )

    # ==================================================================
    # 5. GAP-ONLY CLUSTERING CHECK
    # ==================================================================
    # 1D Gaussian mixture on GAP, K=1..selected K
    gap_2d = GAP.reshape(-1, 1)
    gap_only_rows = []
    best_bic = np.inf
    best_k_gap = None

    for K_go in list(range(1, K_SEL + 1)):
        gmm = GaussianMixture(
            n_components=K_go,
            covariance_type="full",
            n_init=N_INIT,
            random_state=SEED,
            max_iter=500,
            reg_covar=1e-6,
        )
        gmm.fit(gap_2d)
        n = gap_2d.shape[0]
        n_features = gap_2d.shape[1]
        # For 1D full covariance, params = K*1 + K*1 + (K-1) = 2K + (K-1) = 3K-1
        k_params = count_params(K_go, n_features, "full")
        LL = float(gmm.score(gap_2d) * n)
        AIC = float(-2 * LL + 2 * k_params)
        BIC = float(-2 * LL + k_params * np.log(n))
        entropy = safe_entropy(gmm.predict_proba(gap_2d))
        labels_go = gmm.predict(gap_2d)
        sizes = np.bincount(labels_go, minlength=K_go)
        means_go = gmm.means_.flatten()
        stds_go = np.sqrt(gmm.covariances_.flatten())

        gap_only_rows.append({
            "K": K_go,
            "log_likelihood": LL,
            "AIC": AIC,
            "BIC": BIC,
            "entropy": entropy,
            "n_params": k_params,
            "converged": bool(gmm.converged_),
            "profile_sizes": str(sizes.tolist()),
        })

        # Save selected K_go solution
        go_kdir = os.path.join(RESULTS_DIR, f"gap_only_K{K_go}.csv")
        go_df = pd.DataFrame({
            "profile": range(K_go),
            "N": sizes,
            "proportion": gmm.weights_,
            "mean_GAP": means_go,
            "SD_GAP": stds_go,
        })
        go_df.to_csv(go_kdir, index=False)

        if BIC < best_bic:
            best_bic = BIC
            best_k_gap = K_go

    pd.DataFrame(gap_only_rows).to_csv(
        os.path.join(RESULTS_DIR, "gap_only_model_comparison.csv"), index=False
    )

    # Save selected gap-only K
    selected_row = pd.DataFrame({
        "selected_K": [best_k_gap],
        "BIC": [best_bic],
        "selection_rule": "minimum BIC",
    })
    selected_row.to_csv(
        os.path.join(RESULTS_DIR, "gap_only_selected.csv"), index=False
    )

    # ==================================================================
    # 6. OVERALL LEVEL VS GAP
    # ==================================================================
    r, p = stats.pearsonr(LEVEL.values, GAP)
    level_gap = pd.DataFrame({
        "variable": ["LEVEL", "GAP"],
        "mean": [float(LEVEL.mean()), float(GAP.mean())],
        "SD": [float(LEVEL.std(ddof=1)), float(GAP.std(ddof=1))],
        "min": [float(LEVEL.min()), float(GAP.min())],
        "max": [float(LEVEL.max()), float(GAP.max())],
    })
    level_gap.to_csv(
        os.path.join(RESULTS_DIR, "level_gap_relationship.csv"), index=False
    )
    # Add correlation
    corr_row = pd.DataFrame({
        "variable": ["LEVEL_GAP_correlation"],
        "mean": [float(r)],
        "SD": [np.nan],
        "min": [float(p)],
        "max": [N],
    })
    # Append as extra row — actually better as separate file
    level_gap_corr = pd.DataFrame({
        "correlation_LEVEl_GAP_r": [float(r)],
        "p_value": [float(p)],
        "N": [N],
    })
    level_gap_corr.to_csv(
        os.path.join(RESULTS_DIR, "level_gap_correlation.csv"), index=False
    )
    # Append to relationship file
    with open(os.path.join(RESULTS_DIR, "level_gap_relationship.csv"), "a") as f:
        f.write(f"LEVEL_GAP_Pearson_r,{r},{np.nan},{float(p)},{N}\n")

    # ==================================================================
    # 7. SELECTED-K PROFILE CENTROID GEOMETRY
    # ==================================================================
    # (previously hardcoded to K=6; now uses the selected K from Phase 05)
    kdir_sel = os.path.join(PHASE04_DIR, f"K_{K_SEL}")
    means_sel = pd.read_csv(os.path.join(kdir_sel, "profile_means.csv")).values  # (K_SEL, 2)

    centroid_rows = []
    for i in range(K_SEL):
        for j in range(i + 1, K_SEL):
            dist = float(np.linalg.norm(means_sel[i] - means_sel[j]))
            centroid_rows.append({
                "profile_1": i,
                "profile_2": j,
                "euclidean_distance": dist,
            })
    # Distance from origin
    origin_distances = []
    for p in range(K_SEL):
        origin_distances.append({
            "profile": p,
            "centroid_distance_from_origin": float(np.linalg.norm(means_sel[p])),
            "z_INT": float(means_sel[p][0]),
            "z_BE": float(means_sel[p][1]),
        })
    pd.DataFrame(centroid_rows).to_csv(
        os.path.join(RESULTS_DIR, "k6_centroid_geometry.csv"), index=False
    )
    pd.DataFrame(origin_distances).to_csv(
        os.path.join(RESULTS_DIR, "k6_centroid_distances.csv"), index=False
    )

    # ==================================================================
    # 8. SELECTED-K PROFILE CONTRIBUTION TO GAP
    # ==================================================================
    # (previously hardcoded to K=6; now uses the selected K from Phase 05)
    labels_sel = labels_by_k[K_SEL]
    total_var_sel = float(GAP.var(ddof=1))
    profile_means = []
    within_vars = []
    weights = []
    for p in range(K_SEL):
        mask = labels_sel == p
        gap_p = GAP[mask]
        profile_means.append(float(gap_p.mean()))
        within_vars.append(float(gap_p.var(ddof=1)))
        weights.append(int(mask.sum()) / N)
    weights_arr = np.array(weights)
    grand_mean_sel = np.sum(weights_arr * np.array(profile_means))
    between_var_sel = np.sum(weights_arr * (np.array(profile_means) - grand_mean_sel)**2)
    within_var_sel = np.sum(weights_arr * np.array(within_vars))

    k_gap_var = pd.DataFrame({
        "variance_component": ["between_profile", "within_profile", "total"],
        "value": [between_var_sel, within_var_sel, total_var_sel],
    })
    k_gap_var.loc[len(k_gap_var)] = ["between_over_total_pct", float(between_var_sel / total_var_sel * 100)] if total_var_sel > 0 else [np.nan]
    k_gap_var.to_csv(
        os.path.join(RESULTS_DIR, "k6_gap_variance_explained.csv"), index=False
    )

    # ==================================================================
    # 9. MASTER RESULTS
    # ==================================================================
    master_rows = []
    for K in K_RANGE:
        d = decomp_rows[K_RANGE.index(K)]
        master_rows.append({
            "K": K,
            "between_profile_variance_GAP": d["between_profile_variance"],
            "within_profile_variance_GAP": d["within_profile_variance"],
            "between_over_total_pct": float(d["between_profile_variance"] / total_var * 100),
        })
    # Add gap-only selected
    master_rows.append({
        "K": best_k_gap,
        "between_profile_variance_GAP": f"gap_only_selected={best_k_gap}",
        "within_profile_variance_GAP": "",
        "between_over_total_pct": "",
    })
    pd.DataFrame(master_rows).to_csv(
        os.path.join(RESULTS_DIR, "phase16_master.csv"), index=False
    )

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = f"""# Phase 16 — Intention–Behavior Gap Profile Validation

## Purpose
Determine whether observed LPA profiles represent meaningful
INT–BE configurations or mainly reflect overall response level.
Descriptive/computational only. No K selection, no labels, no
interpretation, no plots, no literature.

## Input Result Files
- results/02_measurement/construct_scores.csv
- results/03_gap_analysis/gap_scores.npy
- results/04_lpa_estimation/K_{{K}}/posterior_probabilities.csv

## Formulas
- GAP_i = z(INT_i) - z(BE_i), where z = (x - mean) / SD
- LEVEL = (z_INT + z_BE) / 2
- Variance decomposition: Var_total = Var_between + weighted Var_within
  - weights = profile sample proportions
  - Var_between = sum(w_p * (mean_p - grand_mean)^2)
  - Var_within = sum(w_p * var_p)
- Cohen's d = (M1 - M2) / pooled_SD
  - pooled_SD = sqrt[((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2)]
- Euclidean distance between centroids: sqrt(d1^2 + d2^2)

## Gap-only clustering
- GaussianMixture(n_components=K_go, covariance_type='full', n_init=1000,
  random_state=42, max_iter=500)
- INDICATOR: respondent-level GAP (1D)
- Model selection: MINIMUM BIC (corrected procedure)
- K_go = 1..{K_SEL}

## Parameters
- N = {N}
- SEED = 42 (explicit, documented)
- N_INIT = 1000
- BIC minimization (NOT maximization — historical bug corrected)

## Output Files
- profile_int_be_by_k.csv
- gap_variability_by_profile.csv
- gap_variance_decomposition.csv
- pairwise_gap_separation.csv
- gap_only_model_comparison.csv
- gap_only_K{{1..{K_SEL}}}.csv
- gap_only_selected.csv
- level_gap_relationship.csv
- level_gap_correlation.csv
- k6_centroid_geometry.csv
- k6_centroid_distances.csv
- k6_gap_variance_explained.csv
- phase16_master.csv
- README.md

## No Modifications
All calculations read existing result files. No refitting of the
primary multivariate LPA. No dataset changes. No model-selection
change. No plots.
"""
    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write(readme)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 16 — GAP PROFILE VALIDATION COMPLETE")
    print("=" * 60)
    print(f"N = {N}, SEED = {SEED}, N_INIT = {N_INIT}")
    print(f"\nGap variance decomposition:")
    print(pd.DataFrame(decomp_rows).to_string(index=False))
    print(f"\nGap-only model (selected by min BIC, K={best_k_gap}):")
    print(pd.DataFrame(gap_only_rows).to_string(index=False))
    print(f"\nLEVEL-GAP Pearson r = {r:.4f}, p = {p:.6f}")
    print(f"K={K_SEL} centroid distances from origin:")
    print(pd.DataFrame(origin_distances).to_string(index=False))

    print("\nOutputs:")
    for fn in ["profile_int_be_by_k.csv", "gap_variability_by_profile.csv",
               "gap_variance_decomposition.csv", "pairwise_gap_separation.csv",
               "gap_only_model_comparison.csv", "gap_only_selected.csv",
               "level_gap_relationship.csv", "level_gap_correlation.csv",
               "k6_centroid_geometry.csv", "k6_centroid_distances.csv",
               "k6_gap_variance_explained.csv", "phase16_master.csv", "README.md"]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")
    for k in range(1, K_SEL + 1):
        path = os.path.join(RESULTS_DIR, f"gap_only_K{k}.csv")
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
