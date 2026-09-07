"""
Phase 08 — Robustness and Sensitivity
=======================================
Project: Intention–Behaviour Gap in Household Energy-Saving Behaviour

Checks:
1. Random-start stability
2. Random-seed sensitivity
3. Covariance specification (full, diag, spherical)
4. Score sensitivity (mean vs factor score)
5. Classification uncertainty
6. Outlier sensitivity (|z| > 3, sensitivity only)
7. Profile stability across conditions
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORES_PATH = "results/02_measurement/construct_scores.csv"
PHASE04_DIR = "results/04_lpa_estimation"
PHASE05_DIR = "results/05_lpa_selection"
RESULTS_DIR = "results/08_robustness"
SEED = 42

# K range follows the fitted models in Phase 04 (K=2..7 after the primary-
# solution switch); read dynamically so future K extensions propagate.
import pandas as _pd
K_RANGE = sorted(_pd.read_csv(os.path.join(PHASE04_DIR, "model_fit.csv"))["K"].unique().tolist())
COVARIANCE_TYPES = ["full", "diag", "spherical"]
INDICATORS = ["z_INT", "z_BE"]


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def safe_entropy(posterior):
    p = np.clip(posterior, 1e-15, 1.0)
    n, K = posterior.shape
    log_k = np.log(K)
    if log_k == 0:
        return 0.0
    total = np.sum(p * np.log(p))
    return float(-total / (n * log_k))


def count_params(K, n_features, cov_type):
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


def fit_gmm(X, K, cov_type, seed, n_init=1000):
    gmm = GaussianMixture(
        n_components=K,
        covariance_type=cov_type,
        n_init=n_init,
        random_state=seed,
        max_iter=500,
        reg_covar=1e-6,
    )
    gmm.fit(X)
    n = X.shape[0]
    n_features = X.shape[1]
    k_params = count_params(K, n_features, cov_type)
    LL = float(gmm.score(X) * n)
    AIC = float(-2 * LL + 2 * k_params)
    BIC = float(-2 * LL + k_params * np.log(n))
    entropy = safe_entropy(gmm.predict_proba(X))
    labels = gmm.predict(X)
    sizes = np.bincount(labels, minlength=K).tolist()
    return {
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "LL": LL,
        "AIC": AIC,
        "BIC": BIC,
        "entropy": entropy,
        "sizes": sizes,
        "weights": gmm.weights_.tolist(),
        "means": gmm.means_.tolist(),
    }


def main():
    ensure_dir(RESULTS_DIR)
    np.random.seed(SEED)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    N = scores.shape[0]
    selected = pd.read_csv(os.path.join(PHASE05_DIR, "selected_model.csv"))
    K_sel = int(selected["selected_K"].values[0])

    z_INT = (scores["INT"] - scores["INT"].mean()) / scores["INT"].std(ddof=1)
    z_BE = (scores["BE"] - scores["BE"].mean()) / scores["BE"].std(ddof=1)
    X = np.column_stack([z_INT.values, z_BE.values])

    # ------------------------------------------------------------------
    # CHECK 1 — Random-start stability
    # ------------------------------------------------------------------
    rs_rows = []
    for K in K_RANGE:
        res = fit_gmm(X, K, "full", SEED, n_init=1000)
        rs_rows.append(
            {
                "K": K,
                "n_init": 1000,
                "seed": SEED,
                "converged": res["converged"],
                "n_iter": res["n_iter"],
                "log_likelihood": res["LL"],
                "AIC": res["AIC"],
                "BIC": res["BIC"],
                "entropy": res["entropy"],
                "class_sizes": res["sizes"],
            }
        )
    pd.DataFrame(rs_rows).to_csv(
        os.path.join(RESULTS_DIR, "random_start_stability.csv"), index=False
    )

    # ------------------------------------------------------------------
    # CHECK 2 — Random-seed sensitivity
    # ------------------------------------------------------------------
    seeds = [42, 123, 999, 2024, 7]
    seed_rows = []
    for K in K_RANGE:
        for s in seeds:
            res = fit_gmm(X, K, "full", s, n_init=100)
            seed_rows.append(
                {
                    "K": K,
                    "seed": s,
                    "converged": res["converged"],
                    "n_iter": res["n_iter"],
                    "log_likelihood": res["LL"],
                    "BIC": res["BIC"],
                    "entropy": res["entropy"],
                    "class_sizes": res["sizes"],
                }
            )
    pd.DataFrame(seed_rows).to_csv(
        os.path.join(RESULTS_DIR, "random_seed_sensitivity.csv"), index=False
    )

    # ------------------------------------------------------------------
    # CHECK 3 — Covariance specification
    # ------------------------------------------------------------------
    cov_rows = []
    for K in K_RANGE:
        for cov in COVARIANCE_TYPES:
            res = fit_gmm(X, K, cov, SEED, n_init=100)
            cov_rows.append(
                {
                    "K": K,
                    "covariance_type": cov,
                    "converged": res["converged"],
                    "n_iter": res["n_iter"],
                    "log_likelihood": res["LL"],
                    "AIC": res["AIC"],
                    "BIC": res["BIC"],
                    "entropy": res["entropy"],
                    "class_sizes": res["sizes"],
                }
            )
    pd.DataFrame(cov_rows).to_csv(
        os.path.join(RESULTS_DIR, "covariance_sensitivity.csv"), index=False
    )

    # ------------------------------------------------------------------
    # CHECK 4 — Score sensitivity (mean vs factor score)
    # ------------------------------------------------------------------
    # Factor scores via 1-component PCA per construct (proxy for factor score).
    # mean scores are already available from Phase 02.
    data_raw = pd.read_excel("data.xls", sheet_name=0)
    int_items = ["INT1", "INT2", "INT3"]
    be_items = ["BE1", "BE2", "BE3", "BE4"]

    def factor_score_proxy(items, n_factors=1):
        """1-component PCA score as a factor-score proxy."""
        X_items = data_raw[items].dropna().values.astype(float)
        # Center by item mean
        Xc = X_items - X_items.mean(axis=0)
        pca = PCA(n_components=n_factors)
        scores = pca.fit_transform(Xc)
        # Return first component scaled to match construct SD direction
        # Use the loadings-weighted score (regression-based factor score proxy)
        return scores.flatten()

    fs_INT = factor_score_proxy(int_items)
    fs_BE = factor_score_proxy(be_items)

    mean_INT = scores["INT"].values
    mean_BE = scores["BE"].values

    # PCA factor scores have sign ambiguity — align sign to mean score
    # correlation to ensure comparable direction
    r_raw = np.corrcoef(mean_INT, fs_INT)[0, 1]
    if r_raw < 0:
        fs_INT = -fs_INT
    r_raw2 = np.corrcoef(mean_BE, fs_BE)[0, 1]
    if r_raw2 < 0:
        fs_BE = -fs_BE

    r_int = float(np.corrcoef(mean_INT, fs_INT)[0, 1])
    r_be = float(np.corrcoef(mean_BE, fs_BE)[0, 1])

    # Re-run LPA with factor-score-based indicators
    z_fs_INT = (fs_INT - fs_INT.mean()) / fs_INT.std(ddof=1)
    z_fs_BE = (fs_BE - fs_BE.mean()) / fs_BE.std(ddof=1)
    X_fs = np.column_stack([z_fs_INT, z_fs_BE])

    fs_lpa_rows = []
    for K in K_RANGE:
        res_mean = fit_gmm(X, K, "full", SEED, n_init=100)
        res_fs = fit_gmm(X_fs, K, "full", SEED, n_init=100)
        fs_lpa_rows.append(
            {
                "K": K,
                "BIC_mean": res_mean["BIC"],
                "BIC_factor": res_fs["BIC"],
                "sizes_mean": res_mean["sizes"],
                "sizes_factor": res_fs["sizes"],
            }
        )

    score_sens = pd.DataFrame(
        {
            "score_comparison": ["mean_vs_factor_INT", "mean_vs_factor_BE"],
            "pearson_r": [r_int, r_be],
        }
    )
    score_sens.to_csv(
        os.path.join(RESULTS_DIR, "score_sensitivity.csv"), index=False
    )
    pd.DataFrame(fs_lpa_rows).to_csv(
        os.path.join(RESULTS_DIR, "score_sensitivity_lpa.csv"), index=False
    )

    # ------------------------------------------------------------------
    # CHECK 5 — Classification uncertainty (for selected K)
    # ------------------------------------------------------------------
    kdir = os.path.join(PHASE04_DIR, f"K_{K_sel}")
    post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
    post_cols = [c for c in post_df.columns if c.startswith("post_profile_")]
    posteriors = post_df[post_cols].values
    max_post = posteriors.max(axis=1)
    cls_rows = []
    for K in K_RANGE:
        kdir_k = os.path.join(PHASE04_DIR, f"K_{K}")
        pdf_k = pd.read_csv(os.path.join(kdir_k, "posterior_probabilities.csv"))
        pc_k = [c for c in pdf_k.columns if c.startswith("post_profile_")]
        post_k = pdf_k[pc_k].values
        mp = post_k.max(axis=1)
        cls_rows.append(
            {
                "K": K,
                "pct_lt_50": (mp < 0.50).mean() * 100,
                "pct_lt_70": (mp < 0.70).mean() * 100,
                "pct_ge_70": (mp >= 0.70).mean() * 100,
            }
        )
    pd.DataFrame(cls_rows).to_csv(
        os.path.join(RESULTS_DIR, "classification_sensitivity.csv"), index=False
    )

    # ------------------------------------------------------------------
    # CHECK 6 — Outlier sensitivity (|z| > 3)
    # ------------------------------------------------------------------
    # Applied as sensitivity only; primary analysis unchanged
    z_all = np.concatenate([z_INT.values, z_BE.values])
    mask_outlier = (np.abs(z_INT) > 3) | (np.abs(z_BE) > 3)
    X_clean = X[~mask_outlier]
    n_removed = int(mask_outlier.sum())

    outlier_rows = []
    for K in K_RANGE:
        res_all = fit_gmm(X, K, "full", SEED, n_init=100)
        res_clean = fit_gmm(X_clean, K, "full", SEED, n_init=100)
        outlier_rows.append(
            {
                "K": K,
                "BIC_full": res_all["BIC"],
                "BIC_no_outliers": res_clean["BIC"],
                "sizes_full": res_all["sizes"],
                "sizes_no_outliers": res_clean["sizes"],
                "n_removed": n_removed,
                "outlier_rule": "|z|>3",
            }
        )
    pd.DataFrame(outlier_rows).to_csv(
        os.path.join(RESULTS_DIR, "outlier_sensitivity.csv"), index=False
    )

    # ------------------------------------------------------------------
    # CHECK 7 — Profile stability
    # ------------------------------------------------------------------
    stab_rows = []
    # Across seeds (already in seed_rows) — summarize
    seed_df = pd.DataFrame(seed_rows)
    for K in K_RANGE:
        sub = seed_df[seed_df["K"] == K]
        stab_rows.append(
            {
                "K": K,
                "condition": "across_seeds",
                "BIC_std": float(sub["BIC"].std()),
                "BIC_min": float(sub["BIC"].min()),
                "BIC_max": float(sub["BIC"].max()),
            }
        )
    # Across covariance specs
    cov_df = pd.DataFrame(cov_rows)
    for K in K_RANGE:
        sub = cov_df[cov_df["K"] == K]
        stab_rows.append(
            {
                "K": K,
                "condition": "across_covariance",
                "BIC_std": float(sub["BIC"].std()),
                "BIC_min": float(sub["BIC"].min()),
                "BIC_max": float(sub["BIC"].max()),
            }
        )
    pd.DataFrame(stab_rows).to_csv(
        os.path.join(RESULTS_DIR, "profile_stability.csv"), index=False
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 08 — ROBUSTNESS CHECKS COMPLETE")
    print("=" * 60)
    print(f"Selected K = {K_sel}")
    print(f"Mean vs factor score r: INT={r_int:.4f}, BE={r_be:.4f}")
    print(f"Outliers removed (|z|>3): {n_removed}")
    print(f"\nRandom-start stability (n_init=1000):")
    print(pd.DataFrame(rs_rows)[["K", "converged", "BIC", "entropy"]].to_string(index=False))
    print("\nKey outputs:")
    for fn in [
        "random_start_stability.csv",
        "random_seed_sensitivity.csv",
        "covariance_sensitivity.csv",
        "score_sensitivity.csv",
        "score_sensitivity_lpa.csv",
        "classification_sensitivity.csv",
        "outlier_sensitivity.csv",
        "profile_stability.csv",
    ]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
