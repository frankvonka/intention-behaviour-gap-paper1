"""
Phase 04 — Latent Profile Analysis — Estimation
===================================================
Project: Intention–Behaviour Gap in Household Energy-Saving Behaviour

Latent profile indicators: INT, BE (standardized z-scores).

Fits Gaussian mixture models for K = 2..6 using a full covariance
specification (primary model).

For each K:
- 1000 random initializations (n_init) where feasible
- records convergence, log-likelihood, AIC, BIC, entropy
- saves posterior probabilities, class assignments, profile means,
  covariance matrices, class proportions
- random-start diagnostics

BIC is minimized (NOT maximized).
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORES_PATH = "results/02_measurement/construct_scores.csv"
RESULTS_DIR = "results/04_lpa_estimation"
SEED = 42

K_RANGE = [2, 3, 4, 5, 6, 7]
COVARIANCE_TYPE = "full"  # primary specification
N_INIT = 1000  # random initializations per K

# Indicators (use standardized z-scores per Phase 03 procedure)
INDICATORS = ["z_INT", "z_BE"]


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def safe_entropy(posterior):
    """
    Classification entropy:
    E = -[sum_i sum_j p_ij * log(p_ij)] / (n * log(K))
    Uses numerically-safe log.
    """
    # Clip to avoid log(0)
    p = np.clip(posterior, 1e-15, 1.0)
    n, K = posterior.shape
    log_k = np.log(K)
    if log_k == 0:
        return 0.0
    total = np.sum(p * np.log(p))
    return float(-total / (n * log_k))


def count_params_full(K, n_features):
    """Number of parameters in a full-covariance GMM."""
    # Means: K * n_features
    means = K * n_features
    # Covariances: K * n_features * (n_features + 1) / 2
    covs = K * n_features * (n_features + 1) / 2.0
    # Weights: K - 1
    weights = K - 1
    return int(means + covs + weights)


def run_gmm(X, K, seed):
    """
    Fit a full-covariance GMM with K components.
    Returns a dict of all relevant quantities.
    """
    gmm = GaussianMixture(
        n_components=K,
        covariance_type=COVARIANCE_TYPE,
        n_init=N_INIT,
        random_state=seed,
        max_iter=500,
        reg_covar=1e-6,
    )
    gmm.fit(X)

    n = X.shape[0]
    k_params = count_params_full(K, X.shape[1])
    log_lik = float(gmm.score(X) * n)  # log-likelihood = score * n
    AIC = float(-2 * log_lik + 2 * k_params)
    BIC = float(-2 * log_lik + k_params * np.log(n))
    entropy = safe_entropy(gmm.predict_proba(X))

    # Profile (component) means in the standardized space
    means = gmm.means_  # shape (K, n_features)
    covariances = gmm.covariances_  # shape (K, n_features, n_features)
    weights = gmm.weights_  # shape (K,)
    labels = gmm.predict(X)
    posteriors = gmm.predict_proba(X)

    return {
        "K": K,
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "log_likelihood": log_lik,
        "n_params": k_params,
        "AIC": AIC,
        "BIC": BIC,
        "entropy": entropy,
        "weights": weights.tolist(),
        "means": means.tolist(),
        "covariances": covariances.tolist(),
        "labels": labels.tolist(),
        "posteriors": posteriors.tolist(),
        "n_init": N_INIT,
    }


def main():
    ensure_dir(RESULTS_DIR)
    np.random.seed(SEED)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    N = scores.shape[0]

    # Standardize INT and BE across respondents (consistent with Phase 03)
    z_INT = (scores["INT"] - scores["INT"].mean()) / scores["INT"].std(ddof=1)
    z_BE = (scores["BE"] - scores["BE"].mean()) / scores["BE"].std(ddof=1)

    X = np.column_stack([z_INT.values, z_BE.values])
    feature_names = INDICATORS
    n_features = X.shape[1]

    # Save standardized indicators
    pd.DataFrame(
        {"respondent": scores.index, "z_INT": z_INT.values, "z_BE": z_BE.values}
    ).to_csv(os.path.join(RESULTS_DIR, "lpa_indicators.csv"), index=False)

    # ------------------------------------------------------------------
    # Fit models for each K
    # ------------------------------------------------------------------
    fit_rows = []
    k_results = {}

    for K in K_RANGE:
        res = run_gmm(X, K, SEED)
        k_results[K] = res

        # Class sizes
        labels = np.array(res["labels"])
        sizes = np.bincount(labels, minlength=K)

        fit_rows.append(
            {
                "K": K,
                "covariance_type": COVARIANCE_TYPE,
                "n_init": N_INIT,
                "random_seed": SEED,
                "converged": res["converged"],
                "n_iter": res["n_iter"],
                "log_likelihood": res["log_likelihood"],
                "n_params": res["n_params"],
                "AIC": res["AIC"],
                "BIC": res["BIC"],
                "entropy": res["entropy"],
                "smallest_class_pct": float(sizes.min() / N * 100),
                "largest_class_pct": float(sizes.max() / N * 100),
            }
        )

        # Save per-K files
        kdir = os.path.join(RESULTS_DIR, f"K_{K}")
        os.makedirs(kdir, exist_ok=True)

        # Profile parameters
        # NOTE: `size` = hard-assigned counts (argmax posterior); `weight` =
        # GMM mixture weights (soft). These differ whenever classification is
        # uncertain (e.g. K=6 P1: weight 0.281 vs assigned 377/1166 = 0.323).
        # Report profile SHARE from `size`/N; never present `weight` as N-share.
        _sizes = np.asarray(sizes, dtype=int)
        profile = pd.DataFrame(
            {
                "profile": range(K),
                "size": _sizes,
                "assigned_share": _sizes / N,
                "weight": np.array(res["weights"], dtype=float),
                "mean_z_INT": np.array(res["means"])[:, 0],
                "mean_z_BE": np.array(res["means"])[:, 1],
            }
        )
        profile.to_csv(os.path.join(kdir, "profile_parameters.csv"), index=False)
        # Keep legacy `proportion` (= weight) for backward-compat, clearly labeled.
        _psizes = profile[["profile", "size"]].copy()
        _psizes["proportion_WEIGHT_NOT_SHARE"] = profile["weight"]
        _psizes["proportion"] = profile["weight"]
        _psizes["assigned_share"] = profile["assigned_share"]
        _psizes.to_csv(
            os.path.join(kdir, "profile_sizes.csv"), index=False
        )

        # Covariance matrices saved as JSON (3D array)
        cov_records = {}
        for j in range(K):
            cov_records[f"profile_{j}"] = np.array(res["covariances"][j]).tolist()
        with open(os.path.join(kdir, "covariance_matrices.json"), "w") as f:
            json.dump(
                {
                    "feature_names": feature_names,
                    "covariances": cov_records,
                },
                f,
                indent=2,
            )

        # Posterior probabilities
        post_df = pd.DataFrame(
            res["posteriors"],
            columns=[f"post_profile_{j}" for j in range(K)],
        )
        post_df.insert(0, "respondent", scores.index)
        post_df.insert(1, "assigned_class", labels)
        post_df.to_csv(os.path.join(kdir, "posterior_probabilities.csv"), index=False)

        # Profile means
        np.array(res["means"]).tofile(
            os.path.join(kdir, "profile_means.npy"),
        )
        pd.DataFrame(
            res["means"], columns=feature_names
        ).to_csv(os.path.join(kdir, "profile_means.csv"), index=False)

        # Random-start diagnostics
        # (sklearn uses n_init internally; we report the converged best)
        diag = pd.DataFrame(
            {
                "metric": [
                    "converged",
                    "n_init_requested",
                    "n_iter_used",
                    "best_log_likelihood",
                    "BIC",
                    "random_seed",
                ],
                "value": [
                    res["converged"],
                    N_INIT,
                    res["n_iter"],
                    res["log_likelihood"],
                    res["BIC"],
                    SEED,
                ],
            }
        )
        diag.to_csv(os.path.join(kdir, "random_start_diagnostics.csv"), index=False)

        print(
            f"K={K}: conv={res['converged']}, LL={res['log_likelihood']:.2f}, "
            f"AIC={res['AIC']:.2f}, BIC={res['BIC']:.2f}, "
            f"entropy={res['entropy']:.4f}, "
            f"sizes={sizes.tolist()}"
        )

    # ------------------------------------------------------------------
    # Save model_fit.csv (the primary summary)
    # ------------------------------------------------------------------
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(os.path.join(RESULTS_DIR, "model_fit.csv"), index=False)

    # Save the selected K placeholder (to be determined in Phase 05)
    # Here we note the BIC-minimum as a diagnostic only
    best_bic_row = fit_df.loc[fit_df["BIC"].idxmin()]
    with open(os.path.join(RESULTS_DIR, "model_fit_summary.txt"), "w") as f:
        f.write(f"Data: {SCORES_PATH}\n")
        f.write(f"N = {N}, indicators = {feature_names}\n")
        f.write(f"covariance_type = {COVARIANCE_TYPE}, n_init = {N_INIT}, seed = {SEED}\n")
        f.write(f"Best BIC (minimum) at K = {int(best_bic_row['K'])}, BIC = {best_bic_row['BIC']:.2f}\n")
        f.write("NOTE: Model selection is performed in Phase 05, not here.\n")

    print("\n" + "=" * 60)
    print("PHASE 04 — LPA ESTIMATION COMPLETE")
    print("=" * 60)
    print(f"N = {N}, indicators = {feature_names}")
    print(f"covariance_type = {COVARIANCE_TYPE}, n_init = {N_INIT}")
    print(f"Seed = {SEED}")
    print(f"Best BIC (min) at K = {int(best_bic_row['K'])}, BIC = {best_bic_row['BIC']:.2f}")
    print(f"\nModel fit summary:\n{fit_df.to_string(index=False)}")
    print("\nKey outputs:")
    print(f"  {os.path.join(RESULTS_DIR, 'model_fit.csv')} — OK")
    for K in K_RANGE:
        kdir = os.path.join(RESULTS_DIR, f"K_{K}")
        for fn in [
            "profile_parameters.csv",
            "profile_sizes.csv",
            "covariance_matrices.json",
            "posterior_probabilities.csv",
            "profile_means.csv",
            "profile_means.npy",
            "random_start_diagnostics.csv",
        ]:
            path = os.path.join(kdir, fn)
            print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
