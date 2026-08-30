"""
Reviewer-fix pass — Task 2: Full vs diagonal covariance K=2..10.

For each K in 2..10, fit sklearn GaussianMixture twice:
  - covariance_type = "full"
  - covariance_type = "diag"
Both with the SAME indicators, n_init, seed, and preprocessing as the primary
specification (mean INT and BE scores, n_init=1000, random_state=42, max_iter=500,
reg_covar=1e-6, indicators standardized within sample, ddof=1).

Saves results/paper1_strengthening/covariance_k_extended.csv with one row per
(K, cov_type).
"""
import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

SCORES_PATH = "results/02_measurement/construct_scores.csv"
RESULTS_DIR = "results/paper1_strengthening"
SEED = 42
N_INIT = 1000
MAX_ITER = 500
REG_COVAR = 1e-6
K_RANGE = [2, 3, 4, 5, 6, 7, 8, 9, 10]
COV_TYPES = ["full", "diag"]
INDICATORS = ["z_INT", "z_BE"]


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def safe_entropy(posterior):
    p = np.clip(posterior, 1e-15, 1.0)
    n, K = posterior.shape
    log_k = np.log(K)
    if log_k == 0:
        return 0.0
    return float(1.0 - np.sum(p * np.log(p)) / (n * log_k))


def count_params(cov_type, K, n_features):
    means = K * n_features
    if cov_type == "full":
        covs = K * n_features * (n_features + 1) / 2.0
    elif cov_type == "diag":
        covs = K * n_features
    elif cov_type == "spherical":
        covs = K
    else:
        raise ValueError(cov_type)
    return int(means + covs + (K - 1))


def main():
    ensure_dir(RESULTS_DIR)
    np.random.seed(SEED)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    N = scores.shape[0]
    z_INT = (scores["INT"] - scores["INT"].mean()) / scores["INT"].std(ddof=1)
    z_BE = (scores["BE"] - scores["BE"].mean()) / scores["BE"].std(ddof=1)
    X = np.column_stack([z_INT.values, z_BE.values])

    rows = []
    for cov in COV_TYPES:
        for K in K_RANGE:
            gmm = GaussianMixture(
                n_components=K,
                covariance_type=cov,
                n_init=N_INIT,
                random_state=SEED,
                max_iter=MAX_ITER,
                reg_covar=REG_COVAR,
            )
            gmm.fit(X)
            n = X.shape[0]
            k_params = count_params(cov, K, X.shape[1])
            log_lik = float(gmm.score(X) * n)
            AIC = float(-2 * log_lik + 2 * k_params)
            BIC = float(-2 * log_lik + k_params * np.log(n))
            entropy = safe_entropy(gmm.predict_proba(X))
            sizes = np.bincount(gmm.predict(X), minlength=K)
            rows.append({
                "K": K,
                "covariance_type": cov,
                "n_init": N_INIT,
                "random_seed": SEED,
                "converged": bool(gmm.converged_),
                "n_iter": int(gmm.n_iter_),
                "log_likelihood": log_lik,
                "n_params": k_params,
                "AIC": AIC,
                "BIC": BIC,
                "entropy": entropy,
                "min_class_pct": float(sizes.min() / N * 100),
                "min_class_N": int(sizes.min()),
                "class_sizes": str(sizes.tolist()),
            })
            print(f"cov={cov:>5s} K={K}: conv={gmm.converged_}, LL={log_lik:.2f}, "
                  f"AIC={AIC:.2f}, BIC={BIC:.2f}, entropy={entropy:.4f}, min_N={sizes.min()}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "covariance_k_extended.csv"), index=False)

    print("\n" + "=" * 70)
    print("TASK 2 — FULL vs DIAGONAL COVARIANCE K=2..10")
    print("=" * 70)
    pivot_bic = df.pivot(index="K", columns="covariance_type", values="BIC")
    pivot_aic = df.pivot(index="K", columns="covariance_type", values="AIC")
    pivot_ll = df.pivot(index="K", columns="covariance_type", values="log_likelihood")
    print("BIC by (K, covariance_type):")
    print(pivot_bic.round(2).to_string())
    print("\nAIC by (K, covariance_type):")
    print(pivot_aic.round(2).to_string())
    print("\nlog-likelihood by (K, covariance_type):")
    print(pivot_ll.round(2).to_string())

    for cov in COV_TYPES:
        sub = df[df["covariance_type"] == cov]
        best = sub.loc[sub["BIC"].idxmin()]
        print(f"\nMinimum BIC ({cov}) at K = {int(best['K'])}, BIC = {best['BIC']:.2f}, "
              f"min_class_N = {int(best['min_class_N'])}")

    print(f"\nSaved: {os.path.join(RESULTS_DIR, 'covariance_k_extended.csv')}")


if __name__ == "__main__":
    main()
