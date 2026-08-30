"""
Reviewer-fix pass — Task 1: K=2..10 model comparison (primary specification)

Mirrors scripts/04_lpa_estimation.py run_gmm() EXACTLY for K=2..10:
- sklearn GaussianMixture, covariance_type="full"
- n_init=1000, random_state=42
- max_iter=500, reg_covar=1e-6
- indicators: z_INT, z_BE standardized WITHIN the working sample (ddof=1)

This does NOT modify the original Phase 04 evidence. New outputs go under
results/paper1_strengthening/.

Hard stop: produces k_extended_model_comparison.csv and K/ profile tables.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

SCORES_PATH = "results/02_measurement/construct_scores.csv"
RESULTS_DIR = "results/paper1_strengthening"
SEED = 42
N_INIT = 1000
COVARIANCE_TYPE = "full"
MAX_ITER = 500
REG_COVAR = 1e-6
K_RANGE = [2, 3, 4, 5, 6, 7, 8, 9, 10]
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


def count_params_full(K, n_features):
    return int(K * n_features + K * n_features * (n_features + 1) / 2.0 + (K - 1))


def run_gmm(X, K, seed):
    gmm = GaussianMixture(
        n_components=K,
        covariance_type=COVARIANCE_TYPE,
        n_init=N_INIT,
        random_state=seed,
        max_iter=MAX_ITER,
        reg_covar=REG_COVAR,
    )
    gmm.fit(X)
    n = X.shape[0]
    k_params = count_params_full(K, X.shape[1])
    log_lik = float(gmm.score(X) * n)
    AIC = float(-2 * log_lik + 2 * k_params)
    BIC = float(-2 * log_lik + k_params * np.log(n))
    entropy = safe_entropy(gmm.predict_proba(X))
    return {
        "K": K,
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "log_likelihood": log_lik,
        "n_params": k_params,
        "AIC": AIC,
        "BIC": BIC,
        "entropy": entropy,
        "weights": gmm.weights_.tolist(),
        "means": gmm.means_.tolist(),
        "covariances": gmm.covariances_.tolist(),
        "labels": gmm.predict(X).tolist(),
        "posteriors": gmm.predict_proba(X).tolist(),
    }


def main():
    ensure_dir(RESULTS_DIR)
    np.random.seed(SEED)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    N = scores.shape[0]
    z_INT = (scores["INT"] - scores["INT"].mean()) / scores["INT"].std(ddof=1)
    z_BE = (scores["BE"] - scores["BE"].mean()) / scores["BE"].std(ddof=1)
    X = np.column_stack([z_INT.values, z_BE.values])
    n_features = X.shape[1]

    rows = []
    per_k_dirs = {}
    for K in K_RANGE:
        res = run_gmm(X, K, SEED)
        labels = np.array(res["labels"])
        sizes = np.bincount(labels, minlength=K)
        rows.append({
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
            "min_class_N": int(sizes.min()),
            "class_sizes": str(sizes.tolist()),
        })
        kdir = os.path.join(RESULTS_DIR, f"task1_K_{K}")
        os.makedirs(kdir, exist_ok=True)
        per_k_dirs[K] = kdir

        profile = pd.DataFrame({
            "profile": range(K),
            "size": sizes,
            "proportion": res["weights"],
            "mean_z_INT": np.array(res["means"])[:, 0],
            "mean_z_BE": np.array(res["means"])[:, 1],
        })
        profile.to_csv(os.path.join(kdir, "profile_parameters.csv"), index=False)

        cov_records = {f"profile_{j}": np.array(res["covariances"][j]).tolist()
                       for j in range(K)}
        with open(os.path.join(kdir, "covariance_matrices.json"), "w") as f:
            json.dump({"feature_names": INDICATORS, "covariances": cov_records}, f, indent=2)

        np.array(res["means"]).tofile(os.path.join(kdir, "profile_means.npy"))
        pd.DataFrame(res["means"], columns=INDICATORS).to_csv(
            os.path.join(kdir, "profile_means.csv"), index=False)
        print(f"K={K}: conv={res['converged']}, LL={res['log_likelihood']:.2f}, "
              f"AIC={res['AIC']:.2f}, BIC={res['BIC']:.2f}, "
              f"entropy={res['entropy']:.4f}, min_N={int(sizes.min())}, sizes={sizes.tolist()}")

    fit_df = pd.DataFrame(rows)
    fit_df.to_csv(os.path.join(RESULTS_DIR, "k_extended_model_comparison.csv"), index=False)

    best_bic = fit_df.loc[fit_df["BIC"].idxmin()]
    best_aic = fit_df.loc[fit_df["AIC"].idxmin()]
    k6 = fit_df.loc[fit_df["K"] == 6].iloc[0]

    print("\n" + "=" * 70)
    print("TASK 1 — K=2..10 EXTENDED MODEL COMPARISON (full covariance)")
    print("=" * 70)
    print(fit_df[["K", "log_likelihood", "n_params", "AIC", "BIC", "entropy",
                  "min_class_N"]].to_string(index=False))
    print(f"\nMinimum BIC at K = {int(best_bic['K'])}, BIC = {best_bic['BIC']:.2f}")
    print(f"Minimum AIC at K = {int(best_aic['K'])}, AIC = {best_aic['AIC']:.2f}")
    print(f"K=6 reference: BIC = {k6['BIC']:.2f}, AIC = {k6['AIC']:.2f}, "
          f"LL = {k6['log_likelihood']:.2f}")
    print(f"\nSaved: {os.path.join(RESULTS_DIR, 'k_extended_model_comparison.csv')}")
    for K in K_RANGE:
        kdir = per_k_dirs[K]
        print(f"  {kdir}/profile_parameters.csv + covariance_matrices.json — OK")


if __name__ == "__main__":
    main()
