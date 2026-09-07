"""
Task 27: K=7 diagnostics + 5-fold CV extension (K=2..7)
Produces the four evidence files consumed by the figures/manuscript that no
other pipeline script generates:
  - k7_classification_quality.csv
  - k7_posterior_probabilities.csv
  - k7_stability.csv
  - cv5_aggregated_by_k.csv
Uses the same primary specification as 04/23: sklearn GaussianMixture,
covariance_type="full", n_init=1000, random_state=42, max_iter=500,
reg_covar=1e-6, indicators z(INT), z(BE) standardized with ddof=1.
"""
import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment

SCORES_PATH = "results/02_measurement/construct_scores.csv"
RESULTS_DIR = "results/paper1_strengthening"
SEED = 42
N_INIT = 1000
MAX_ITER = 500
REG_COVAR = 1e-6
K7 = 7
N_BOOT = 200
N_FOLDS = 5


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def safe_entropy(posterior):
    p = np.clip(posterior, 1e-15, 1.0)
    n, K = posterior.shape
    return float(1.0 - np.sum(p * np.log(p)) / (n * np.log(K)))


def fit_gmm(X, K, seed):
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        n_init=N_INIT,
        random_state=seed,
        max_iter=MAX_ITER,
        reg_covar=REG_COVAR,
    )
    gmm.fit(X)
    return gmm


def main():
    ensure_dir(RESULTS_DIR)
    np.random.seed(SEED)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    z_INT = (scores["INT"] - scores["INT"].mean()) / scores["INT"].std(ddof=1)
    z_BE = (scores["BE"] - scores["BE"].mean()) / scores["BE"].std(ddof=1)
    X = np.column_stack([z_INT.values, z_BE.values])
    n = X.shape[0]

    # ------------------------------------------------------------------
    # 1. K=7 fit: posteriors + classification quality
    # ------------------------------------------------------------------
    gmm7 = fit_gmm(X, K7, SEED)
    post = gmm7.predict_proba(X)
    max_post = post.max(axis=1)

    qual = pd.DataFrame({
        "statistic": [
            "mean_max_posterior", "median_max_posterior", "SD_max_posterior",
            "pct_ge_0.90", "pct_ge_0.80", "pct_ge_0.70", "pct_lt_0.70",
        ],
        "value": [
            float(max_post.mean()), float(np.median(max_post)),
            float(max_post.std(ddof=1)),
            float((max_post >= 0.90).mean() * 100),
            float((max_post >= 0.80).mean() * 100),
            float((max_post >= 0.70).mean() * 100),
            float((max_post < 0.70).mean() * 100),
        ],
    })
    qual.to_csv(os.path.join(RESULTS_DIR, "k7_classification_quality.csv"), index=False)

    post_df = pd.DataFrame(
        post, columns=[f"post_profile_{j}" for j in range(K7)])
    post_df.insert(0, "respondent", scores.index.values)
    post_df.to_csv(os.path.join(RESULTS_DIR, "k7_posterior_probabilities.csv"), index=False)
    print(f"K=7 classification: mean max posterior = {max_post.mean():.4f}, "
          f"median = {np.median(max_post):.4f}")

    # ------------------------------------------------------------------
    # 2. K=7 bootstrap directional stability (200 replications)
    # Reference: profile means from the seed-42 K=7 fit, matched by
    # Hungarian assignment on the 2D profile-mean vectors.
    # ------------------------------------------------------------------
    ref_means = gmm7.means_  # (K, 2)
    counts = np.zeros(K7)
    for b in range(N_BOOT):
        idx = np.random.choice(n, size=n, replace=True)
        Xb = X[idx]
        gb = GaussianMixture(
            n_components=K7, covariance_type="full", n_init=50,
            random_state=SEED + b + 1, max_iter=MAX_ITER, reg_covar=REG_COVAR,
        )
        gb.fit(Xb)
        cost = np.linalg.norm(ref_means[:, None, :] - gb.means_[None, :, :], axis=2)
        _, col_ind = linear_sum_assignment(cost)
        boot_means = gb.means_[col_ind]  # matched to reference profiles
        counts += (boot_means[:, 0] > boot_means[:, 1]).astype(int)

    pct_int = counts / N_BOOT * 100
    stab = pd.DataFrame({
        "profile": [f"P{j}" for j in range(K7)],
        "pct_INT_gt_BE": pct_int,
        "pct_BE_gt_INT": 100 - pct_int,
    })
    stab.to_csv(os.path.join(RESULTS_DIR, "k7_stability.csv"), index=False)
    print("K=7 directional stability (% INT > BE): "
          + ", ".join(f"P{j}={pct_int[j]:.1f}" for j in range(K7)))

    # ------------------------------------------------------------------
    # 3. 5-fold cross-validation, K=2..7 (aggregate across folds)
    # ------------------------------------------------------------------
    rng = np.random.RandomState(SEED)
    fold_ids = rng.randint(0, N_FOLDS, size=n)

    rows = []
    for K in range(2, 8):
        bic_tr, ll_te = [], []
        for f in range(N_FOLDS):
            tr = X[fold_ids != f]
            te = X[fold_ids == f]
            gm = GaussianMixture(
                n_components=K, covariance_type="full", n_init=100,
                random_state=SEED, max_iter=MAX_ITER, reg_covar=REG_COVAR,
            )
            gm.fit(tr)
            n_params = int(K * 2 + K * 2 * 3 / 2.0 + (K - 1))
            bic_tr.append(-2 * gm.score(tr) * tr.shape[0] + n_params * np.log(tr.shape[0]))
            ll_te.append(gm.score(te) * te.shape[0])
        rows.append({
            "K": K,
            "mean_bic_train": float(np.mean(bic_tr)),
            "sd_bic_train": float(np.std(bic_tr, ddof=1)),
            "mean_ll_test": float(np.mean(ll_te)),
            "sd_ll_test": float(np.std(ll_te, ddof=1)),
        })
        print(f"CV K={K}: mean_ll_test={rows[-1]['mean_ll_test']:.2f} "
              f"(sd {rows[-1]['sd_ll_test']:.2f})")

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "cv5_aggregated_by_k.csv"), index=False)

    print("\nTask 27 complete:")
    for fn in ["k7_classification_quality.csv", "k7_posterior_probabilities.csv",
               "k7_stability.csv", "cv5_aggregated_by_k.csv"]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
