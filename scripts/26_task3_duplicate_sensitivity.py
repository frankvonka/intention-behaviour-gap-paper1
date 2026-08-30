"""
Reviewer-fix pass — Task 3: Duplicate-row sensitivity (N=1124 excluding 42 dups).

Procedure:
  1. Load data.xls; identify the 42 all-column-duplicate rows.
  2. Recompute INT-BE Pearson r on the 1124 unique rows.
  3. Recompute z(INT)-z(BE) GAP stats on the 1124 unique rows.
  4. Re-estimate K=6 LPA on the 1124 unique rows (same primary spec).
  5. Compare all three to the frozen values.
Saves results/paper1_strengthening/duplicate_sensitivity.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

DATA_PATH = "data.xls"
SCORES_PATH = "results/02_measurement/construct_scores.csv"
RESULTS_DIR = "results/paper1_strengthening"
SEED = 42
N_INIT = 1000
MAX_ITER = 500
REG_COVAR = 1e-6
K_PRIMARY = 6
CONSTRUCTS = {
    "INT": ["INT1", "INT2", "INT3"],
    "BE":  ["BE1", "BE2", "BE3", "BE4"],
}


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def pearson_ci(r, n, alpha=0.05):
    if n <= 3:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2.0)
    return (float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se)))


def safe_entropy(posterior):
    p = np.clip(posterior, 1e-15, 1.0)
    n, K = posterior.shape
    log_k = np.log(K)
    if log_k == 0:
        return 0.0
    return float(1.0 - np.sum(p * np.log(p)) / (n * log_k))


def count_params_full(K, n_features):
    return int(K * n_features + K * n_features * (n_features + 1) / 2.0 + (K - 1))


def main():
    ensure_dir(RESULTS_DIR)
    np.random.seed(SEED)

    # ---- Load and identify duplicates ----
    df = pd.read_excel(DATA_PATH, sheet_name=0).copy()
    N_full = df.shape[0]
    dup_mask = df.duplicated(keep="first")  # mark all but first as duplicate
    n_dup = int(dup_mask.sum())
    df_u = df.loc[~dup_mask].copy()
    N_unique = df_u.shape[0]

    # ---- Recompute INT/BE means on the unique sample ----
    int_u = df_u[CONSTRUCTS["INT"]].astype(float).mean(axis=1)
    be_u = df_u[CONSTRUCTS["BE"]].astype(float).mean(axis=1)
    r_u, p_u = stats.pearsonr(int_u, be_u)
    n_valid_u = int(min(int_u.notna().sum(), be_u.notna().sum()))
    ci_lo_u, ci_hi_u = pearson_ci(r_u, n_valid_u)
    cov_u = float(((int_u - int_u.mean()) * (be_u - be_u.mean())).sum() / (n_valid_u - 1))

    # ---- GAP on the unique sample ----
    z_INT = (int_u - int_u.mean()) / int_u.std(ddof=1)
    z_BE = (be_u - be_u.mean()) / be_u.std(ddof=1)
    gap_u = z_INT - z_BE
    n_int_gt = int((gap_u > 0).sum())
    n_be_gt = int((gap_u < 0).sum())
    n_eq = int((gap_u == 0).sum())
    gap_stats = {
        "mean": float(gap_u.mean()),
        "SD": float(gap_u.std(ddof=1)),
        "median": float(gap_u.median()),
        "min": float(gap_u.min()),
        "max": float(gap_u.max()),
        "n_INT_gt_BE": n_int_gt,
        "n_BE_gt_INT": n_be_gt,
        "n_eq": n_eq,
        "pct_INT_gt_BE": n_int_gt / len(gap_u) * 100,
        "pct_BE_gt_INT": n_be_gt / len(gap_u) * 100,
    }

    # ---- LPA K=6 on the unique sample (primary spec) ----
    X = np.column_stack([z_INT.values, z_BE.values])
    gmm = GaussianMixture(
        n_components=K_PRIMARY,
        covariance_type="full",
        n_init=N_INIT,
        random_state=SEED,
        max_iter=MAX_ITER,
        reg_covar=REG_COVAR,
    )
    gmm.fit(X)
    n = X.shape[0]
    k_params = count_params_full(K_PRIMARY, X.shape[1])
    LL = float(gmm.score(X) * n)
    BIC = float(-2 * LL + k_params * np.log(n))
    AIC = float(-2 * LL + 2 * k_params)
    entropy = safe_entropy(gmm.predict_proba(X))
    labels = gmm.predict(X)
    sizes = np.bincount(labels, minlength=K_PRIMARY)
    post_max = gmm.predict_proba(X).max(axis=1)
    lpa_stats = {
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "log_likelihood": LL,
        "AIC": AIC,
        "BIC": BIC,
        "entropy": entropy,
        "n_params": k_params,
        "class_sizes": str(sizes.tolist()),
        "min_class_N": int(sizes.min()),
        "mean_max_posterior": float(post_max.mean()),
        "pct_post_lt_0_70": float((post_max < 0.70).mean() * 100),
    }
    # per-profile INT, BE, GAP
    profile_rows = []
    for j in range(K_PRIMARY):
        m = labels == j
        profile_rows.append({
            "profile": j,
            "N": int(m.sum()),
            "mean_INT": float(int_u[m].mean()),
            "mean_BE": float(be_u[m].mean()),
            "mean_INT_minus_BE": float(int_u[m].mean() - be_u[m].mean()),
            "mean_z_INT": float(z_INT[m].mean()),
            "mean_z_BE": float(z_BE[m].mean()),
            "mean_gap": float(gap_u[m].mean()),
        })
    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv(os.path.join(RESULTS_DIR, "duplicate_sensitivity_profiles.csv"), index=False)

    # ---- Frozen values (from results/paper1_final + temp.md) ----
    FROZEN = {
        "N_full": 1166,
        "n_duplicates": 42,
        "N_unique": 1124,
        "r": 0.651458,
        "p": 8.83e-142,
        "ci_lo": 0.6171,
        "ci_hi": 0.6833,
        "gap_mean": -4.875e-16,
        "gap_SD": 0.834916,
        "gap_pct_INT_gt_BE": 44.94,
        "gap_pct_BE_gt_INT": 55.06,
        "K6_BIC": 2129.17,
        "K6_LL": -941.01,
        "K6_class_sizes_full": [124, 377, 262, 90, 54, 259],
    }

    out_rows = [
        {"metric": "N_full_sample", "frozen": FROZEN["N_full"], "unique_sample": N_full, "match": N_full == FROZEN["N_full"]},
        {"metric": "n_duplicates_removed", "frozen": FROZEN["n_duplicates"], "unique_sample": n_dup, "match": n_dup == FROZEN["n_duplicates"]},
        {"metric": "N_unique_sample", "frozen": FROZEN["N_unique"], "unique_sample": N_unique, "match": N_unique == FROZEN["N_unique"]},
        {"metric": "pearson_r_INT_BE", "frozen": FROZEN["r"], "unique_sample": r_u, "match": bool(np.isclose(r_u, FROZEN["r"], atol=5e-4))},
        {"metric": "pearson_p_INT_BE", "frozen": FROZEN["p"], "unique_sample": p_u, "match": "loose (different N)"},
        {"metric": "pearson_CI95_lower", "frozen": FROZEN["ci_lo"], "unique_sample": ci_lo_u, "match": bool(np.isclose(ci_lo_u, FROZEN["ci_lo"], atol=5e-3))},
        {"metric": "pearson_CI95_upper", "frozen": FROZEN["ci_hi"], "unique_sample": ci_hi_u, "match": bool(np.isclose(ci_hi_u, FROZEN["ci_hi"], atol=5e-3))},
        {"metric": "gap_mean", "frozen": FROZEN["gap_mean"], "unique_sample": gap_stats["mean"], "match": bool(abs(gap_stats["mean"]) < 1e-10)},
        {"metric": "gap_SD", "frozen": FROZEN["gap_SD"], "unique_sample": gap_stats["SD"], "match": bool(np.isclose(gap_stats["SD"], FROZEN["gap_SD"], atol=5e-3))},
        {"metric": "pct_INT_gt_BE", "frozen": FROZEN["gap_pct_INT_gt_BE"], "unique_sample": gap_stats["pct_INT_gt_BE"], "match": bool(abs(gap_stats["pct_INT_gt_BE"] - FROZEN["gap_pct_INT_gt_BE"]) < 1.0)},
        {"metric": "pct_BE_gt_INT", "frozen": FROZEN["gap_pct_BE_gt_INT"], "unique_sample": gap_stats["pct_BE_gt_INT"], "match": bool(abs(gap_stats["pct_BE_gt_INT"] - FROZEN["gap_pct_BE_gt_INT"]) < 1.0)},
        {"metric": "K6_BIC", "frozen": FROZEN["K6_BIC"], "unique_sample": BIC, "match": bool(np.isclose(BIC, FROZEN["K6_BIC"], atol=5))},
        {"metric": "K6_log_likelihood", "frozen": FROZEN["K6_LL"], "unique_sample": LL, "match": bool(np.isclose(LL, FROZEN["K6_LL"], atol=5))},
        {"metric": "K6_entropy", "frozen": 1.1418, "unique_sample": entropy, "match": bool(np.isclose(entropy, 1.1418, atol=0.05))},
        {"metric": "K6_min_class_N", "frozen": 54, "unique_sample": lpa_stats["min_class_N"], "match": "compare profile tables"},
        {"metric": "K6_mean_max_posterior", "frozen": 0.8917, "unique_sample": lpa_stats["mean_max_posterior"], "match": bool(np.isclose(lpa_stats["mean_max_posterior"], 0.8917, atol=0.05))},
    ]
    out = pd.DataFrame(out_rows)
    out.to_csv(os.path.join(RESULTS_DIR, "duplicate_sensitivity.csv"), index=False)

    print("=" * 70)
    print("TASK 3 — DUPLICATE-ROW SENSITIVITY (N_unique = 1124)")
    print("=" * 70)
    print(f"Full sample: N = {N_full} ({n_dup} duplicates identified, N_unique = {N_unique})")
    print()
    print("INT-BE correlation (unique sample):")
    print(f"  r = {r_u:.6f}   (frozen {FROZEN['r']})")
    print(f"  p = {p_u:.6e}  (frozen {FROZEN['p']})")
    print(f"  95% CI = [{ci_lo_u:.4f}, {ci_hi_u:.4f}]  (frozen [{FROZEN['ci_lo']}, {FROZEN['ci_hi']}])")
    print(f"  cov = {cov_u:.6f}")
    print()
    print("GAP (unique sample):")
    for k, v in gap_stats.items():
        print(f"  {k}: {v}")
    print()
    print("K=6 LPA (unique sample, full covariance, n_init=1000, seed=42):")
    for k, v in lpa_stats.items():
        print(f"  {k}: {v}")
    print()
    print("Per-profile (unique sample):")
    print(profile_df.to_string(index=False))
    print()
    print("Per-metric match (frozen vs unique):")
    print(out.to_string(index=False))
    print(f"\nSaved: {os.path.join(RESULTS_DIR, 'duplicate_sensitivity.csv')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'duplicate_sensitivity_profiles.csv')}")


if __name__ == "__main__":
    main()
