"""
Phase 14 — K=6 Profile Stability
==================================
Tests whether the K=6 profile structure is stable under bootstrap
resampling and split-sample analysis.

Same K=6 specification as the primary analysis:
- indicators: z_INT, z_BE (standardized within sample)
- covariance_type = "full"
- n_init = 1000
- random seed recorded explicitly

No K re-selection. No label invention. No interpretation.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORES_PATH = "results/02_measurement/construct_scores.csv"
PHASE04_DIR = "results/04_lpa_estimation"
RESULTS_DIR = "results/14_k6_stability"

K = 6
COVARIANCE_TYPE = "full"
N_INIT_PRIMARY = 1000  # primary K=6 reference fit
N_INIT_BOOTSTRAP = 100  # bootstrap fits (computationally feasible for B=200)
SEED = 42
B = 200  # bootstrap samples
SPLIT_SEED = 42


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def count_params_full(K, n_features):
    means = K * n_features
    covs = K * n_features * (n_features + 1) / 2.0
    weights = K - 1
    return int(means + covs + weights)


def safe_entropy(posterior):
    p = np.clip(posterior, 1e-15, 1.0)
    n, Kk = posterior.shape
    log_k = np.log(Kk)
    if log_k == 0:
        return 0.0
    return float(1.0 - np.sum(p * np.log(p)) / (n * log_k))


def fit_k6(X, seed, n_init=N_INIT_BOOTSTRAP):
    """Fit K=6 full-covariance GMM on standardized indicators X."""
    gmm = GaussianMixture(
        n_components=K,
        covariance_type=COVARIANCE_TYPE,
        n_init=n_init,
        random_state=seed,
        max_iter=500,
        reg_covar=1e-6,
    )
    gmm.fit(X)
    n = X.shape[0]
    k_params = count_params_full(K, X.shape[1])
    LL = float(gmm.score(X) * n)
    BIC = float(-2 * LL + k_params * np.log(n))
    AIC = float(-2 * LL + 2 * k_params)
    labels = gmm.predict(X)
    sizes = np.bincount(labels, minlength=K)
    return {
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "LL": LL,
        "BIC": BIC,
        "AIC": AIC,
        "entropy": safe_entropy(gmm.predict_proba(X)),
        "means": gmm.means_.copy(),  # (K, 2): [z_INT, z_BE]
        "sizes": sizes,
        "labels": labels,
    }


def match_profiles(ref_means, boot_means):
    """
    One-to-one matching of bootstrap profiles to reference profiles
    using minimum Euclidean distance on (mean z_INT, mean z_BE).
    Hungarian algorithm (scipy linear_sum_assignment).
    Returns (assignment, cost) where assignment[ref_profile] = boot_profile.
    """
    cost_mat = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost_mat[i, j] = np.linalg.norm(ref_means[i] - boot_means[j])
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    assignment = np.empty(K, dtype=int)
    assignment[row_ind] = col_ind
    total_cost = float(cost_mat[row_ind, col_ind].sum())
    return assignment, total_cost


def main():
    ensure_dir(RESULTS_DIR)
    rng = np.random.default_rng(SEED)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    N = scores.shape[0]
    INT = scores["INT"].values.astype(float)
    BE = scores["BE"].values.astype(float)

    # Original standardized indicators (same procedure as primary analysis)
    z_INT = (INT - INT.mean()) / INT.std(ddof=1)
    z_BE = (BE - BE.mean()) / BE.std(ddof=1)
    X_full = np.column_stack([z_INT, z_BE])

    # Original K=6 fit (same settings) — reference (use full n_init=1000)
    ref = fit_k6(X_full, SEED, n_init=N_INIT_PRIMARY)
    ref_means = ref["means"]

    # ==================================================================
    # 1. BOOTSTRAP STABILITY (B = 200)
    # ==================================================================
    boot_records = []
    successful = []
    for b in range(B):
        idx = rng.integers(0, N, size=N)  # sample with replacement
        Xb = X_full[idx]
        # Same preprocessing: standardize within bootstrap sample
        mu = Xb.mean(axis=0)
        sd = Xb.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xb_std = (Xb - mu) / sd
        res = fit_k6(Xb_std, seed=SEED + b + 1)
        rec = {
            "bootstrap": b,
            "converged": res["converged"],
            "n_iter": res["n_iter"],
            "log_likelihood": res["LL"],
            "BIC": res["BIC"],
            "matching_cost": np.nan,
        }
        # Store per-profile sizes and means (matched) in arrays
        if res["converged"]:
            assignment, cost = match_profiles(ref_means, res["means"])
            rec["matching_cost"] = cost
            matched_sizes = res["sizes"][assignment]
            matched_means = res["means"][assignment]
            successful.append({
                "bootstrap": b,
                "sizes": matched_sizes,
                "means": matched_means,  # (K,2) matched to original profile order
            })
            for p in range(K):
                rec[f"matched_size_p{p}"] = int(matched_sizes[p])
                rec[f"mean_zINT_p{p}"] = float(matched_means[p, 0])
                rec[f"mean_zBE_p{p}"] = float(matched_means[p, 1])
        boot_records.append(rec)
        if (b + 1) % 25 == 0:
            print(f"  bootstrap {b + 1}/{B} done ({len(successful)} converged)")

    boot_df = pd.DataFrame(boot_records)
    boot_df.to_csv(os.path.join(RESULTS_DIR, "bootstrap_results.csv"), index=False)

    n_success = len(successful)

    # ==================================================================
    # 3. PROFILE STABILITY across successful bootstraps
    # ==================================================================
    stab_rows = []
    for p in range(K):
        sizes_p = np.array([s["sizes"][p] for s in successful])
        int_p = np.array([s["means"][p, 0] for s in successful])
        be_p = np.array([s["means"][p, 1] for s in successful])
        diff_p = int_p - be_p
        stab_rows.append({
            "profile": p,
            "n_successful_bootstraps": n_success,
            "pct_bootstraps_profile_exists": 100.0,  # all K profiles exist in every fit
            "mean_matched_N": float(sizes_p.mean()),
            "SD_matched_N": float(sizes_p.std(ddof=1)),
            "min_matched_N": int(sizes_p.min()),
            "max_matched_N": int(sizes_p.max()),
            "mean_zINT": float(int_p.mean()),
            "SD_zINT": float(int_p.std(ddof=1)),
            "mean_zBE": float(be_p.mean()),
            "SD_zBE": float(be_p.std(ddof=1)),
            "mean_zINT_minus_zBE": float(diff_p.mean()),
            "SD_zINT_minus_zBE": float(diff_p.std(ddof=1)),
        })
    stab_df = pd.DataFrame(stab_rows)
    stab_df.to_csv(os.path.join(RESULTS_DIR, "profile_stability.csv"), index=False)

    # ==================================================================
    # 4. CONFIGURATION STABILITY (sign of INT - BE per profile)
    # ==================================================================
    cfg_rows = []
    for p in range(K):
        diffs = np.array([s["means"][p, 0] - s["means"][p, 1] for s in successful])
        n_pos = int((diffs > 0).sum())
        n_neg = int((diffs < 0).sum())
        n_zero = int((diffs == 0).sum())
        cfg_rows.append({
            "profile": p,
            "n_bootstraps": n_success,
            "n_INT_gt_BE": n_pos,
            "n_BE_gt_INT": n_neg,
            "n_exactly_equal": n_zero,
            "pct_INT_gt_BE": n_pos / n_success * 100,
            "pct_BE_gt_INT": n_neg / n_success * 100,
        })
    pd.DataFrame(cfg_rows).to_csv(
        os.path.join(RESULTS_DIR, "configuration_stability.csv"), index=False
    )

    # ==================================================================
    # 5. SPLIT-SAMPLE VALIDATION
    # ==================================================================
    split_rng = np.random.default_rng(SPLIT_SEED)
    perm = split_rng.permutation(N)
    half = N // 2
    idx1, idx2 = perm[:half], perm[half:]

    halves = {}
    for name, idx in [("half_1", idx1), ("half_2", idx2)]:
        Xh = X_full[idx]
        mu = Xh.mean(axis=0)
        sd = Xh.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xh_std = (Xh - mu) / sd
        halves[name] = fit_k6(Xh_std, seed=SEED)

    split_rows = []
    for name, res in halves.items():
        for p in range(K):
            split_rows.append({
                "half": name,
                "n_respondents": len(idx1) if name == "half_1" else len(idx2),
                "profile": p,
                "log_likelihood": res["LL"],
                "BIC": res["BIC"],
                "AIC": res["AIC"],
                "entropy": res["entropy"],
                "size": int(res["sizes"][p]),
                "mean_zINT": float(res["means"][p, 0]),
                "mean_zBE": float(res["means"][p, 1]),
            })
    split_df = pd.DataFrame(split_rows)

    # Match half_2 profiles to half_1 profiles
    assignment, cost = match_profiles(halves["half_1"]["means"], halves["half_2"]["means"])
    match_rows = []
    for p1 in range(K):
        p2 = assignment[p1]
        m1 = halves["half_1"]["means"][p1]
        m2 = halves["half_2"]["means"][p2]
        match_rows.append({
            "half_1_profile": p1,
            "half_2_matched_profile": int(p2),
            "half_1_size": int(halves["half_1"]["sizes"][p1]),
            "half_2_size": int(halves["half_2"]["sizes"][p2]),
            "size_difference": int(halves["half_1"]["sizes"][p1] - halves["half_2"]["sizes"][p2]),
            "mean_zINT_difference": float(m1[0] - m2[0]),
            "mean_zBE_difference": float(m1[1] - m2[1]),
            "euclidean_distance": float(np.linalg.norm(m1 - m2)),
        })
    match_df = pd.DataFrame(match_rows)
    match_df["matching_cost_total"] = cost

    split_df.to_csv(os.path.join(RESULTS_DIR, "split_sample_results.csv"), index=False)
    match_df.to_csv(os.path.join(RESULTS_DIR, "split_sample_matching.csv"), index=False)

    # ==================================================================
    # 6. CEILING / BOUNDARY CHECK (original data)
    # ==================================================================
    kdir = os.path.join(PHASE04_DIR, f"K_{K}")
    post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
    labels_orig = post_df["assigned_class"].values

    bound_rows = []
    for p in range(K):
        mask = labels_orig == p
        int_p = INT[mask]
        be_p = BE[mask]
        row = {
            "profile": p,
            "N": int(mask.sum()),
            "INT_variance": float(int_p.var(ddof=1)),
            "BE_variance": float(be_p.var(ddof=1)),
            "INT_min": float(int_p.min()),
            "INT_max": float(int_p.max()),
            "BE_min": float(be_p.min()),
            "BE_max": float(be_p.max()),
        }
        if p == 0:
            max_int = int_p.max()
            row["prop_INT_at_max"] = float((int_p == max_int).mean())
        bound_rows.append(row)
    pd.DataFrame(bound_rows).to_csv(
        os.path.join(RESULTS_DIR, "boundary_diagnostics.csv"), index=False
    )

    # ==================================================================
    # 7. MASTER STABILITY FILE
    # ==================================================================
    master_rows = [
        {"metric": "K", "value": K},
        {"metric": "N", "value": N},
        {"metric": "n_bootstrap", "value": B},
        {"metric": "n_successful_bootstraps", "value": n_success},
        {"metric": "bootstrap_seed_base", "value": SEED},
        {"metric": "n_init_primary", "value": N_INIT_PRIMARY},
        {"metric": "n_init_bootstrap", "value": N_INIT_BOOTSTRAP},
        {"metric": "covariance_type", "value": COVARIANCE_TYPE},
        {"metric": "original_LL", "value": ref["LL"]},
        {"metric": "original_BIC", "value": ref["BIC"]},
        {"metric": "bootstrap_mean_LL", "value": float(boot_df.loc[boot_df["converged"], "log_likelihood"].mean())},
        {"metric": "bootstrap_SD_LL", "value": float(boot_df.loc[boot_df["converged"], "log_likelihood"].std(ddof=1))},
        {"metric": "bootstrap_mean_matching_cost", "value": float(boot_df.loc[boot_df["converged"], "matching_cost"].mean())},
        {"metric": "split_sample_seed", "value": SPLIT_SEED},
        {"metric": "split_half_1_n", "value": len(idx1)},
        {"metric": "split_half_2_n", "value": len(idx2)},
        {"metric": "split_matching_cost", "value": cost},
    ]
    pd.DataFrame(master_rows).to_csv(
        os.path.join(RESULTS_DIR, "k6_stability_master.csv"), index=False
    )

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = f"""# Phase 14 — K=6 Profile Stability

## Purpose
Test stability of the K=6 profile structure under bootstrap resampling
and split-sample analysis. No K re-selection, no interpretation.

## Bootstrap Procedure
- B = {B} bootstrap samples
- Each sample: N = {N} respondents sampled WITH replacement
- Indicators: z_INT, z_BE standardized WITHIN each bootstrap sample
  (same preprocessing procedure as the primary analysis)
- Model: GaussianMixture(n_components={K}, covariance_type="{COVARIANCE_TYPE}",
  n_init_primary={N_INIT_PRIMARY}, n_init_bootstrap={N_INIT_BOOTSTRAP}, max_iter=500, reg_covar=1e-6)
- Random seeds: SEED + b + 1 for bootstrap b (base SEED = {SEED})
- Records: convergence, log likelihood, BIC, matched profile sizes,
  matched profile means

## Profile Matching
- Bootstrap profiles matched to original K=6 profiles by minimum
  Euclidean distance on the 2D vector (mean z_INT, mean z_BE)
- One-to-one assignment via the Hungarian algorithm
  (scipy.optimize.linear_sum_assignment)
- Matching cost recorded per bootstrap sample

## Split-Sample Procedure
- Fixed seed = {SPLIT_SEED}; random permutation split into halves
  (n = {len(idx1)} and {len(idx2)})
- Same K=6 model fitted independently to each half
- Profiles matched between halves by the same minimum-distance method

## Boundary Diagnostics
- Per-profile INT/BE variance, min, max from the ORIGINAL data
- Profile 0: proportion of observations with INT at its maximum
  observed value (numerical evidence only)

## Output Files
- bootstrap_results.csv — per-bootstrap convergence, LL, BIC, matched sizes/means
- profile_stability.csv — stability statistics per original profile
- configuration_stability.csv — sign of INT-BE across bootstraps
- split_sample_results.csv — per-half fit results
- split_sample_matching.csv — cross-half profile matching
- boundary_diagnostics.csv — ceiling/boundary evidence
- k6_stability_master.csv — key stability metrics

## No Modifications
Primary K=6 results untouched. No dataset changes. No plots.
"""
    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write(readme)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 14 — K=6 STABILITY COMPLETE")
    print("=" * 60)
    print(f"B = {B}, successful = {n_success}")
    print(f"Original: LL = {ref['LL']:.2f}, BIC = {ref['BIC']:.2f}")
    print("\nProfile stability (matched across bootstraps):")
    print(stab_df[["profile", "mean_matched_N", "SD_matched_N",
                   "mean_zINT_minus_zBE", "SD_zINT_minus_zBE"]].to_string(index=False))
    print("\nConfiguration stability:")
    print(pd.DataFrame(cfg_rows).to_string(index=False))
    print(f"\nSplit-sample matching cost: {cost:.4f}")
    print("\nOutputs:")
    for fn in ["bootstrap_results.csv", "profile_stability.csv",
               "configuration_stability.csv", "split_sample_results.csv",
               "split_sample_matching.csv", "boundary_diagnostics.csv",
               "k6_stability_master.csv", "README.md"]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
