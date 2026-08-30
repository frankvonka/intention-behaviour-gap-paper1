"""
Phase 17 — Alternative Model Validation
=========================================
Sensitivity analysis: tests whether the central INT-BE profile
structure is robust to alternative statistical specifications.

Specifications tested:
1. Covariance type: full / diag / spherical (10 construct indicators)
2. Score representation: mean scores vs sklearn FactorAnalysis scores
3. Random seeds: 1..5 (primary 2-indicator specification)

No model selection. No interpretation. No modification of previous
phases.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FactorAnalysis

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORES_PATH = "results/02_measurement/construct_scores.csv"
PHASE03_DIR = "results/03_gap_analysis"
PHASE04_DIR = "results/04_lpa_estimation"
DATA_PATH = "data.xls"
RESULTS_DIR = "results/17_alternative_model_validation"

K = 6
N_INIT = 1000
SEED = 42
SEEDS = [1, 2, 3, 4, 5]

# Indicators as enumerated in the phase specification
INDICATORS_10 = ["INT", "BE", "ATT", "CON", "SNO", "COVID", "PU", "PEU", "PO", "PRI"]

CONSTRUCT_ITEMS = {
    "ATT":  ["ATT1", "ATT2", "ATT3"],
    "CON":  ["CON1", "CON2", "CON3"],
    "SNO":  ["SNO1", "SNO2", "SNO3"],
    "COVID": ["COVID1", "COVID2", "COVID3"],
    "INT":  ["INT1", "INT2", "INT3"],
    "BE":   ["BE1", "BE2", "BE3", "BE4"],
    "PU":   ["PU1", "PU2", "PU3"],
    "PEU":  ["PEU1", "PEU2"],
    "PO":   ["PO1", "PO2"],
    "PRI":  ["PRI1", "PRI2", "PRI3"],
}


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def count_params(K, d, cov_type):
    means = K * d
    if cov_type == "full":
        covs = K * d * (d + 1) / 2.0
    elif cov_type == "diag":
        covs = K * d
    elif cov_type == "spherical":
        covs = K
    else:
        covs = K * d
    return int(means + covs + (K - 1))


def safe_entropy(posterior):
    p = np.clip(posterior, 1e-15, 1.0)
    n, Kk = posterior.shape
    log_k = np.log(Kk)
    if log_k == 0:
        return 0.0
    return float(1.0 - np.sum(p * np.log(p)) / (n * log_k))


def fit_gmm(X, cov_type, seed, n_init=N_INIT):
    gmm = GaussianMixture(
        n_components=K,
        covariance_type=cov_type,
        n_init=n_init,
        random_state=seed,
        max_iter=500,
        reg_covar=1e-6,
    )
    gmm.fit(X)
    n, d = X.shape
    k_params = count_params(K, d, cov_type)
    LL = float(gmm.score(X) * n)
    return {
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "LL": LL,
        "AIC": float(-2 * LL + 2 * k_params),
        "BIC": float(-2 * LL + k_params * np.log(n)),
        "entropy": safe_entropy(gmm.predict_proba(X)),
        "labels": gmm.predict(X),
        "n_params": k_params,
    }


def profile_int_be_gap(labels, INT, BE, GAP):
    """Per-profile INT mean, BE mean, GAP mean from respondent scores."""
    rows = []
    for p in range(K):
        mask = labels == p
        rows.append({
            "profile": p,
            "N": int(mask.sum()),
            "INT_mean": float(INT[mask].mean()),
            "BE_mean": float(BE[mask].mean()),
            "GAP_mean": float(GAP[mask].mean()),
            "INT_minus_BE": float(INT[mask].mean() - BE[mask].mean()),
        })
    return pd.DataFrame(rows)


def match_to_reference(ref_int_be, alt_int_be):
    """
    Hungarian matching of alternative profiles to reference profiles
    on standardized (INT, BE) profile means.
    Returns assignment (alt_profile -> ref_profile) and distances.
    """
    # Standardize each dimension across profiles
    def std(m):
        mu = m.mean(axis=0)
        sd = m.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        return (m - mu) / sd

    ref_m = std(ref_int_be)
    alt_m = std(alt_int_be)
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = np.linalg.norm(alt_m[i] - ref_m[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = np.empty(K, dtype=int)
    assignment[row_ind] = col_ind
    distances = cost[row_ind, col_ind]
    return assignment, distances


def main():
    ensure_dir(RESULTS_DIR)
    np.random.seed(SEED)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    N = scores.shape[0]
    INT = scores["INT"].values.astype(float)
    BE = scores["BE"].values.astype(float)
    GAP = np.load(os.path.join(PHASE03_DIR, "gap_scores.npy"))

    # ==================================================================
    # 1. ORIGINAL K=6 REFERENCE
    # ==================================================================
    fit = pd.read_csv(os.path.join(PHASE04_DIR, "model_fit.csv"))
    frow = fit[fit["K"] == K].iloc[0]
    kdir = os.path.join(PHASE04_DIR, f"K_{K}")
    post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
    ref_labels = post_df["assigned_class"].values
    ref_table = profile_int_be_gap(ref_labels, INT, BE, GAP)

    ref_rows = [
        {"metric": "BIC", "value": float(frow["BIC"])},
        {"metric": "AIC", "value": float(frow["AIC"])},
        {"metric": "log_likelihood", "value": float(frow["log_likelihood"])},
        {"metric": "entropy", "value": float(frow["entropy"])},
        {"metric": "profile_sizes", "value": str(ref_table["N"].tolist())},
        {"metric": "profile_INT_means", "value": str(ref_table["INT_mean"].round(6).tolist())},
        {"metric": "profile_BE_means", "value": str(ref_table["BE_mean"].round(6).tolist())},
        {"metric": "profile_GAP_means", "value": str(ref_table["GAP_mean"].round(6).tolist())},
    ]
    pd.DataFrame(ref_rows).to_csv(
        os.path.join(RESULTS_DIR, "reference_k6.csv"), index=False
    )
    ref_table.to_csv(os.path.join(RESULTS_DIR, "reference_k6_profiles.csv"), index=False)
    ref_int_be = ref_table[["INT_mean", "BE_mean"]].values

    # ==================================================================
    # 2. COVARIANCE SPECIFICATION SENSITIVITY (10 indicators)
    # ==================================================================
    X10 = scores[INDICATORS_10].values.astype(float)
    mu10 = X10.mean(axis=0)
    sd10 = X10.std(axis=0, ddof=1)
    X10_std = (X10 - mu10) / sd10

    cov_rows = []
    cov_profile_tables = {}
    for cov in ["full", "diag", "spherical"]:
        print(f"  covariance={cov} ...", flush=True)
        res = fit_gmm(X10_std, cov, SEED)
        table = profile_int_be_gap(res["labels"], INT, BE, GAP)
        cov_profile_tables[cov] = (res, table)
        cov_rows.append({
            "specification": f"covariance_{cov}",
            "indicators": ",".join(INDICATORS_10),
            "converged": res["converged"],
            "n_iter": res["n_iter"],
            "log_likelihood": res["LL"],
            "AIC": res["AIC"],
            "BIC": res["BIC"],
            "entropy": res["entropy"],
            "n_params": res["n_params"],
            "profile_sizes": str(table["N"].tolist()),
            "profile_INT_means": str(table["INT_mean"].round(6).tolist()),
            "profile_BE_means": str(table["BE_mean"].round(6).tolist()),
            "profile_GAP_means": str(table["GAP_mean"].round(6).tolist()),
        })
    pd.DataFrame(cov_rows).to_csv(
        os.path.join(RESULTS_DIR, "covariance_sensitivity.csv"), index=False
    )

    # ==================================================================
    # 3. SCORE REPRESENTATION SENSITIVITY (INT, BE indicators)
    # ==================================================================
    data = pd.read_excel(DATA_PATH, sheet_name=0)

    def zscore(v):
        return (v - v.mean()) / v.std(ddof=1)

    # A. Mean-score representation (primary specification)
    X_mean = np.column_stack([zscore(INT), zscore(BE)])
    print("  representation=mean ...", flush=True)
    res_mean = fit_gmm(X_mean, "full", SEED)
    table_mean = profile_int_be_gap(res_mean["labels"], INT, BE, GAP)

    # B. FactorAnalysis scores for INT and BE constructs
    fa_scores = {}
    for c in ["INT", "BE"]:
        fa = FactorAnalysis(n_components=1, random_state=SEED)
        fs = fa.fit_transform(data[CONSTRUCT_ITEMS[c]].values.astype(float)).flatten()
        if np.corrcoef(fs, scores[c].values)[0, 1] < 0:
            fs = -fs  # align sign with mean score
        fa_scores[c] = fs
    X_fa = np.column_stack([zscore(fa_scores["INT"]), zscore(fa_scores["BE"])])
    print("  representation=factor ...", flush=True)
    res_fa = fit_gmm(X_fa, "full", SEED)
    table_fa = profile_int_be_gap(res_fa["labels"], INT, BE, GAP)

    # Match factor-score profiles to mean-score profiles on (INT, BE)
    assignment_fa, dist_fa = match_to_reference(
        table_mean[["INT_mean", "BE_mean"]].values,
        table_fa[["INT_mean", "BE_mean"]].values,
    )
    table_fa_matched = table_fa.iloc[np.argsort(assignment_fa)].reset_index(drop=True)

    score_rows = []
    for name, res, table in [("mean_scores", res_mean, table_mean),
                             ("factor_scores", res_fa, table_fa_matched)]:
        score_rows.append({
            "specification": name,
            "converged": res["converged"],
            "log_likelihood": res["LL"],
            "AIC": res["AIC"],
            "BIC": res["BIC"],
            "entropy": res["entropy"],
            "profile_sizes": str(table["N"].tolist()),
            "profile_INT_means": str(table["INT_mean"].round(6).tolist()),
            "profile_BE_means": str(table["BE_mean"].round(6).tolist()),
            "profile_GAP_means": str(table["GAP_mean"].round(6).tolist()),
        })
    pd.DataFrame(score_rows).to_csv(
        os.path.join(RESULTS_DIR, "score_representation_sensitivity.csv"), index=False
    )

    # ==================================================================
    # 4. RANDOM-SEED SENSITIVITY (primary 2-indicator specification)
    # ==================================================================
    seed_rows = []
    seed_tables = {}
    for s in SEEDS:
        print(f"  seed={s} ...", flush=True)
        res = fit_gmm(X_mean, "full", s)
        table = profile_int_be_gap(res["labels"], INT, BE, GAP)
        seed_tables[s] = (res, table)
        seed_rows.append({
            "seed": s,
            "n_init": N_INIT,
            "converged": res["converged"],
            "log_likelihood": res["LL"],
            "BIC": res["BIC"],
            "AIC": res["AIC"],
            "entropy": res["entropy"],
            "profile_sizes": str(table["N"].tolist()),
        })
    pd.DataFrame(seed_rows).to_csv(
        os.path.join(RESULTS_DIR, "seed_sensitivity.csv"), index=False
    )

    # ==================================================================
    # 5. GAP-DIRECTION REPLICATION
    # ==================================================================
    gap_dir_rows = []
    # Reference
    for _, r in ref_table.iterrows():
        gap_dir_rows.append({
            "specification": "reference_k6",
            "profile": int(r["profile"]),
            "INT_minus_BE": float(r["INT_minus_BE"]),
            "GAP_mean": float(r["GAP_mean"]),
        })
    # Covariance alternatives
    for cov, (res, table) in cov_profile_tables.items():
        assignment, _ = match_to_reference(ref_int_be, table[["INT_mean", "BE_mean"]].values)
        for p in range(K):
            r = table.iloc[p]
            gap_dir_rows.append({
                "specification": f"covariance_{cov}",
                "profile": int(assignment[p]),  # matched reference profile id
                "INT_minus_BE": float(r["INT_minus_BE"]),
                "GAP_mean": float(r["GAP_mean"]),
            })
    # Factor-score representation (already matched to mean-score order)
    for _, r in table_fa_matched.iterrows():
        gap_dir_rows.append({
            "specification": "factor_scores",
            "profile": int(r["profile"]),
            "INT_minus_BE": float(r["INT_minus_BE"]),
            "GAP_mean": float(r["GAP_mean"]),
        })
    # Seeds
    for s, (res, table) in seed_tables.items():
        assignment, _ = match_to_reference(ref_int_be, table[["INT_mean", "BE_mean"]].values)
        for p in range(K):
            r = table.iloc[p]
            gap_dir_rows.append({
                "specification": f"seed_{s}",
                "profile": int(assignment[p]),
                "INT_minus_BE": float(r["INT_minus_BE"]),
                "GAP_mean": float(r["GAP_mean"]),
            })
    pd.DataFrame(gap_dir_rows).to_csv(
        os.path.join(RESULTS_DIR, "gap_direction_sensitivity.csv"), index=False
    )

    # ==================================================================
    # 6. PROFILE-MATCHING STABILITY
    # ==================================================================
    match_rows = []
    for cov, (res, table) in cov_profile_tables.items():
        assignment, distances = match_to_reference(ref_int_be, table[["INT_mean", "BE_mean"]].values)
        for p in range(K):
            match_rows.append({
                "specification": f"covariance_{cov}",
                "alt_profile": p,
                "matched_reference_profile": int(assignment[p]),
                "euclidean_distance_std_INT_BE": float(distances[p]),
            })
    assignment_fa2, dist_fa2 = match_to_reference(ref_int_be, table_fa[["INT_mean", "BE_mean"]].values)
    for p in range(K):
        match_rows.append({
            "specification": "factor_scores",
            "alt_profile": p,
            "matched_reference_profile": int(assignment_fa2[p]),
            "euclidean_distance_std_INT_BE": float(dist_fa2[p]),
        })
    for s, (res, table) in seed_tables.items():
        assignment, distances = match_to_reference(ref_int_be, table[["INT_mean", "BE_mean"]].values)
        for p in range(K):
            match_rows.append({
                "specification": f"seed_{s}",
                "alt_profile": p,
                "matched_reference_profile": int(assignment[p]),
                "euclidean_distance_std_INT_BE": float(distances[p]),
            })
    pd.DataFrame(match_rows).to_csv(
        os.path.join(RESULTS_DIR, "profile_matching_sensitivity.csv"), index=False
    )

    # ==================================================================
    # 7. MASTER FILE
    # ==================================================================
    master_rows = [
        {"specification": "reference_k6", "BIC": float(frow["BIC"]),
         "AIC": float(frow["AIC"]), "entropy": float(frow["entropy"]),
         "total_matching_distance": 0.0},
    ]
    for cov, (res, table) in cov_profile_tables.items():
        assignment, distances = match_to_reference(ref_int_be, table[["INT_mean", "BE_mean"]].values)
        master_rows.append({
            "specification": f"covariance_{cov}",
            "BIC": res["BIC"], "AIC": res["AIC"], "entropy": res["entropy"],
            "total_matching_distance": float(distances.sum()),
        })
    master_rows.append({
        "specification": "mean_scores", "BIC": res_mean["BIC"],
        "AIC": res_mean["AIC"], "entropy": res_mean["entropy"],
        "total_matching_distance": 0.0,
    })
    master_rows.append({
        "specification": "factor_scores", "BIC": res_fa["BIC"],
        "AIC": res_fa["AIC"], "entropy": res_fa["entropy"],
        "total_matching_distance": float(dist_fa2.sum()),
    })
    for s, (res, table) in seed_tables.items():
        assignment, distances = match_to_reference(ref_int_be, table[["INT_mean", "BE_mean"]].values)
        master_rows.append({
            "specification": f"seed_{s}", "BIC": res["BIC"],
            "AIC": res["AIC"], "entropy": res["entropy"],
            "total_matching_distance": float(distances.sum()),
        })
    pd.DataFrame(master_rows).to_csv(
        os.path.join(RESULTS_DIR, "phase17_master.csv"), index=False
    )

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = f"""# Phase 17 — Alternative Model Validation

## Purpose
Sensitivity analysis of the K=6 INT-BE profile structure under
alternative statistical specifications. No model selection, no
interpretation.

## Specifications Tested
1. **Covariance type** (full / diag / spherical), K=6, indicators =
   all 10 constructs (INT, BE, ATT, CON, SNO, COVID, PU, PEU, PO, PRI),
   standardized across respondents
2. **Score representation** (arithmetic mean scores vs sklearn
   FactorAnalysis 1-component scores for INT and BE), K=6 full
   covariance on the 2 standardized indicators
3. **Random seeds** 1, 2, 3, 4, 5 — primary 2-indicator (z_INT, z_BE)
   full-covariance specification

## Parameters
- K = {K} (fixed; no re-selection)
- n_init = {N_INIT} per fit
- Base seed = {SEED}; seed list = {SEEDS}
- max_iter = 500, reg_covar = 1e-6
- Scaling: respondent-level z-standardization per indicator

## Profile Matching
- Alternative profiles matched to the primary K=6 reference by
  minimum Euclidean distance on standardized (INT, BE) profile means
- One-to-one assignment via the Hungarian algorithm
  (scipy.optimize.linear_sum_assignment)

## Output Files
- reference_k6.csv / reference_k6_profiles.csv — frozen primary solution
- covariance_sensitivity.csv — full/diag/spherical fits
- score_representation_sensitivity.csv — mean vs factor scores
- seed_sensitivity.csv — seeds 1..5
- gap_direction_sensitivity.csv — signed INT-BE per matched profile
- profile_matching_sensitivity.csv — matched profile + distance
- phase17_master.csv — summary across all specifications

## No Modifications
Previous phase outputs untouched. Dataset unchanged. No plots.
"""
    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write(readme)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 17 — ALTERNATIVE MODEL VALIDATION COMPLETE")
    print("=" * 60)
    print("\nMaster summary:")
    print(pd.DataFrame(master_rows).to_string(index=False))
    print("\nOutputs:")
    for fn in ["reference_k6.csv", "reference_k6_profiles.csv",
               "covariance_sensitivity.csv", "score_representation_sensitivity.csv",
               "seed_sensitivity.csv", "gap_direction_sensitivity.csv",
               "profile_matching_sensitivity.csv", "phase17_master.csv", "README.md"]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
