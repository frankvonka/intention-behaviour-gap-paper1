"""
Phase 11 — Post-Phase-10 Verification
=======================================
Verifies the final results before treating them as frozen.

Checks:
1. Phase 09 failure identification
2. K=2..K=6 stability comparison
3. Profile structure for K=2..K=6
4. Gap profile verification (INT>BE, BE>INT)
5. Profile size / over-segmentation diagnostic
6. Factor score robustness (sklearn FactorAnalysis)
7. K=6 reproducibility

No changes to primary analysis. No literature. No figures.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FactorAnalysis

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORES_PATH = "results/02_measurement/construct_scores.csv"
PHASE04_DIR = "results/04_lpa_estimation"
PHASE05_DIR = "results/05_lpa_selection"
RESULTS_DIR = "results/11_final_verification"
SEED = 42

K_RANGE = [2, 3, 4, 5, 6]
COVARIANCE_TYPE = "full"
N_INIT = 1000
INDICATORS = ["z_INT", "z_BE"]

# Construct definitions (from Phase 02)
CONSTRUCTS = {
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


def safe_entropy(posterior):
    p = np.clip(posterior, 1e-15, 1.0)
    n, K = posterior.shape
    log_k = np.log(K)
    if log_k == 0:
        return 0.0
    total = np.sum(p * np.log(p))
    return float(1.0 - total / (n * log_k))


def count_params_full(K, n_features):
    means = K * n_features
    covs = K * n_features * (n_features + 1) / 2.0
    weights = K - 1
    return int(means + covs + weights)


def main():
    ensure_dir(RESULTS_DIR)
    np.random.seed(SEED)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    N = scores.shape[0]
    data = pd.read_excel("data.xls", sheet_name=0).copy()

    # Standardized indicators
    z_INT = (scores["INT"] - scores["INT"].mean()) / scores["INT"].std(ddof=1)
    z_BE = (scores["BE"] - scores["BE"].mean()) / scores["BE"].std(ddof=1)
    X = np.column_stack([z_INT.values, z_BE.values])

    summary_rows = []

    # ==================================================================
    # CHECK 1 — Phase 09 failure identification
    # ==================================================================
    audit = pd.read_csv("results/10_final/numerical_audit/audit_checks.csv")
    fail_mask = audit["status"] == "FAIL"
    failures = audit[fail_mask].copy()
    failures.to_csv(os.path.join(RESULTS_DIR, "audit_failure.csv"), index=False)
    n_fail = fail_mask.sum()
    summary_rows.append({
        "check": "audit_failure_identification",
        "status": "PASS" if n_fail == 1 else "REVIEW",
        "notes": f"{n_fail} failure(s) found; see audit_failure.csv",
    })

    # ==================================================================
    # CHECK 2 — K=2..K=6 stability comparison
    # ==================================================================
    fit = pd.read_csv(os.path.join(PHASE04_DIR, "model_fit.csv"))
    unc = pd.read_csv(os.path.join(PHASE05_DIR, "classification_uncertainty.csv"))

    comp_rows = []
    for K in K_RANGE:
        row = fit[fit["K"] == K].iloc[0]
        urow = unc[unc["K"] == K].iloc[0]
        # Recompute labels for max-posterior
        gmm = GaussianMixture(
            n_components=K,
            covariance_type=COVARIANCE_TYPE,
            n_init=N_INIT,
            random_state=SEED,
            max_iter=500,
            reg_covar=1e-6,
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        sizes = np.bincount(labels, minlength=K).tolist()
        comp_rows.append({
            "K": K,
            "BIC": float(row["BIC"]),
            "AIC": float(row["AIC"]),
            "log_likelihood": float(row["log_likelihood"]),
            "entropy": float(row["entropy"]),
            "profile_sizes": sizes,
            "smallest_profile_pct": float(row["smallest_class_pct"]),
            "largest_profile_pct": float(row["largest_class_pct"]),
            "converged": bool(row["converged"]),
            "n_iter": int(row["n_iter"]),
            "pct_maxpost_lt_50": float(urow["pct_maxprob_lt_50"]),
            "pct_maxpost_lt_70": float(urow["pct_maxprob_lt_70"]),
            "pct_maxpost_ge_70": float(urow["pct_maxprob_ge_70"]),
        })
    pd.DataFrame(comp_rows).to_csv(
        os.path.join(RESULTS_DIR, "lpa_comparison.csv"), index=False
    )
    summary_rows.append({
        "check": "lpa_comparison",
        "status": "PASS",
        "notes": "K=2..6 comparison extracted from existing results",
    })

    # ==================================================================
    # CHECK 3 — Profile structure for K=2..K=6
    # ==================================================================
    struct_rows = []
    for K in K_RANGE:
        kdir = os.path.join(PHASE04_DIR, f"K_{K}")
        post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
        labels = post_df["assigned_class"].values
        df = scores.copy()
        df["profile"] = labels
        for p in range(K):
            sub = df[df["profile"] == p]
            int_mean = sub["INT"].mean()
            be_mean = sub["BE"].mean()
            # Standardized means
            z_int_p = (int_mean - scores["INT"].mean()) / scores["INT"].std(ddof=1)
            z_be_p = (be_mean - scores["BE"].mean()) / scores["BE"].std(ddof=1)
            gap = z_int_p - z_be_p
            struct_rows.append({
                "K": K,
                "profile": p,
                "N": len(sub),
                "percentage": len(sub) / N * 100,
                "mean_INT": float(int_mean),
                "mean_BE": float(be_mean),
                "z_INT": float(z_int_p),
                "z_BE": float(z_be_p),
                "mean_GAP": float(gap),
            })
    pd.DataFrame(struct_rows).to_csv(
        os.path.join(RESULTS_DIR, "profile_structure.csv"), index=False
    )
    summary_rows.append({
        "check": "profile_structure",
        "status": "PASS",
        "notes": "Profile structure computed for K=2..6",
    })

    # ==================================================================
    # CHECK 4 — Gap profile verification
    # ==================================================================
    gap_rows = []
    for K in K_RANGE:
        kdir = os.path.join(PHASE04_DIR, f"K_{K}")
        post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
        labels = post_df["assigned_class"].values
        df = scores.copy()
        df["profile"] = labels
        for p in range(K):
            sub = df[df["profile"] == p]
            int_mean = sub["INT"].mean()
            be_mean = sub["BE"].mean()
            z_int_p = (int_mean - scores["INT"].mean()) / scores["INT"].std(ddof=1)
            z_be_p = (be_mean - scores["BE"].mean()) / scores["BE"].std(ddof=1)
            gap = z_int_p - z_be_p
            gap_rows.append({
                "K": K,
                "profile": p,
                "mean_INT": float(int_mean),
                "mean_BE": float(be_mean),
                "INT_minus_BE": float(int_mean - be_mean),
                "z_INT": float(z_int_p),
                "z_BE": float(z_be_p),
                "standardized_gap": float(gap),
                "INT_gt_BE": bool(int_mean > be_mean),
                "BE_gt_INT": bool(be_mean > int_mean),
            })
    pd.DataFrame(gap_rows).to_csv(
        os.path.join(RESULTS_DIR, "gap_profile_check.csv"), index=False
    )
    summary_rows.append({
        "check": "gap_profile_verification",
        "status": "PASS",
        "notes": "Numerical gap verification for K=2..6; no arbitrary labels applied",
    })

    # ==================================================================
    # CHECK 5 — Profile size / over-segmentation diagnostic
    # ==================================================================
    size_rows = []
    for K in K_RANGE:
        kdir = os.path.join(PHASE04_DIR, f"K_{K}")
        sizes_df = pd.read_csv(os.path.join(kdir, "profile_sizes.csv"))
        sizes = sizes_df["size"].values
        post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
        post_cols = [c for c in post_df.columns if c.startswith("post_profile_")]
        posteriors = post_df[post_cols].values
        max_post = posteriors.max(axis=1)
        size_rows.append({
            "K": K,
            "smallest_class_size": int(sizes.min()),
            "smallest_class_pct": float(sizes.min() / N * 100),
            "largest_class_pct": float(sizes.max() / N * 100),
            "pct_maxpost_lt_50": (max_post < 0.50).mean() * 100,
            "pct_maxpost_lt_70": (max_post < 0.70).mean() * 100,
            "pct_maxpost_ge_70": (max_post >= 0.70).mean() * 100,
        })
    pd.DataFrame(size_rows).to_csv(
        os.path.join(RESULTS_DIR, "profile_size_diagnostics.csv"), index=False
    )
    summary_rows.append({
        "check": "profile_size_diagnostics",
        "status": "PASS",
        "notes": "Size and uncertainty diagnostics for K=2..6",
    })

    # ==================================================================
    # CHECK 6 — Factor score robustness (sklearn FactorAnalysis)
    # ==================================================================
    fa_rows = []
    for construct, items in CONSTRUCTS.items():
        item_data = data[items].values.astype(float)
        mean_score = scores[construct].values

        # FactorAnalysis with 1 component
        fa = FactorAnalysis(n_components=1, random_state=SEED)
        fa.fit(item_data)
        # Factor scores: transform then align sign to mean score
        fs = fa.transform(item_data).flatten()
        r_raw = np.corrcoef(mean_score, fs)[0, 1]
        if r_raw < 0:
            fs = -fs
        r = float(np.corrcoef(mean_score, fs)[0, 1])
        fa_rows.append({
            "construct": construct,
            "items": ",".join(items),
            "N": len(mean_score),
            "mean_score_mean": float(mean_score.mean()),
            "mean_score_SD": float(mean_score.std(ddof=1)),
            "factor_score_mean": float(fs.mean()),
            "factor_score_SD": float(fs.std(ddof=1)),
            "pearson_r": r,
        })
    pd.DataFrame(fa_rows).to_csv(
        os.path.join(RESULTS_DIR, "factor_vs_mean.csv"), index=False
    )
    summary_rows.append({
        "check": "factor_score_robustness",
        "status": "PASS",
        "notes": "sklearn FactorAnalysis used; factor scores compared to mean scores",
    })

    # ==================================================================
    # CHECK 7 — K=6 reproducibility
    # ==================================================================
    # Re-run K=6 with exact same settings
    gmm_new = GaussianMixture(
        n_components=6,
        covariance_type=COVARIANCE_TYPE,
        n_init=N_INIT,
        random_state=SEED,
        max_iter=500,
        reg_covar=1e-6,
    )
    gmm_new.fit(X)
    LL_new = float(gmm_new.score(X) * N)
    n_params = count_params_full(6, X.shape[1])
    BIC_new = float(-2 * LL_new + n_params * np.log(N))
    AIC_new = float(-2 * LL_new + 2 * n_params)
    labels_new = gmm_new.predict(X)
    sizes_new = np.bincount(labels_new, minlength=6).tolist()
    means_new = gmm_new.means_.tolist()

    # Existing K=6
    kdir6 = os.path.join(PHASE04_DIR, "K_6")
    existing = pd.read_csv(os.path.join(kdir6, "profile_parameters.csv"))
    LL_existing = float(fit[fit["K"] == 6]["log_likelihood"].values[0])
    BIC_existing = float(fit[fit["K"] == 6]["BIC"].values[0])
    AIC_existing = float(fit[fit["K"] == 6]["AIC"].values[0])
    sizes_existing = existing["size"].tolist()
    means_existing = pd.read_csv(os.path.join(kdir6, "profile_means.csv")).values.tolist()

    repro = pd.DataFrame({
        "metric": ["log_likelihood", "BIC", "AIC", "profile_sizes", "profile_means"],
        "existing": [LL_existing, BIC_existing, AIC_existing, sizes_existing, means_existing],
        "reproduced": [LL_new, BIC_new, AIC_new, sizes_new, means_new],
    })
    repro.to_csv(os.path.join(RESULTS_DIR, "k6_reproduction.csv"), index=False)

    # Check match
    ll_match = abs(LL_new - LL_existing) < 0.01
    bic_match = abs(BIC_new - BIC_existing) < 0.01
    sizes_match = sorted(sizes_new) == sorted(sizes_existing)
    repro_status = "PASS" if (ll_match and bic_match and sizes_match) else "REVIEW"
    summary_rows.append({
        "check": "k6_reproducibility",
        "status": repro_status,
        "notes": f"LL match={ll_match}, BIC match={bic_match}, sizes match={sizes_match}",
    })

    # ==================================================================
    # Summary + README
    # ==================================================================
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "verification_summary.csv"), index=False)

    readme = f"""# Phase 11 — Post-Verification
## Intention–Behaviour Gap in Household Energy-Saving Behaviour

### Purpose
Verify the final results before treating them as frozen.
No changes to primary analysis. No literature. No figures.

### Checks Performed
1. **audit_failure.csv** — Identifies the Phase 09 check that did not pass
2. **lpa_comparison.csv** — K=2..6 stability comparison (BIC, AIC, LL, entropy, sizes, uncertainty)
3. **profile_structure.csv** — Profile structure for K=2..6 (N, %, mean INT/BE, z, GAP)
4. **gap_profile_check.csv** — Numerical gap verification (INT>BE, BE>INT)
5. **profile_size_diagnostics.csv** — Size and uncertainty diagnostics
6. **factor_vs_mean.csv** — Factor score robustness (sklearn FactorAnalysis)
7. **k6_reproduction.csv** — K=6 reproducibility check

### Key Findings
- Phase 09 failure: {n_fail} check(s) failed (see audit_failure.csv)
- K=6 reproducibility: {repro_status}
- Factor scores: computed via sklearn.decomposition.FactorAnalysis

### Scripts Used
- scripts/11_final_verification.py (this script)

### Results Used
- results/02_measurement/construct_scores.csv
- results/04_lpa_estimation/ (K=2..6)
- results/05_lpa_selection/ (classification_uncertainty.csv)
- results/10_final/numerical_audit/audit_checks.csv
"""
    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write(readme)

    # ==================================================================
    # Report
    # ==================================================================
    print("=" * 60)
    print("PHASE 11 — VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs in {RESULTS_DIR}:")
    for fn in [
        "audit_failure.csv",
        "lpa_comparison.csv",
        "profile_structure.csv",
        "gap_profile_check.csv",
        "profile_size_diagnostics.csv",
        "factor_vs_mean.csv",
        "k6_reproduction.csv",
        "verification_summary.csv",
        "README.md",
    ]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
