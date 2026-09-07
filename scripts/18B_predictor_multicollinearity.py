"""
Phase 18B — Profile Predictor Multicollinearity Audit
=====================================================
Diagnostic sensitivity of the Phase 18 multinomial model to
multicollinearity. No model selection, no interpretation, no
variable removal beyond the leave-one-out diagnostics specified.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORES_PATH = "results/02_measurement/construct_scores.csv"
PHASE04_DIR = "results/04_lpa_estimation"
DATA_PATH = "data.xls"
RESULTS_DIR = "results/18B_predictor_multicollinearity"
PHASE18_DIR = "results/18_profile_predictors"
PHASE05_DIR = "results/05_lpa_selection"

K = int(pd.read_csv(os.path.join(PHASE05_DIR, "selected_model.csv"))["selected_K"].iloc[0])
REFERENCE_PROFILE = 0
ALPHA_FDR = 0.05
HIGH_CORR_THRESHOLD = 0.70

CONSTRUCT_PREDICTORS = ["ATT", "CON", "SNO", "COVID", "PU", "PEU", "PO", "PRI"]
DEMOGRAPHICS = ["age", "gender", "Education", "Occupation", "income"]
PREDICTORS = CONSTRUCT_PREDICTORS + DEMOGRAPHICS  # 13

# Standardize continuous predictors only (gender is binary)
CONTINUOUS = ["ATT", "CON", "SNO", "COVID", "PU", "PEU", "PO", "PRI",
              "age", "Education", "Occupation", "income"]


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def benjamini_hochberg(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adj_sorted = sorted_p * m / np.arange(1, m + 1)
    for i in range(m - 2, -1, -1):
        if adj_sorted[i] > adj_sorted[i + 1]:
            adj_sorted[i] = adj_sorted[i + 1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    result = np.empty(m)
    result[order] = adj_sorted
    return result


def fit_mnlogit(y, X_df):
    """Fit MNLogit and return DataFrame of (profile, predictor, beta, SE, z, p)."""
    Xd = sm.add_constant(X_df)
    model = MNLogit(y, Xd)
    res = model.fit(method="bfgs", maxiter=1000, disp=False)
    params = res.params
    bse = res.bse
    z_vals = res.tvalues
    p_vals = res.pvalues

    rows = []
    fdr_input = []
    for cat_pos in list(params.columns):
        target_profile = int(cat_pos) + 1
        for pred in params.index:
            if pred == "const":
                continue
            beta = float(params.loc[pred, cat_pos])
            se = float(bse.loc[pred, cat_pos])
            z = float(z_vals.loc[pred, cat_pos])
            p = float(p_vals.loc[pred, cat_pos])
            or_val = float(np.exp(beta))
            ci_lo = float(np.exp(beta - 1.96 * se))
            ci_hi = float(np.exp(beta + 1.96 * se))
            rows.append({
                "profile": target_profile,
                "reference_profile": REFERENCE_PROFILE,
                "predictor": pred,
                "beta": beta,
                "SE": se,
                "z": z,
                "p": p,
                "OR": or_val,
                "CI_lower": ci_lo,
                "CI_upper": ci_hi,
            })
            fdr_input.append(p)
    df = pd.DataFrame(rows)
    if len(fdr_input) > 0:
        df["p_FDR"] = benjamini_hochberg(fdr_input)
        df["significant_FDR"] = df["p_FDR"] < ALPHA_FDR
    diag = {
        "log_likelihood": float(res.llf),
        "log_likelihood_null": float(res.llnull),
        "AIC": float(res.aic),
        "BIC": float(res.bic),
        "McFadden_pseudo_R2": float(1.0 - res.llf / res.llnull),
        "converged": bool(res.mle_retvals.get("converged", False)),
        "n_predictors": int(X_df.shape[1]),
        "n_tests": int(len(fdr_input)),
    }
    return df, diag, res


def main():
    ensure_dir(RESULTS_DIR)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    raw = pd.read_excel(DATA_PATH, sheet_name=0).copy()
    N = scores.shape[0]

    # Frozen selected-K classification
    kdir = os.path.join(PHASE04_DIR, f"K_{K}")
    post_df = pd.read_csv(os.path.join(kdir, "posterior_probabilities.csv"))
    labels = post_df["assigned_class"].values

    # Predictor matrix (raw)
    X_raw = pd.concat([scores[CONSTRUCT_PREDICTORS],
                       raw[DEMOGRAPHICS]], axis=1).apply(pd.to_numeric, errors="coerce")
    y = pd.Series(labels, index=scores.index)

    # ------------------------------------------------------------------
    # 1. REPRODUCE PHASE 18 PRIMARY MODEL
    # ------------------------------------------------------------------
    primary_df, primary_diag, _ = fit_mnlogit(y, X_raw)
    primary_df.to_csv(os.path.join(RESULTS_DIR, "primary_reproduction.csv"), index=False)

    # Compare against Phase 18
    p18 = pd.read_csv(os.path.join(PHASE18_DIR, "multinomial_results.csv"))
    a = primary_df.sort_values(["profile", "predictor"]).reset_index(drop=True)
    b = p18[["profile", "predictor", "beta", "SE", "p", "p_FDR", "OR",
             "CI_lower", "CI_upper"]].sort_values(
        ["profile", "predictor"]).reset_index(drop=True)
    merged = a.merge(b, on=["profile", "predictor"], suffixes=("_p18B", "_p18"))
    max_abs_beta_diff = float(np.max(np.abs(merged["beta_p18B"] - merged["beta_p18"])))
    max_abs_or_diff = float(np.max(np.abs(merged["OR_p18B"] - merged["OR_p18"])))

    repro_summary = pd.DataFrame([
        {"metric": "log_likelihood", "p18B": primary_diag["log_likelihood"]},
        {"metric": "AIC", "p18B": primary_diag["AIC"]},
        {"metric": "BIC", "p18B": primary_diag["BIC"]},
        {"metric": "McFadden_pseudo_R2", "p18B": primary_diag["McFadden_pseudo_R2"]},
        {"metric": "converged", "p18B": float(primary_diag["converged"])},
        {"metric": "max_abs_beta_diff_vs_phase18", "p18B": max_abs_beta_diff},
        {"metric": "max_abs_OR_diff_vs_phase18", "p18B": max_abs_or_diff},
    ])
    repro_summary.to_csv(os.path.join(RESULTS_DIR, "primary_reproduction_summary.csv"),
                         index=False)

    # ------------------------------------------------------------------
    # 2. FULL VIF DIAGNOSTIC
    # ------------------------------------------------------------------
    Xv = X_raw.values.astype(float)
    vif_rows = []
    for i, p in enumerate(PREDICTORS):
        v = float(variance_inflation_factor(Xv, i))
        vif_rows.append({"predictor": p, "VIF": v, "tolerance": 1.0 / v if v > 0 else np.nan})
    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    vif_df.to_csv(os.path.join(RESULTS_DIR, "vif_full.csv"), index=False)

    # ------------------------------------------------------------------
    # 3. PREDICTOR CORRELATION MATRIX
    # ------------------------------------------------------------------
    corr = X_raw.corr(method="pearson")
    corr.to_csv(os.path.join(RESULTS_DIR, "predictor_correlation.csv"))

    # ------------------------------------------------------------------
    # 4. CONDITION DIAGNOSTICS
    # ------------------------------------------------------------------
    corr_mat = corr.values
    eigs = np.linalg.eigvalsh(corr_mat)
    eigs_sorted = np.sort(eigs)[::-1]
    cond_rows = [
        {"metric": "condition_number_corr", "value": float(np.linalg.cond(corr_mat))},
        {"metric": "smallest_eigenvalue_corr", "value": float(eigs_sorted[-1])},
        {"metric": "largest_eigenvalue_corr", "value": float(eigs_sorted[0])},
    ]
    for i, ev in enumerate(eigs_sorted):
        cond_rows.append({"metric": f"eigenvalue_{i+1}", "value": float(ev)})
    pd.DataFrame(cond_rows).to_csv(
        os.path.join(RESULTS_DIR, "condition_diagnostics.csv"), index=False)

    # ------------------------------------------------------------------
    # 5. HIGH CORRELATION PAIRS
    # ------------------------------------------------------------------
    pairs_rows = []
    for i in range(len(PREDICTORS)):
        for j in range(i + 1, len(PREDICTORS)):
            r = float(corr.loc[PREDICTORS[i], PREDICTORS[j]])
            if abs(r) >= HIGH_CORR_THRESHOLD:
                pairs_rows.append({
                    "predictor_1": PREDICTORS[i],
                    "predictor_2": PREDICTORS[j],
                    "pearson_r": r,
                    "abs_r": abs(r),
                })
    pairs_df = pd.DataFrame(pairs_rows).sort_values("abs_r", ascending=False)
    if len(pairs_df) == 0:
        pairs_df = pd.DataFrame(columns=["predictor_1", "predictor_2", "pearson_r", "abs_r"])
    pairs_df.to_csv(os.path.join(RESULTS_DIR, "high_correlation_pairs.csv"), index=False)

    # ------------------------------------------------------------------
    # 6. LEAVE-ONE-PREDICTOR-OUT
    # ------------------------------------------------------------------
    lop_rows = []
    lop_diag_rows = []
    for dropped in [None] + CONSTRUCT_PREDICTORS:
        if dropped is None:
            spec_name = "A_primary"
            pred_set = PREDICTORS
        else:
            spec_name = f"drop_{dropped}"
            pred_set = [p for p in PREDICTORS if p != dropped]
        X_sub = X_raw[pred_set]
        sub_df, sub_diag, _ = fit_mnlogit(y, X_sub)
        sub_df["specification"] = spec_name
        sub_df["dropped_predictor"] = dropped if dropped is not None else ""
        lop_rows.append(sub_df)
        sub_diag["specification"] = spec_name
        sub_diag["dropped_predictor"] = dropped if dropped is not None else ""
        lop_diag_rows.append(sub_diag)

    lop_df = pd.concat(lop_rows, ignore_index=True)
    lop_diag_df = pd.DataFrame(lop_diag_rows)
    lop_df.to_csv(os.path.join(RESULTS_DIR, "leave_one_predictor_out.csv"), index=False)
    lop_diag_df.to_csv(os.path.join(RESULTS_DIR, "leave_one_predictor_out_diagnostics.csv"),
                       index=False)

    # ------------------------------------------------------------------
    # 7. STANDARDIZED-CONTINUOUS SENSITIVITY
    # ------------------------------------------------------------------
    X_std = X_raw.copy()
    for c in CONTINUOUS:
        s = X_std[c].std(ddof=1)
        if s > 0:
            X_std[c] = (X_std[c] - X_std[c].mean()) / s
        else:
            X_std[c] = 0.0
    # gender untouched (binary)

    std_df, std_diag, _ = fit_mnlogit(y, X_std)
    std_df["specification"] = "standardized_continuous"
    std_df.to_csv(os.path.join(RESULTS_DIR, "standardized_predictor_model.csv"), index=False)

    # ------------------------------------------------------------------
    # 8. COEFFICIENT STABILITY ACROSS SENSITIVITY MODELS
    # ------------------------------------------------------------------
    prim = primary_df[["profile", "predictor", "beta", "p", "p_FDR", "OR"]].rename(
        columns={"beta": "primary_beta", "p": "primary_p", "p_FDR": "primary_FDR_p",
                 "OR": "primary_OR"}
    )

    lop_only = lop_df[~lop_df["specification"].isin(["A_primary"])].copy()
    lop_pivot = lop_only.pivot_table(
        index=["profile", "predictor"],
        columns="dropped_predictor",
        values=["beta", "p_FDR"],
        aggfunc="first",
    )
    lop_pivot.columns = [f"{stat}_drop_{dp}" for stat, dp in lop_pivot.columns]
    lop_pivot = lop_pivot.reset_index()

    std_subset = std_df[["profile", "predictor", "beta", "p_FDR"]].rename(
        columns={"beta": "std_beta", "p_FDR": "std_FDR_p"}
    )

    stability = prim.merge(lop_pivot, on=["profile", "predictor"], how="left")
    stability = stability.merge(std_subset, on=["profile", "predictor"], how="left")

    # Track whether primary FDR-sig remains sig in each sensitivity model
    lop_drop_cols = sorted([c for c in lop_only["dropped_predictor"].dropna().unique()])
    stability["n_FDR_sig_dropouts_of_total"] = 0
    for dp in lop_drop_cols:
        sig_col = f"fdr_sig_in_drop_{dp}"
        col = f"p_FDR_drop_{dp}"
        stability[sig_col] = stability[col].notna() & (stability[col] < ALPHA_FDR)
        stability["n_FDR_sig_dropouts_of_total"] += stability[sig_col].fillna(False).astype(int)
    stability["fdr_sig_in_standardized"] = stability["std_FDR_p"] < ALPHA_FDR
    stability["primary_significant_FDR"] = stability["primary_FDR_p"] < ALPHA_FDR

    stability.to_csv(os.path.join(RESULTS_DIR, "coefficient_stability.csv"), index=False)

    # ------------------------------------------------------------------
    # 9. PHASE 18B MASTER
    # ------------------------------------------------------------------
    master_rows = [
        {"item": "N", "value": float(N)},
        {"item": "K", "value": float(K)},
        {"item": "reference_profile", "value": float(REFERENCE_PROFILE)},
        {"item": "n_predictors_primary", "value": float(len(PREDICTORS))},
        {"item": "n_tests_primary", "value": float(primary_diag["n_tests"])},
        {"item": "primary_LL", "value": primary_diag["log_likelihood"]},
        {"item": "primary_AIC", "value": primary_diag["AIC"]},
        {"item": "primary_BIC", "value": primary_diag["BIC"]},
        {"item": "primary_pseudo_R2", "value": primary_diag["McFadden_pseudo_R2"]},
        {"item": "primary_converged", "value": float(primary_diag["converged"])},
        {"item": "max_VIF", "value": float(vif_df["VIF"].max())},
        {"item": "condition_number_corr", "value": float(np.linalg.cond(corr_mat))},
        {"item": "smallest_eigenvalue_corr", "value": float(eigs_sorted[-1])},
        {"item": "n_pairs_abs_r_ge_0.70", "value": float(len(pairs_df))},
        {"item": "max_abs_beta_diff_vs_phase18", "value": max_abs_beta_diff},
        {"item": "max_abs_OR_diff_vs_phase18", "value": max_abs_or_diff},
    ]
    pd.DataFrame(master_rows).to_csv(
        os.path.join(RESULTS_DIR, "phase18B_master.csv"), index=False)

    # ------------------------------------------------------------------
    # 10. README
    # ------------------------------------------------------------------
    high_vif = vif_df[vif_df["VIF"] >= 10.0][["predictor", "VIF"]].to_string(index=False)
    top_pairs = pairs_df.head(20).to_string(index=False) if len(pairs_df) else "(none)"

    # Per-spec FDR counts
    lop_diag_disp = lop_diag_df[[
        "specification", "log_likelihood", "AIC", "BIC", "McFadden_pseudo_R2",
        "converged", "n_predictors", "n_tests",
    ]].to_string(index=False)

    readme = f"""# Phase 18B — Profile Predictor Multicollinearity Audit

## Purpose
Investigate the multicollinearity observed in the Phase 18
multinomial predictor model. Diagnostic and sensitivity only.
No model selection, no interpretation, no variable removal beyond
the one-at-a-time leave-out diagnostics specified.

## Tests Performed
1. **Primary model reproduction** — same specification as Phase 18
2. **VIF diagnostic** — variance_inflation_factor for all 13 predictors
3. **Predictor correlation matrix** — Pearson r (N = {N})
4. **Condition diagnostics** — condition number, eigenvalues of
   predictor correlation matrix, smallest eigenvalue
5. **High-correlation pairs** — pairs with |r| >= {HIGH_CORR_THRESHOLD}
6. **Leave-one-predictor-out** — drop each of the 8 construct
   predictors one at a time (demographics always retained)
7. **Standardized-continuous sensitivity** — standardize all
   continuous predictors (gender NOT standardized)
8. **Coefficient stability table** — primary coefficients vs
   leave-one-out vs standardized

## Models
- Outcome: frozen K={K} profile membership (minimum BIC among non-degenerate
  fits, per results/05_lpa_selection/selected_model.csv)
- Reference: Profile {REFERENCE_PROFILE}
- Estimator: statsmodels MNLogit, BFGS, maxiter=1000
- Predictors (primary, 13): {PREDICTORS}
- Predictors (leave-one-out, 12 each): drop one construct at a time

## FDR Procedure
- Benjamini-Hochberg within each model across ALL predictor/profile
  contrasts (NOT per-profile, NOT across models)
- alpha = {ALPHA_FDR}

## Primary Model Reproduction
- LL = {primary_diag['log_likelihood']:.4f}
- AIC = {primary_diag['AIC']:.4f}
- BIC = {primary_diag['BIC']:.4f}
- McFadden pseudo-R2 = {primary_diag['McFadden_pseudo_R2']:.4f}
- Converged = {primary_diag['converged']}
- max |beta diff vs Phase 18| = {max_abs_beta_diff:.6f}
- max |OR diff vs Phase 18|   = {max_abs_or_diff:.6f}

## VIF (descending)
{high_vif}

## Condition Diagnostics
- condition number (corr) = {float(np.linalg.cond(corr_mat)):.4f}
- smallest eigenvalue     = {float(eigs_sorted[-1]):.6f}
- n pairs |r| >= 0.70     = {len(pairs_df)}

## Leave-One-Out Diagnostics
{lop_diag_disp}

## High-Correlation Pairs (top 20)
{top_pairs}

## Coefficient Stability
- Comparison table in `coefficient_stability.csv`
- Numerical stability summary in `phase18B_master.csv`

## No Modifications
Phase 18 outputs untouched. Dataset unchanged. No plots.
"""
    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write(readme)

    # ------------------------------------------------------------------
    # VERIFICATION
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 18B — MULTICOLLINEARITY AUDIT COMPLETE")
    print("=" * 60)
    print(f"N = {N}, K = {K}, ref = {REFERENCE_PROFILE}")
    print(f"Primary LL={primary_diag['log_likelihood']:.4f}, "
          f"AIC={primary_diag['AIC']:.4f}, BIC={primary_diag['BIC']:.4f}, "
          f"R2={primary_diag['McFadden_pseudo_R2']:.4f}, "
          f"conv={primary_diag['converged']}")
    print(f"Max |beta diff| vs Phase 18: {max_abs_beta_diff:.6f}")
    print(f"Max VIF: {vif_df['VIF'].max():.4f}")
    print(f"Condition number (corr): {float(np.linalg.cond(corr_mat)):.4f}")
    print(f"Smallest eigenvalue: {float(eigs_sorted[-1]):.6f}")
    print(f"High-corr pairs (|r|>=0.70): {len(pairs_df)}")
    print(f"\nVIF table:")
    print(vif_df.to_string(index=False))
    print(f"\nLeave-one-out diagnostics:")
    print(lop_diag_df[[
        "specification", "log_likelihood", "AIC", "BIC",
        "McFadden_pseudo_R2", "converged"
    ]].to_string(index=False))

    print("\nOutputs:")
    expected = [
        "primary_reproduction.csv", "primary_reproduction_summary.csv",
        "vif_full.csv", "predictor_correlation.csv",
        "condition_diagnostics.csv", "high_correlation_pairs.csv",
        "leave_one_predictor_out.csv", "leave_one_predictor_out_diagnostics.csv",
        "standardized_predictor_model.csv", "coefficient_stability.csv",
        "phase18B_master.csv", "README.md",
    ]
    for fn in expected:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
