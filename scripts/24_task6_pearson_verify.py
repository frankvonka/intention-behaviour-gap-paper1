"""
Reviewer-fix pass — Task 6: Verify the frozen INT–BE Pearson r with scipy.stats.pearsonr.

Frozen values are LOADED from results/02_measurement/int_be_correlation.csv
(Phase 02 output: columns r, p, N, CI95_lower, CI95_upper) — no hardcoded numbers.

This script recomputes r and p directly from data.xls and reports them
side by side with the frozen values so the match/mismatch is explicit.

This script has no K dependence (it verifies the correlation only, not the LPA).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = "data.xls"
SCORES_PATH = "results/02_measurement/construct_scores.csv"
FROZEN_CSV = "results/02_measurement/int_be_correlation.csv"
OUT = "results/paper1_strengthening/pearson_verification.csv"
SEED = 42
CONSTRUCTS = {
    "INT": ["INT1", "INT2", "INT3"],
    "BE": ["BE1", "BE2", "BE3", "BE4"],
}


def cronbach_alpha(items: pd.DataFrame) -> float:
    items = items.dropna(axis=1)
    k = items.shape[1]
    if k < 2:
        return float("nan")
    iv = items.var(axis=0, ddof=1)
    tv = items.sum(axis=1).var(ddof=1)
    if tv == 0.0:
        return float("nan")
    return float((k / (k - 1.0)) * (1.0 - iv.sum() / tv))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.random.seed(SEED)

    # ---- Frozen values, LOADED from the Phase 02 result file (no literals) ----
    _fz = pd.read_csv(FROZEN_CSV).iloc[0]
    FROZEN = {
        "r": float(_fz["r"]),
        "p": float(_fz["p"]),
        "N": int(_fz["N"]),
        "ci_lo": float(_fz["CI95_lower"]),
        "ci_hi": float(_fz["CI95_upper"]),
    }

    df = pd.read_excel(DATA_PATH, sheet_name=0).copy()
    N_data = df.shape[0]

    # Recompute construct scores the SAME way Phase 02 does (mean of items)
    scores = pd.DataFrame(index=df.index)
    alphas = {}
    for c, items in CONSTRUCTS.items():
        present = [it for it in items if it in df.columns]
        scores[c] = df[present].astype(float).mean(axis=1)
        alphas[c] = cronbach_alpha(df[present].astype(float))

    # Reload the saved construct scores (these are the canonical Phase-02 scores)
    scores_csv = pd.read_csv(SCORES_PATH, index_col="respondent")
    N_scores = scores_csv.shape[0]

    out_rows = []

    for label, x, y, n in [
        ("data.xls_construct_means", scores["INT"], scores["BE"], len(scores)),
        ("saved_construct_scores", scores_csv["INT"], scores_csv["BE"], len(scores_csv)),
    ]:
        r, p = stats.pearsonr(x, y)
        n_valid = int(x.notna().sum() & y.notna().sum())
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n_valid - 3)
        z_crit = stats.norm.ppf(0.975)
        ci_lo = float(np.tanh(z - z_crit * se))
        ci_hi = float(np.tanh(z + z_crit * se))
        out_rows.append({
            "source": label,
            "N": n_valid,
            "r": float(r),
            "p": float(p),
            "p_scientific": float(p),
            "CI95_lower_fisher": ci_lo,
            "CI95_upper_fisher": ci_hi,
            "covariance": float(((x - x.mean()) * (y - y.mean())).sum() / (n_valid - 1)),
            "r_frozen": FROZEN["r"],
            "p_frozen": FROZEN["p"],
            "N_frozen": FROZEN["N"],
            "CI95_lower_frozen": FROZEN["ci_lo"],
            "CI95_upper_frozen": FROZEN["ci_hi"],
            "match_frozen": bool(
                np.isclose(r, FROZEN["r"], atol=1e-4)
                and n_valid == FROZEN["N"]
                and np.isclose(ci_lo, FROZEN["ci_lo"], atol=1e-2)
                and np.isclose(ci_hi, FROZEN["ci_hi"], atol=1e-2)
            ),
        })

    out = pd.DataFrame(out_rows)
    # Add alpha row for completeness
    alpha_row = {
        "source": "construct_alphas",
        "N": N_data,
        "r": np.nan, "p": np.nan, "p_scientific": np.nan,
        "CI95_lower_fisher": np.nan, "CI95_upper_fisher": np.nan,
        "covariance": np.nan,
        "r_frozen": FROZEN["r"], "p_frozen": FROZEN["p"], "N_frozen": FROZEN["N"],
        "CI95_lower_frozen": FROZEN["ci_lo"], "CI95_upper_frozen": FROZEN["ci_hi"],
        "match_frozen": False,
    }
    out = pd.concat([out, pd.DataFrame([alpha_row])], ignore_index=True)

    # Annotate alpha values in a comment-style row is not CSV-friendly; print separately
    out.to_csv(OUT, index=False)

    print("=" * 70)
    print("TASK 6 — PearON r VERIFICATION (scipy.stats.pearsonr)")
    print("=" * 70)
    print(f"Frozen: r={FROZEN['r']}, p={FROZEN['p']:.3e}, "
          f"N={FROZEN['N']}, 95% CI=[{FROZEN['ci_lo']}, {FROZEN['ci_hi']}]")
    print(f"\nData: {DATA_PATH}, N_data={N_data}")
    print(f"Cronbach alpha: INT={alphas['INT']:.4f}, BE={alphas['BE']:.4f}")
    print(f"Saved construct scores: {SCORES_PATH}, N={N_scores}")
    print()
    for _, row in out.dropna(subset=["r"]).iterrows():
        print(f"  [{row['source']}]")
        print(f"    N = {int(row['N'])}")
        print(f"    r = {row['r']:.6f}  (frozen {row['r_frozen']})")
        print(f"    p = {row['p_scientific']:.6e}  (frozen {row['p_frozen']:.6e})")
        print(f"    95% CI (Fisher) = [{row['CI95_lower_fisher']:.4f}, {row['CI95_upper_fisher']:.4f}]"
              f"  (frozen [{row['CI95_lower_frozen']}, {row['CI95_upper_frozen']})")
        print(f"    covariance = {row['covariance']:.6f}")
        print(f"    MATCH frozen? {row['match_frozen']}")
        print()

    print(f"Saved: {OUT}")

    # Explicit verdict
    ok = out.dropna(subset=["r"]).loc[out["source"] == "saved_construct_scores", "match_frozen"]
    if bool(ok.iloc[0]):
        print(f"VERDICT: FROZEN r={FROZEN['r']:.6f}, p={FROZEN['p']:.3e}, "
              f"CI[{FROZEN['ci_lo']:.4f},{FROZEN['ci_hi']:.4f}] — VERIFIED (recomputed matches).")
    else:
        print("VERDICT: MISMATCH — investigate.")


if __name__ == "__main__":
    main()
