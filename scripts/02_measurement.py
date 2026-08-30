"""
Phase 02 — Measurement and Construct Scoring
==============================================
Project: Intention–Behaviour Gap in Household Energy-Saving Behaviour

Computes:
- item-level statistics (mean, SD)
- construct scores (arithmetic mean of available items per respondent)
- Cronbach's alpha for each construct
- Pearson correlation between INT and BE (r, p, N, 95% CI)

Construct definitions are fixed. No reverse coding. No imputation
(since Phase 01 reported 0 missing values).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "data.xls"
RESULTS_DIR = "results/02_measurement"
SEED = 42

# Fixed construct definitions
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


def cronbach_alpha(item_scores: pd.DataFrame) -> float:
    """
    Cronbach's alpha for k items:
    alpha = [k / (k - 1)] * [1 - sum(item_variances) / variance(total)]
    item_scores: DataFrame (n x k), each column is an item.
    """
    items = item_scores.dropna(axis=1)  # no missing in this dataset
    k = items.shape[1]
    if k < 2:
        return np.nan
    item_variances = items.var(axis=0, ddof=1)
    total = items.sum(axis=1)
    total_var = total.var(ddof=1)
    if total_var == 0.0:
        return np.nan
    alpha = (k / (k - 1.0)) * (1.0 - item_variances.sum() / total_var)
    return float(alpha)


def pearson_ci(r, n, alpha=0.05):
    """
    95% CI for Pearson r using Fisher z-transformation.
    """
    if n <= 3:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3.0)
    z_crit = stats.norm.ppf(1 - alpha / 2.0)
    lo = z - z_crit * se
    hi = z + z_crit * se
    return (float(np.tanh(lo)), float(np.tanh(hi)))


def main():
    ensure_dir(RESULTS_DIR)
    np.random.seed(SEED)

    df = pd.read_excel(DATA_PATH, sheet_name=0).copy()
    N = df.shape[0]

    # ------------------------------------------------------------------
    # 1. Item statistics
    # ------------------------------------------------------------------
    all_items = [it for its in CONSTRUCTS.values() for it in its]
    item_stats = []
    for c in all_items:
        s = df[c].astype(float)
        item_stats.append(
            {
                "item": c,
                "construct": next(k for k, v in CONSTRUCTS.items() if c in v),
                "N": int(s.count()),
                "mean": float(s.mean()),
                "SD": float(s.std(ddof=1)),
                "min": float(s.min()),
                "max": float(s.max()),
                "var": float(s.var(ddof=1)),
            }
        )
    item_stats_df = pd.DataFrame(item_stats)
    item_stats_df.to_csv(
        os.path.join(RESULTS_DIR, "item_statistics.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 2. Construct scores (arithmetic mean of available items per row)
    # ------------------------------------------------------------------
    scores = pd.DataFrame(index=df.index)
    for construct, items in CONSTRUCTS.items():
        present = [c for c in items if c in df.columns]
        scores[construct] = df[present].astype(float).mean(axis=1)

    # Preserve respondent ordering — index = respondent id
    scores.index.name = "respondent"
    scores.to_csv(os.path.join(RESULTS_DIR, "construct_scores.csv"))

    # ------------------------------------------------------------------
    # 3. Construct statistics
    # ------------------------------------------------------------------
    construct_stats = []
    for construct, items in CONSTRUCTS.items():
        item_df = df[items].astype(float)
        k = len(items)
        alpha = cronbach_alpha(item_df)
        construct_stats.append(
            {
                "construct": construct,
                "k_items": k,
                "items": ",".join(items),
                "N": int(scores[construct].count()),
                "mean": float(scores[construct].mean()),
                "SD": float(scores[construct].std(ddof=1)),
                "min": float(scores[construct].min()),
                "max": float(scores[construct].max()),
                "Cronbach_alpha": alpha,
            }
        )
    cs_df = pd.DataFrame(construct_stats)
    cs_df.to_csv(os.path.join(RESULTS_DIR, "construct_statistics.csv"), index=False)

    # Also save a dedicated cronbach alpha file (spec requires it)
    cronbach_df = cs_df[["construct", "k_items", "items", "Cronbach_alpha"]].copy()
    cronbach_df.to_csv(os.path.join(RESULTS_DIR, "cronbach_alpha.csv"), index=False)

    # ------------------------------------------------------------------
    # 4. INT–BE Pearson correlation
    # ------------------------------------------------------------------
    x = scores["INT"].astype(float)
    y = scores["BE"].astype(float)
    r, p = stats.pearsonr(x, y)
    n_valid = int(x.notna().sum() & y.notna().sum())
    cov = ((x - x.mean()) * (y - y.mean())).sum() / (n_valid - 1)
    ci_lo, ci_hi = pearson_ci(r, n_valid)

    int_be = pd.DataFrame(
        {
            "variable_pair": ["INT-BE"],
            "r": [float(r)],
            "p": [float(p)],
            "N": [n_valid],
            "covariance": [float(cov)],
            "CI95_lower": [ci_lo],
            "CI95_upper": [ci_hi],
        }
    )
    int_be.to_csv(os.path.join(RESULTS_DIR, "int_be_correlation.csv"), index=False)

    # ------------------------------------------------------------------
    # 5. Full construct correlation matrix (auxiliary — useful later)
    # ------------------------------------------------------------------
    corr_mat = scores.corr(method="pearson")
    corr_mat.to_csv(os.path.join(RESULTS_DIR, "construct_correlation_matrix.csv"))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 02 — MEASUREMENT COMPLETE")
    print("=" * 60)
    print(f"N = {N}")
    print("\nCronbach alpha:")
    for _, row in cronbach_df.iterrows():
        print(f"  {row['construct']:>6s} (k={int(row['k_items'])}): {row['Cronbach_alpha']:.4f}")
    print("\nINT–BE correlation:")
    print(f"  r = {r:.4f}")
    print(f"  p = {p:.6f}")
    print(f"  N = {n_valid}")
    print(f"  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

    print("\nConstruct means:")
    for _, row in cs_df.iterrows():
        print(
            f"  {row['construct']:>6s}: mean={row['mean']:.4f}, SD={row['SD']:.4f}"
        )

    print("\nKey outputs:")
    for fn in [
        "item_statistics.csv",
        "construct_statistics.csv",
        "cronbach_alpha.csv",
        "int_be_correlation.csv",
        "construct_scores.csv",
        "construct_correlation_matrix.csv",
    ]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
