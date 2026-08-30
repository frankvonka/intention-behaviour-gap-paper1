"""
Phase 03 — Continuous Intention–Behaviour Gap
==============================================
Project: Intention–Behaviour Gap in Household Energy-Saving Behaviour

Computes:
- standardized INT and BE (z-scores)
- GAP_i = z_INT_i - z_BE_i
- gap statistics (mean, SD, median, min, max, counts by sign)

No arbitrary threshold categories are created.
"""
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORES_PATH = "results/02_measurement/construct_scores.csv"
RESULTS_DIR = "results/03_gap_analysis"
SEED = 42


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    ensure_dir(RESULTS_DIR)
    np.random.seed(SEED)

    scores = pd.read_csv(SCORES_PATH, index_col="respondent")
    N = scores.shape[0]

    INT = scores["INT"].astype(float)
    BE = scores["BE"].astype(float)

    # Standardize across respondents
    z_INT = (INT - INT.mean()) / INT.std(ddof=1)
    z_BE = (BE - BE.mean()) / BE.std(ddof=1)

    # Gap
    GAP = z_INT - z_BE

    # ------------------------------------------------------------------
    # 1. Standardized scores + gap
    # ------------------------------------------------------------------
    std_df = pd.DataFrame(
        {
            "respondent": scores.index,
            "INT_raw": INT.values,
            "BE_raw": BE.values,
            "z_INT": z_INT.values,
            "z_BE": z_BE.values,
            "GAP": GAP.values,
        }
    )
    std_df.to_csv(
        os.path.join(RESULTS_DIR, "standardized_scores.csv"), index=False
    )

    # Save gap scores as .npy (spec requires)
    np.save(os.path.join(RESULTS_DIR, "gap_scores.npy"), GAP.values)

    # ------------------------------------------------------------------
    # 2. Gap statistics
    # ------------------------------------------------------------------
    pos = int((GAP > 0).sum())
    neg = int((GAP < 0).sum())
    zero = int((GAP == 0).sum())

    gap_stats = pd.DataFrame(
        {
            "statistic": [
                "N",
                "mean",
                "SD",
                "median",
                "min",
                "max",
                "positive_gap_count",
                "negative_gap_count",
                "zero_gap_count",
                "positive_gap_pct",
                "negative_gap_pct",
                "zero_gap_pct",
            ],
            "value": [
                N,
                float(GAP.mean()),
                float(GAP.std(ddof=1)),
                float(GAP.median()),
                float(GAP.min()),
                float(GAP.max()),
                pos,
                neg,
                zero,
                pos / N * 100,
                neg / N * 100,
                zero / N * 100,
            ],
        }
    )
    gap_stats.to_csv(os.path.join(RESULTS_DIR, "gap_statistics.csv"), index=False)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 03 — GAP ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"N = {N}")
    print(f"z_INT: mean={z_INT.mean():.6f}, SD={z_INT.std(ddof=1):.6f}")
    print(f"z_BE : mean={z_BE.mean():.6f}, SD={z_BE.std(ddof=1):.6f}")
    print(f"GAP  : mean={GAP.mean():.4f}, SD={GAP.std(ddof=1):.4f}")
    print(f"      median={GAP.median():.4f}, min={GAP.min():.4f}, max={GAP.max():.4f}")
    print(f"Positive GAP (INT>BE): {pos} ({pos/N*100:.2f}%)")
    print(f"Negative GAP (BE>INT): {neg} ({neg/N*100:.2f}%)")
    print(f"Zero GAP:              {zero} ({zero/N*100:.2f}%)")

    print("\nKey outputs:")
    for fn in ["gap_scores.npy", "standardized_scores.csv", "gap_statistics.csv"]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
