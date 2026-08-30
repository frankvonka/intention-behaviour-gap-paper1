"""
Phase 01 — Data Inspection and Validation
=========================================
Project: Intention–Behaviour Gap in Household Energy-Saving Behaviour

Reads data.xls, validates structure, checks ranges, missing values,
duplicates, and item-level summary statistics.
"""
import json
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "data.xls"
RESULTS_DIR = "results/01_data_inspection"
SEED = 42

# Expected psychometric items grouped by construct
EXPECTED_ITEMS = {
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

ALL_ITEMS = [item for items in EXPECTED_ITEMS.values() for item in items]
DEMOGRAPHICS = ["gender", "age", "Education", "Occupation", "income"]

# Expected Likert scale range
LIKERT_MIN, LIKERT_MAX = 1, 5


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    ensure_dir(RESULTS_DIR)

    # ------------------------------------------------------------------
    # Load data — operate on a copy in memory, never modify data.xls
    # ------------------------------------------------------------------
    xl = pd.ExcelFile(DATA_PATH)
    sheets = xl.sheet_names
    df = pd.read_excel(DATA_PATH, sheet_name=0)  # first (only) sheet "data"
    df = df.copy()  # copy in memory — raw data.xls is never touched

    N = df.shape[0]
    ncols = df.shape[1]

    # ------------------------------------------------------------------
    # 1. Workbook / dimensions metadata
    # ------------------------------------------------------------------
    metadata = {
        "file": DATA_PATH,
        "sheets": sheets,
        "active_sheet": sheets[0] if sheets else None,
        "N_rows": N,
        "N_columns": ncols,
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    with open(os.path.join(RESULTS_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    # Save a simple one-row version for quick inspection
    pd.DataFrame([{"sheet": s} for s in sheets]).to_csv(
        os.path.join(RESULTS_DIR, "workbook_sheets.csv"), index=False
    )
    pd.DataFrame(
        {"N_rows": [N], "N_columns": [ncols], "sheet": [sheets[0] if sheets else None]}
    ).to_csv(os.path.join(RESULTS_DIR, "data_dimensions.csv"), index=False)

    # ------------------------------------------------------------------
    # 2. Column information
    # ------------------------------------------------------------------
    col_info = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
            "n_missing": [df[c].isna().sum() for c in df.columns],
        }
    )
    col_info.to_csv(os.path.join(RESULTS_DIR, "column_information.csv"), index=False)

    # ------------------------------------------------------------------
    # 3. Missing values
    # ------------------------------------------------------------------
    missing = pd.DataFrame(
        {
            "column": df.columns,
            "n_missing": [df[c].isna().sum() for c in df.columns],
            "pct_missing": [df[c].isna().mean() * 100 for c in df.columns],
        }
    )
    missing.to_csv(os.path.join(RESULTS_DIR, "missing_values.csv"), index=False)

    # ------------------------------------------------------------------
    # 4. Duplicate rows
    # ------------------------------------------------------------------
    n_dup_total = df.duplicated().sum()
    dup_info = pd.DataFrame(
        {
            "check": ["all_columns", "demographics_only"],
            "n_duplicates": [n_dup_total, df[DEMOGRAPHICS].duplicated().sum()],
        }
    )
    dup_info.to_csv(
        os.path.join(RESULTS_DIR, "duplicate_information.csv"), index=False
    )

    # Save duplicate row content (report, not remove)
    dup_mask = df.duplicated(keep=False)
    df[dup_mask].to_csv(
        os.path.join(RESULTS_DIR, "duplicate_rows.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 5. First / last 5 rows
    # ------------------------------------------------------------------
    df.head(5).to_csv(os.path.join(RESULTS_DIR, "first_5_rows.csv"), index=False)
    df.tail(5).to_csv(os.path.join(RESULTS_DIR, "last_5_rows.csv"), index=False)

    # ------------------------------------------------------------------
    # 6. Item-level ranges (expected 1–5 Likert)
    # ------------------------------------------------------------------
    item_range_rows = []
    invalid_value_rows = []
    value_ranges_rows = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Value ranges for all numeric columns
    for c in numeric_cols:
        value_ranges_rows.append(
            {
                "column": c,
                "min": df[c].min(),
                "max": df[c].max(),
                "mean": df[c].mean(),
                "median": df[c].median(),
                "sd": df[c].std(),
            }
        )
    pd.DataFrame(value_ranges_rows).to_csv(
        os.path.join(RESULTS_DIR, "value_ranges.csv"), index=False
    )

    # Psychometric item ranges with full stats
    for col in ALL_ITEMS:
        if col not in df.columns:
            item_range_rows.append(
                {"item": col, "status": "MISSING_FROM_DATA", "N": 0}
            )
            continue

        s = df[col]
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        valid = s.dropna()

        item_range_rows.append(
            {
                "item": col,
                "N": int(len(s)),
                "n_missing": n_missing,
                "n_unique": n_unique,
                "min": float(valid.min()) if len(valid) else np.nan,
                "max": float(valid.max()) if len(valid) else np.nan,
                "mean": float(valid.mean()) if len(valid) else np.nan,
                "SD": float(valid.std()) if len(valid) else np.nan,
            }
        )

        # Report invalid Likert values (outside 1–5)
        if len(valid):
            bad_mask = (valid < LIKERT_MIN) | (valid > LIKERT_MAX)
            n_bad = int(bad_mask.sum())
            bad_vals = valid[bad_mask].tolist()
            invalid_value_rows.append(
                {
                    "item": col,
                    "n_invalid": n_bad,
                    "invalid_values": bad_vals if n_bad > 0 else [],
                }
            )

    pd.DataFrame(item_range_rows).to_csv(
        os.path.join(RESULTS_DIR, "item_ranges.csv"), index=False
    )

    invalid_df = pd.DataFrame(invalid_value_rows)
    if not invalid_df.empty:
        invalid_df.to_csv(
            os.path.join(RESULTS_DIR, "impossible_values.csv"), index=False
        )

    # ------------------------------------------------------------------
    # 7. Categorical levels for demographics
    # ------------------------------------------------------------------
    cat_levels = []
    for c in DEMOGRAPHICS:
        if c in df.columns:
            vc = df[c].value_counts(dropna=True)
            for lvl, cnt in vc.items():
                cat_levels.append(
                    {"variable": c, "level": lvl, "count": int(cnt)}
                )
    pd.DataFrame(cat_levels).to_csv(
        os.path.join(RESULTS_DIR, "categorical_levels.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 8. Summary report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 01 — DATA INSPECTION COMPLETE")
    print("=" * 60)
    print(f"Rows: {N}, Columns: {ncols}")
    print(f"Sheets: {sheets}")
    print(f"Total duplicate rows (all cols): {n_dup_total}")

    invalid_summary = invalid_df.copy() if not invalid_df.empty else pd.DataFrame()
    if not invalid_df.empty:
        total_bad = invalid_df["n_invalid"].sum()
        print(f"Invalid Likert values (outside 1–5): {total_bad}")

    print("\nKey outputs:")
    for fn in [
        "dataset_metadata.json",
        "data_dimensions.csv",
        "column_information.csv",
        "missing_values.csv",
        "duplicate_information.csv",
        "duplicate_rows.csv",
        "item_ranges.csv",
        "value_ranges.csv",
        "categorical_levels.csv",
    ]:
        path = os.path.join(RESULTS_DIR, fn)
        print(f"  {path} — {'OK' if os.path.exists(path) else 'MISSING'}")

    print(f"\nEnvironment: seed={SEED}")
    print(f"pandas {pd.__version__}, numpy {np.__version__}")


if __name__ == "__main__":
    main()
