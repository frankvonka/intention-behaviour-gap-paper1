#!/usr/bin/env python3
"""
Phase 01: Data Inspection and Raw-Data Verification
Establishes exactly what is contained in data.xls.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Configuration
INPUT_FILE = "data.xls"
OUTPUT_DIR = "results/01_data_inspection"

def ensure_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def save_csv(df, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")

def save_json(data, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {filepath}")

def main():
    print("=" * 60)
    print("PHASE 01: DATA INSPECTION AND RAW-DATA VERIFICATION")
    print("=" * 60)

    ensure_output_dir()

    # Load the dataset
    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    print(f"Shape: {df.shape}")

    # 1. Data dimensions
    print("\n--- Data Dimensions ---")
    dims = pd.DataFrame({
        'metric': ['n_rows', 'n_columns'],
        'value': [df.shape[0], df.shape[1]]
    })
    print(dims)
    save_csv(dims, 'data_dimensions.csv')

    # 2. Column information
    print("\n--- Column Information ---")
    col_info = pd.DataFrame({
        'column_name': df.columns,
        'dtype': [str(dt) for dt in df.dtypes],
        'non_null_count': df.notna().sum().values,
        'null_count': df.isna().sum().values
    })
    print(col_info.to_string())
    save_csv(col_info, 'column_information.csv')

    # 3. Missing values
    print("\n--- Missing Values ---")
    missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isna().sum().values,
        'missing_pct': (df.isna().sum() / len(df) * 100).values
    })
    print(missing.to_string())
    save_csv(missing, 'missing_values.csv')

    # 4. Duplicate rows
    print("\n--- Duplicate Rows ---")
    dup_mask = df.duplicated()
    n_duplicates = dup_mask.sum()
    dup_info = pd.DataFrame({
        'metric': ['total_duplicates', 'unique_rows'],
        'value': [int(n_duplicates), int(len(df) - n_duplicates)]
    })
    print(dup_info)
    save_csv(dup_info, 'duplicate_information.csv')

    # Save duplicate rows if any
    if n_duplicates > 0:
        dup_rows = df[dup_mask].copy()
        dup_rows.to_csv(os.path.join(OUTPUT_DIR, 'duplicate_rows.csv'), index=False)
        print(f"Saved duplicate rows to duplicate_rows.csv")

    # 5. Value ranges for numeric columns
    print("\n--- Value Ranges (Numeric Columns) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    ranges_list = []
    for col in numeric_cols:
        ranges_list.append({
            'column': col,
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'median': df[col].median(),
            'q25': df[col].quantile(0.25),
            'q75': df[col].quantile(0.75)
        })
    ranges_df = pd.DataFrame(ranges_list)
    print(ranges_df.to_string())
    save_csv(ranges_df, 'value_ranges.csv')

    # 6. Value ranges for object/categorical columns
    print("\n--- Value Counts (Categorical Columns) ---")
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        vc = df[col].value_counts()
        vc_df = pd.DataFrame({'value': vc.index, 'count': vc.values})
        vc_df.to_csv(os.path.join(OUTPUT_DIR, f'value_counts_{col}.csv'), index=False)
        print(f"{col}: {len(vc)} unique values")

    # 7. Full dataset info as JSON
    print("\n--- Full Dataset Metadata ---")
    metadata = {
        'input_file': INPUT_FILE,
        'n_rows': int(df.shape[0]),
        'n_columns': int(df.shape[1]),
        'columns': list(df.columns),
        'dtypes': {col: str(dt) for col, dt in df.dtypes.items()},
        'missing_per_column': {col: int(df[col].isna().sum()) for col in df.columns},
        'total_missing': int(df.isna().sum().sum()),
        'duplicate_rows': int(n_duplicates),
        'numeric_columns': list(numeric_cols),
        'categorical_columns': list(object_cols),
        'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024**2)
    }
    save_json(metadata, 'dataset_metadata.json')

    # 8. First few rows
    print("\n--- First 5 Rows ---")
    print(df.head().to_string())
    df.head().to_csv(os.path.join(OUTPUT_DIR, 'first_5_rows.csv'), index=False)

    # 9. Last few rows
    print("\n--- Last 5 Rows ---")
    print(df.tail().to_string())
    df.tail().to_csv(os.path.join(OUTPUT_DIR, 'last_5_rows.csv'), index=False)

    # 10. Check for impossible values (negative values where not expected, etc.)
    print("\n--- Impossible Value Checks ---")
    impossible_checks = []
    for col in numeric_cols:
        col_data = df[col]
        if (col_data < 0).any():
            neg_count = (col_data < 0).sum()
            impossible_checks.append({
                'column': col,
                'issue': 'negative_values',
                'count': int(neg_count),
                'min_value': float(col_data.min())
            })
    if impossible_checks:
        impossible_df = pd.DataFrame(impossible_checks)
        print(impossible_df.to_string())
        save_csv(impossible_df, 'impossible_values.csv')
    else:
        print("No negative values found in numeric columns")
        save_csv(pd.DataFrame(columns=['column', 'issue', 'count', 'min_value']), 'impossible_values.csv')

    print("\n" + "=" * 60)
    print("PHASE 01 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()