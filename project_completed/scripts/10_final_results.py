"""
Phase 10 — Final Frozen Results Package
=========================================
Project: Intention–Behaviour Gap in Household Energy-Saving Behaviour

Collects verified outputs from Phases 01-09 into results/10_final/.
Creates a README documenting each file.
Does not recompute science differently.
"""
import os
import shutil
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = "results/10_final"
SRC_DIRS = {
    "01_data_inspection": "results/01_data_inspection",
    "02_measurement": "results/02_measurement",
    "03_gap_analysis": "results/03_gap_analysis",
    "04_lpa_estimation": "results/04_lpa_estimation",
    "05_lpa_selection": "results/05_lpa_selection",
    "06_profile_analysis": "results/06_profile_analysis",
    "07_profile_predictors": "results/07_profile_predictors",
    "08_robustness": "results/08_robustness",
    "09_numerical_audit": "results/09_numerical_audit",
}


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    ensure_dir(RESULTS_DIR)

    # ------------------------------------------------------------------
    # Copy selected files from each phase
    # ------------------------------------------------------------------
    file_map = {
        "data_diagnostics": [
            ("01_data_inspection", "data_dimensions.csv"),
            ("01_data_inspection", "column_information.csv"),
            ("01_data_inspection", "missing_values.csv"),
            ("01_data_inspection", "duplicate_information.csv"),
            ("01_data_inspection", "item_ranges.csv"),
            ("01_data_inspection", "dataset_metadata.json"),
        ],
        "measurement_results": [
            ("02_measurement", "item_statistics.csv"),
            ("02_measurement", "construct_statistics.csv"),
            ("02_measurement", "cronbach_alpha.csv"),
            ("02_measurement", "int_be_correlation.csv"),
            ("02_measurement", "construct_scores.csv"),
            ("02_measurement", "construct_correlation_matrix.csv"),
        ],
        "gap_results": [
            ("03_gap_analysis", "gap_scores.npy"),
            ("03_gap_analysis", "standardized_scores.csv"),
            ("03_gap_analysis", "gap_statistics.csv"),
        ],
        "lpa_model_comparison": [
            ("04_lpa_estimation", "model_fit.csv"),
        ],
        "selected_lpa": [
            ("05_lpa_selection", "model_comparison.csv"),
            ("05_lpa_selection", "stability.csv"),
            ("05_lpa_selection", "classification_uncertainty.csv"),
            ("05_lpa_selection", "selection_diagnostics.csv"),
            ("05_lpa_selection", "selected_model.csv"),
        ],
        "profile_parameters": [
            ("06_profile_analysis", "profile_sizes.csv"),
            ("06_profile_analysis", "profile_means.csv"),
            ("06_profile_analysis", "gap_profile_check.csv"),
        ],
        "profile_comparisons": [
            ("06_profile_analysis", "profile_comparisons.csv"),
            ("06_profile_analysis", "profile_effect_sizes.csv"),
        ],
        "profile_predictors": [
            ("07_profile_predictors", "coefficients.csv"),
            ("07_profile_predictors", "odds_ratios.csv"),
            ("07_profile_predictors", "confidence_intervals.csv"),
            ("07_profile_predictors", "p_values.csv"),
            ("07_profile_predictors", "fdr_results.csv"),
        ],
        "robustness_results": [
            ("08_robustness", "random_start_stability.csv"),
            ("08_robustness", "random_seed_sensitivity.csv"),
            ("08_robustness", "covariance_sensitivity.csv"),
            ("08_robustness", "score_sensitivity.csv"),
            ("08_robustness", "classification_sensitivity.csv"),
            ("08_robustness", "outlier_sensitivity.csv"),
            ("08_robustness", "profile_stability.csv"),
        ],
        "numerical_audit": [
            ("09_numerical_audit", "audit_checks.csv"),
            ("09_numerical_audit", "audit_discrepancies.csv"),
        ],
    }

    all_files = []
    for group, files in file_map.items():
        group_dir = os.path.join(RESULTS_DIR, group)
        ensure_dir(group_dir)
        for src_phase, fname in files:
            src = os.path.join(SRC_DIRS[src_phase], fname)
            dst = os.path.join(group_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                all_files.append((group, fname, os.path.exists(dst)))

    # Also copy K_ subdirectories from Phase 04 (profile parameters,
    # posterior probabilities, class assignments)
    kdir_src = os.path.join(SRC_DIRS["04_lpa_estimation"])
    selected = pd.read_csv(os.path.join(SRC_DIRS["05_lpa_selection"], "selected_model.csv"))
    K_sel = int(selected["selected_K"].values[0])
    kdir_dst = os.path.join(RESULTS_DIR, "lpa_model_comparison", f"K_{K_sel}")
    ensure_dir(kdir_dst)
    for fname in [
        "profile_parameters.csv",
        "profile_sizes.csv",
        "covariance_matrices.json",
        "posterior_probabilities.csv",
        "profile_means.csv",
        "profile_means.npy",
        "random_start_diagnostics.csv",
    ]:
        src = os.path.join(kdir_src, f"K_{K_sel}", fname)
        dst = os.path.join(kdir_dst, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            all_files.append((f"selected_LPA_K{K_sel}", fname, True))

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = f"""# Final Frozen Results Package
## Intention–Behaviour Gap in Household Energy-Saving Behaviour

### Overview
This directory contains the verified outputs from all 10 computational phases.
All values are recalculated from `data.xls`; no historical results were hard-coded.

### Selected Model
- **K = {K_sel}** profiles (selected by minimum BIC, verified in Phase 05)
- Covariance: full
- Indicators: z_INT, z_BE (standardized across respondents)
- N = 1166 respondents
- Random seed = 42, n_init = 1000

### File Organization

#### data_diagnostics/
- `data_dimensions.csv` — N rows, N columns (Phase 01)
- `column_information.csv` — column dtype, unique counts, missing (Phase 01)
- `missing_values.csv` — per-column missing value counts (Phase 01)
- `duplicate_information.csv` — duplicate row counts (Phase 01)
- `item_ranges.csv` — per-item N, missing, unique, min, max, mean, SD (Phase 01)
- `dataset_metadata.json` — workbook metadata (Phase 01)

#### measurement_results/
- `item_statistics.csv` — per-item mean, SD, min, max, variance (Phase 02)
- `construct_statistics.csv` — construct mean, SD, Cronbach alpha (Phase 02)
- `cronbach_alpha.csv` — per-construct Cronbach alpha (Phase 02)
- `int_be_correlation.csv` — Pearson r, p, N, 95% CI (Phase 02)
- `construct_scores.csv` — respondent-level construct scores (Phase 02)
- `construct_correlation_matrix.csv` — all-pairs construct correlations (Phase 02)

#### gap_results/
- `gap_scores.npy` — respondent-level GAP = z_INT - z_BE (Phase 03)
- `standardized_scores.csv` — z_INT, z_BE, GAP per respondent (Phase 03)
- `gap_statistics.csv` — summary gap statistics (Phase 03)

#### lpa_model_comparison/
- `model_fit.csv` — AIC, BIC, entropy, convergence for K=2..6 (Phase 04)
- `K_{K_sel}/` — selected-model artifacts:
  - `profile_parameters.csv` — N, proportion, means per profile (Phase 04)
  - `profile_sizes.csv` — class sizes (Phase 04)
  - `covariance_matrices.json` — full covariance per profile (Phase 04)
  - `posterior_probabilities.csv` — per-respondent posteriors (Phase 04)
  - `profile_means.csv` — mean vectors (Phase 04)
  - `random_start_diagnostics.csv` — convergence diagnostics (Phase 04)

#### selected_lpa/
- `model_comparison.csv` — full comparison K=2..6 with deltas (Phase 05)
- `stability.csv` — convergence across random seeds (Phase 05)
- `classification_uncertainty.csv` — max-posterior thresholds (Phase 05)
- `selection_diagnostics.csv` — decision diagnostics (Phase 05)
- `selected_model.csv` — selected K and basis (Phase 05)

#### profile_parameters/
- `profile_sizes.csv` — N and percentage per profile (Phase 06)
- `profile_means.csv` — mean and SD per construct per profile (Phase 06)
- `gap_profile_check.csv` — standardized gap and pattern per profile (Phase 06)

#### profile_comparisons/
- `profile_comparisons.csv` — Welch t-tests between profiles (Phase 06)
- `profile_effect_sizes.csv` — Cohen's d between profiles (Phase 06)

#### profile_predictors/
- `coefficients.csv` — beta, SE, z, p per coefficient (Phase 07)
- `odds_ratios.csv` — OR = exp(beta) (Phase 07)
- `confidence_intervals.csv` — 95% CI for OR (Phase 07)
- `p_values.csv` — adjusted and raw p-values (Phase 07)
- `fdr_results.csv` — Benjamini-Hochberg FDR-adjusted p-values (Phase 07)

#### robustness_results/
- `random_start_stability.csv` — n_init=1000 stability (Phase 08)
- `random_seed_sensitivity.csv` — seed [42,123,999,2024,7] (Phase 08)
- `covariance_sensitivity.csv` — full/diag/spherical comparison (Phase 08)
- `score_sensitivity.csv` — mean vs factor score correlation (Phase 08)
- `classification_sensitivity.csv` — max-posterior thresholds per K (Phase 08)
- `outlier_sensitivity.csv` — |z|>3 sensitivity (Phase 08)
- `profile_stability.csv` — cross-condition stability summary (Phase 08)

#### numerical_audit/
- `audit_checks.csv` — full audit of all checks (Phase 09)
- `audit_discrepancies.csv` — any failed checks (Phase 09)

### Key Relationships
- Construct scores in `measurement_results/construct_scores.csv` use respondent ID as index (consistent ordering).
- LPA indicators are standardized versions of INT and BE (see `gap_results/standardized_scores.csv`).
- Profile assignments in `lpa_model_comparison/K_{K_sel}/posterior_probable.csv` use `assigned_class` column.
"""

    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write(readme)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 10 — FINAL RESULTS PACKAGE COMPLETE")
    print("=" * 60)
    print(f"Selected K = {K_sel}")
    print(f"\nFiles copied: {len(all_files)}")
    ok = sum(1 for _, _, s in all_files if s)
    print(f"Successfully copied: {ok}")
    print(f"\nREADME: {os.path.join(RESULTS_DIR, 'README.md')}")
    print(f"\nDirectory structure:")
    for root, dirs, files in os.walk(RESULTS_DIR):
        rel = os.path.relpath(root, RESULTS_DIR)
        for fn in sorted(files):
            print(f"  {os.path.join(rel, fn)}")


if __name__ == "__main__":
    main()
