#!/usr/bin/env python3
"""
REPRODUCE_ALL.py — Master reproduction script for Paper 1
========================================================
Runs the full analytic pipeline from raw data to final manuscript.
Order matters: each phase reads outputs from prior phases.

Usage:
    python3 scripts/REPRODUCE_ALL.py

Outputs:
    results/01_data_inspection/     Data quality reports
    results/02_measurement/         Construct scores and reliabilities
    results/03_gap_analysis/        GAP distribution statistics
    results/04_lpa/                 LPA estimation (K=2..10)
    results/05_profiles/            K=6 profile descriptives
    results/06_predictors/          Multinomial logistic regression
    results/07_robustness/          Alternative specification table
    results/paper1_final/            Frozen evidence package
    results/paper1_strengthening/    Sensitivity analyses
    paper1_final_manuscript.html     Submission-ready manuscript
"""

import os
import sys
import subprocess
import shutil

os.chdir(os.path.dirname(os.path.abspath(__file__ + "/..")))

SCRIPTS = [
    ("01_data_inspection.py", "Phase 01 — Data inspection"),
    ("02_measurement.py", "Phase 02 — Measurement and construct scores"),
    ("03_gap_analysis.py", "Phase 03 — Intention–behaviour gap analysis"),
    ("04_lpa_estimation.py", "Phase 04 — LPA estimation (K=2..10)"),
    ("05_lpa_selection.py", "Phase 05 — LPA model selection"),
    ("06_profile_analysis.py", "Phase 06 — Profile analysis"),
    ("07_profile_predictors.py", "Phase 07 — Profile-membership predictors"),
    ("08_robustness.py", "Phase 08 — Robustness analyses"),
    ("09_numerical_audit.py", "Phase 09 — Numerical audit"),
    ("10_final_results.py", "Phase 10 — Final results compilation"),
    ("11_final_verification.py", "Phase 11 — Final verification"),
    ("12_paper1_evidence.py", "Phase 12 — Paper 1 evidence package"),
    ("13_k6_profile_validation.py", "Phase 13 — K=6 profile validation"),
    ("14_k6_stability.py", "Phase 14 — K=6 stability analysis"),
    ("15_profile_structure_comparison.py", "Phase 15 — Profile structure comparison"),
    ("16_gap_profile_validation.py", "Phase 16 — GAP-profile validation"),
    ("17_alternative_model_validation.py", "Phase 17 — Alternative model validation"),
    ("18B_predictor_multicollinearity.py", "Phase 18B — Predictor multicollinearity"),
    ("18_profile_predictors.py", "Phase 18 — Profile predictors final"),
    ("19_final_audit.py", "Phase 19 — Final audit"),
    ("20_evidence_extraction.py", "Phase 20 — Evidence extraction"),
    ("21_paper_tables.py", "Phase 21 — Paper tables"),
    ("22_claim_audit.py", "Phase 22 — Claim audit"),
    ("23_task1_k_extended.py", "Task 1 — K-extended model comparison (K=2..10)"),
    ("24_task6_pearson_verify.py", "Task 6 — Pearson r verification"),
    ("25_task2_covariance_extended.py", "Task 2 — Covariance sensitivity (full vs diag)"),
    ("26_task3_duplicate_sensitivity.py", "Task 3 — Duplicate-row sensitivity"),
    ("build_master_evidence.py", "Master evidence compilation"),
]

FAILED = []

print("=" * 70)
print("PAPER 1 — FULL REPRODUCTION PIPELINE")
print("=" * 70)

for script_name, description in SCRIPTS:
    path = os.path.join("scripts", script_name)
    if not os.path.exists(path):
        print(f"\n[SKIP] {script_name} — file not found")
        continue

    print(f"\n{'─' * 60}")
    print(f"Running: {script_name}")
    print(f"  {description}")
    print(f"{'─' * 60}")

    result = subprocess.run(
        [sys.executable, path],
        capture_output=False,
        timeout=600,
    )

    if result.returncode != 0:
        FAILED.append(script_name)
        print(f"[FAIL] {script_name} exited with code {result.returncode}")
    else:
        print(f"[OK]   {script_name}")

# Generate figures and manuscript
print(f"\n{'─' * 60}")
print("Generating figures...")
print(f"{'─' * 60}")
result = subprocess.run([sys.executable, "scripts/generate_figures_paper1.py"])
if result.returncode != 0:
    FAILED.append("generate_figures_paper1.py")
else:
    print("[OK]   generate_figures_paper1.py")

print(f"\n{'─' * 60}")
print("Building manuscript...")
print(f"{'─' * 60}")
result = subprocess.run([sys.executable, "scripts/build_paper1_manuscript.py"])
if result.returncode != 0:
    FAILED.append("build_paper1_manuscript.py")
else:
    print("[OK]   build_paper1_manuscript.py")

# Extract PNGs from base64
print(f"\n{'─' * 60}")
print("Extracting PNG figures...")
print(f"{'─' * 60}")
import json, base64
os.makedirs("results/paper1_final/figures/png", exist_ok=True)
with open("results/paper1_final/figures/figures_base64.json") as f:
    figures = json.load(f)
for key, b64 in figures.items():
    fname = f"Figure_{int(key[3:])}.png"
    with open(f"results/paper1_final/figures/png/{fname}", "wb") as f:
        f.write(base64.b64decode(b64))
print("[OK]   PNGs extracted")

# Summary
print(f"\n{'=' * 70}")
if FAILED:
    print(f"COMPLETED WITH {len(FAILED)} FAILURES:")
    for f in FAILED:
        print(f"  ✗ {f}")
else:
    print("ALL PHASES COMPLETED SUCCESSFULLY")
print(f"{'=' * 70}")

print(f"\nDeliverables:")
print(f"  paper1_final_manuscript.html — Final manuscript")
print(f"  results/paper1_final/figures/png/Figure_1..10.png — Standalone figures")
print(f"  results/paper1_final/ — Frozen evidence package")
print(f"  results/paper1_strengthening/ — Sensitivity analyses")
