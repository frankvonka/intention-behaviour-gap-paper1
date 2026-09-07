# Phase 11 — Post-Verification
## Intention–Behaviour Gap in Household Energy-Saving Behaviour

### Purpose
Verify the final results before treating them as frozen.
No changes to primary analysis. No literature. No figures.

### Checks Performed
1. **audit_failure.csv** — Identifies the Phase 09 check that did not pass
2. **lpa_comparison.csv** — K=2..7 stability comparison (BIC, AIC, LL, entropy, sizes, uncertainty)
3. **profile_structure.csv** — Profile structure for K=2..7 (N, %, mean INT/BE, z, GAP)
4. **gap_profile_check.csv** — Numerical gap verification (INT>BE, BE>INT)
5. **profile_size_diagnostics.csv** — Size and uncertainty diagnostics
6. **factor_vs_mean.csv** — Factor score robustness (sklearn FactorAnalysis)
7. **k6_reproduction.csv** — K=7 reproducibility check (filename kept for compatibility)

### Key Findings
- Phase 09 failure: 1 check(s) failed (see audit_failure.csv)
- K=7 reproducibility: PASS
- Factor scores: computed via sklearn.decomposition.FactorAnalysis

### Scripts Used
- scripts/11_final_verification.py (this script)

### Results Used
- results/02_measurement/construct_scores.csv
- results/04_lpa_estimation/ (K=2..7)
- results/05_lpa_selection/ (classification_uncertainty.csv)
- results/10_final/numerical_audit/audit_checks.csv
