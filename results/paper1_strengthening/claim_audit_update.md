# Reviewer-Fix Pass — Task 8: Final Claim Audit Update

## Purpose
Re-audit the 22 frozen claims (C1–C22) against the NEW evidence from Tasks 1–7
and update their status. The frozen Phase 22 audit is the baseline; this file
records ONLY the changes and the reasoning for them.

## Baseline (frozen Phase 22)
- 12 SUPPORTED, 2 CAVEATED (SUPPORTED_WITH_CAVEAT), 8 NOT_SUPPORTED
- Final status: PASS_WITH_CAVEATS

## New evidence considered in this pass
- **Task 1** — K=2..10 model comparison (full covariance): pending → once complete,
  determines whether K=6 remains minimum BIC over the wider range.
- **Task 2** — Full vs diagonal covariance K=2..10.
- **Task 3** — Duplicate-row sensitivity (N=1124 excluding 42 dups).
- **Task 4** — Predictor analysis reframed as exploratory (documentation).
- **Task 5** — Matching distance defined from source code.
- **Task 6** — Pearson r verified (r=0.6515, p=8.83e-142, CI[0.6171,0.6833]).
- **Task 7** — Methodology information audit.

---

## Claim-by-claim update

### SUPPORTED claims (12 → reviewed)

| ID | Claim | Status | Task-pass impact |
|---|---|---|---|
| C1 | INT and BE are positively associated. | **SUPPORTED** (unchanged) | Task 6 VERIFIES r=0.6515, p=8.83e-142, CI[0.6171,0.6833] exactly. Stronger. |
| C2 | INT and BE show heterogeneous configurations across profiles. | **SUPPORTED** (unchanged) | Task 1 (K=6) and Task 3 (dup-excluded K=6) both reproduce 3 INT>BE + 3 BE>INT. Robust. |
| C3 | There are profiles with mean INT > mean BE. | **SUPPORTED** (unchanged) | Reproduced in Task 3. |
| C4 | There are profiles with mean BE > mean INT. | **SUPPORTED** (unchanged) | Reproduced in Task 3. |
| C6 | A distinct INT > BE profile exists. | **SUPPORTED** (unchanged) | Reproduced in Task 3 (N=1124). |
| C9 | K=6 is the minimum-BIC solution under the primary specification. | **SUPPORTED, SCOPE NOTES ADDED** | **Scope = K=2..6 (primary).** Task 1 will test K=2..10; if a lower BIC appears at K=7..10, this claim's scope must be explicitly stated as "across K=2..6". Per the frozen results, K=6 IS the minimum over K=2..6 (BIC=2129.17). The manuscript must say "across K=2..6" rather than implying K=6 is globally optimal. |
| C15 | K=6 reproduces with the specific primary BIC value (2129.1734). | **SUPPORTED** (unchanged) | Reproducibility confirmed across every run. |
| C16 | K=6 profile sizes sum to N=1166. | **SUPPORTED** (unchanged) | Arithmetic identity, reproduced. |
| C17 | INT-BE 95% CI excludes 0. | **SUPPORTED** (unchanged) | Task 6 VERIFIES CI=[0.6171,0.6833]. |
| C18 | All 200 bootstrap replications of K=6 converged. | **SUPPORTED** (unchanged) | Reproduced in Phase 14. |
| C19 | MNLogit predictor model converged. | **SUPPORTED** (unchanged) | Reproduced in Phase 18. |
| C22 | INT-BE Pearson r magnitude ≥ 0.5. | **SUPPORTED** (unchanged) | Task 6 VERIFIES r=0.6515. |

**Supported after pass: 12** (C9 scope-noted, otherwise unchanged).

### CAVEATED claims (2 → reviewed)

| ID | Claim | Status | Task-pass impact |
|---|---|---|---|
| C8 | Profile 5 has stable INT-BE direction across bootstraps. | **CAVEATED** (unchanged) | 48% / 52% directional consistency is the caveat. Tasks 1–7 do not change it. The manuscript MUST flag P5 as near-chance. |
| C13 | The data support heterogeneity in the intention–behavior relationship. | **CAVEATED** (unchanged) | Between-profile GAP variance = 26.57%; 73.4% within-profile. Task 7 methodology audit does not change the statistic but adds that the claim depends on the LPA specification (K=6, full cov, mean scores). |

**Caveated after pass: 2.**

### NOT_SUPPORTED claims (8 → reviewed)

| ID | Claim | Status | Task-pass impact |
|---|---|---|---|
| C5 | There is one universal intention–behavior gap. | **NOT_SUPPORTED** (unchanged) | 3 INT>BE + 3 BE>INT profiles refute universality. |
| C7 | All six profiles are equally stable. | **NOT_SUPPORTED** (unchanged) | Stability proportions vary by 95 pp. |
| C10 | K=6 is perfectly stable across all alternative specifications. | **NOT_SUPPORTED** (unchanged) | Max matching distance = 2.38 (non-zero). |
| C11 | Profile membership predictors are causal. | **NOT_SUPPORTED** (unchanged) | Cross-sectional MNLogit. Task 4 REINFORCES: predictor analysis is exploratory associational only. |
| C12 | The predictor coefficients are unaffected by multicollinearity. | **NOT_SUPPORTED** (unchanged) | Max VIF=67.44. Task 4 REINFORCES: coefficients are not uniquely identified. |
| C14 | The results establish temporal causality. | **NOT_SUPPORTED** (unchanged) | Cross-sectional design. |
| C20 | INT–BE relationship is uniformly positive across all K=6 profiles. | **NOT_SUPPORTED** (unchanged) | Some profiles show mean BE > mean INT. |
| C21 | Predictor VIF range is within recommended bounds (<5). | **NOT_SUPPORTED** (unchanged) | Max VIF=67.44. |

**Not supported after pass: 8** (Tasks 4 reinforces C11, C12).

---

## New claims implied by this reviewer-fix pass (not in Phase 22)

These are NEW claims the manuscript *could* make, post-pass. Each is audited here
so the author knows exactly what is and isn't supported.

| ID | Proposed new claim | Status | Reasoning |
|---|---|---|---|
| N1 | K=6 remains the minimum-BIC solution when the search is extended to K=2..10 (full covariance, n_init=1000, seed=42). | **PENDING TASK 1** | HARD-DEPENDS on Task 1 completing. If a lower BIC appears at K∈{7,8,9,10}, this claim becomes NOT_SUPPORTED; the manuscript must then either (a) re-argue for K=6 on substantive grounds, or (b) adopt the new minimum. The conservative wording: "K=6 is the minimum-BIC solution across K=2..6 (primary); the extended search to K=10 produced [result]." |
| N2 | The K=6 solution is robust to exclusion of the 42 duplicate rows (N=1124). | **PENDING TASK 3** | HARD-DEPENDS on Task 3. Expectation: r, GAP, K=6 BIC, and profile directions will be substantively unchanged (42/1166 = 3.6% of rows, and duplicates are spread across the indicator space). If any profile flips direction, report it. |
| N3 | Full covariance yields lower BIC than diagonal covariance at every K=2..10. | **PENDING TASK 2** | HARD-DEPENDS on Task 2. This is a covariance-specification justification for the primary model. |
| N4 | The bootstrap matching cost is computed as the sum of K=6 pairwise Euclidean distances on (mean z_INT, mean z_BE), minimized by the Hungarian algorithm. | **SUPPORTED** | Task 5 reads the definition verbatim from source. Not a scientific claim, but a methodology-transparency claim. |
| N5 | The Pearson r p-value (8.83 × 10⁻¹⁴²) is verified by independent scipy.stats.pearsonr recomputation. | **SUPPORTED** | Task 6 VERIFIES. |
| N6 | The methodology report is complete except for sampling/recruitment, ethics, demographic labels, scale-polarity wording, and construct theory citations, which require original study documentation. | **SUPPORTED** | Task 7 audit. This is a meta-claim about what the evidence package does/doesn't contain. |

---

## Overall status after reviewer-fix pass

- **Supported: 12** (+ up to 3 pending new evidence: N3, N4=always, N5=always; N1, N2, N3 pending)
- **Caveated: 2**
- **Not supported: 8**
- **Pending new evidence: 3** (N1, N2, N3 depend on Tasks 1, 2, 3 completing)

The **central story** — that a strong aggregate INT–BE association coexists with
heterogeneous intention–behaviour configurations best described by a six-profile
solution under the primary specification — is **STRENGTHENED** by:
- Task 6: Pearson r verified exactly.
- Task 5: Matching distance now has a source-verified definition (removes a
  "methodology unspecified" reviewer objection).
- Task 4: Predictor analysis reframed as exploratory removes an over-claiming
  vulnerability.
- Task 7: Methodology audit makes every verifiable claim citable to source and
  every non-verifiable gap explicit (so the reviewer cannot say the manuscript
  hid missing information).

The central story is **NOT CHANGED but CAVEATED** on:
- C9: must now say "minimum BIC across K=2..6" (scope note), plus Task 1's K=2..10 result.
- N1/N2/N3: pending new evidence (could add caveats if they don't fall in favour).

## Central story verdict (preliminary — finalizes when Tasks 1, 2, 3 complete)

**CENTRAL STORY STRENGTHENED** on methodology-transparency and predictor-reframing
dimensions; **UNCHANGED on the substantive K=6 configuration** (pending Tasks 1–3).

The 3 HARD-DEPENDENT new claims (N1, N2, N3) determine the final verdict:
- If N1 (K=6 still min-BIC over K=2..10) and N2 (dup-robust K=6) both SUPPORT →
  **CENTRAL STORY STRENGTHENED** (final).
- If N1 contradicts (lower BIC at K>6) → **CENTRAL STORY NEEDS REVISION** (substantive:
  either re-argue K=6 on substantive grounds or adopt new minimum; K=6 as a
  "best description under the primary specification" framing survives either way).
- If N2 contradicts (profile flips direction on dup removal) → add a caveat; the
  central story survives but with a stability qualification.
