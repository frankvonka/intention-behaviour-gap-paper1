# 19-PHASE COMPUTATIONAL ANALYSIS (10-22)
## Intention–Behaviour Gap in Household Energy-Saving Behaviour

[x] Phase 01 — Data inspection
[x] Phase 02 — Measurement
[x] Phase 03 — Intention-behaviour gap
[x] Phase 04 — LPA estimation
[x] Phase 05 — LPA selection
[x] Phase 06 — Profile characterization
[x] Phase 07 — Profile predictors
[x] Phase 08 — Robustness
[x] Phase 09 — Numerical audit
[x] Phase 10 — Final frozen results
[x] Phase 11 — Post-verification
[x] Phase 12 — Paper 1 evidence package
[x] Phase 13 — K=6 profile validation
[x] Phase 14 — K=6 profile stability
[x] Phase 15 — K=2 to K=6 profile structure comparison
[x] Phase 16 — Gap profile validation
[x] Phase 17 — Alternative model validation
[x] Phase 18 — Profile predictor analysis
[x] Phase 18B — Predictor multicollinearity audit
[x] Phase 19 — Final numerical audit and results freeze
    CHECKS: PASS — 13/13 numerical checks PASS
[x] Phase 20 — Final evidence extraction
[x] Phase 21 — Final paper table data
[x] Phase 22 — Final claim audit
    CHECKS: 12 SUPPORTED, 2 CAVETED, 8 NOT_SUPPORTED
[x] Phases 20-22 — FINAL EVIDENCE PACKAGING
    STATUS: COMPLETE

---

# PAPER 1 — REMAINING HUMAN TASKS

The computational evidence is frozen and organized in
`results/paper1_final/`. The following tasks require human
research input and CANNOT be completed by the computation agent.

## REQUIRED BY HUMAN

1. **Literature review & citation insertion**
   - Every [LITERATURE SUPPORT NEEDED] placeholder in the draft
     requires a search of the relevant literature and insertion
     of proper citations.
   - No literature search was performed by the agent.

2. **Theoretical framing**
   - Every [INTERPRETATION REQUIRES REVIEW] marker identifies an
     interpretation that cannot be supported by numerical evidence
     alone and requires theoretical validation by the research team.
   - The Theory of Planned Behaviour is named in the draft but is
     NOT cited and [METHOD DETAIL NEEDS CONFIRMATION] markers remain.

3. **Final interpretation**
   - Draft Section 14 (Discussion), 15 (Theoretical Contribution),
     and 16 (Practical Implications) contain reserved interpretations
     that must be reviewed and confirmed by the research team.

4. **Figures / tables**
   - No figures or plots were created (per constraint).
   - Journal formatting, table numbering, and figure placement
     are deferred to the human author.

5. **Methodological confirmation**
   - [METHOD DETAIL NEEDS CONFIRMATION] markers indicate points where
     the exact published methodological citation is missing from the
     frozen results and must be confirmed by the human researcher.

6. **Unsupported claims verification**
   - Review the 8 NOT_SUPPORTED claims in
     `08_audit/03_unsupported_claims.csv`.
   - Ensure none are presented as findings in the final paper.

## VERIFICATION CHECKLIST (for human review)

Before submission:
- [ ] All [LITERATURE SUPPORT NEEDED] markers resolved
- [ ] All [INTERPRETATION REQUIRES REVIEW] markers validated
- [ ] All [METHOD DETAIL NEEDS CONFIRMATION] markers resolved
- [ ] All 8 NOT_SUPPORTED claims (Phase 22) excluded from findings
- [ ] The 2 CAVETATED claims (Phase 22) retain their caveats:
    - C8: Profile 5 direction-stability = 48% (low; not robust)
    - C13: Between-profile GAP variance = 26.6% (substantial within variance)
- [ ] All numerical values traceable to `09_master/paper1_master_evidence.csv`
