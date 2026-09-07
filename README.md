# Paper 1 — The Intention–Behaviour Gap in Household Energy-Saving Behaviour

Fully reproducible computational pipeline for a person-centred (Latent Profile
Analysis) study of the intention–behaviour gap in household energy-saving
behaviour, N = 1,166 survey respondents.

**Primary solution: K = 7 profiles** (full-covariance Gaussian mixture, BIC-selected).
The leading competing solution (K = 6) is retained as a frozen robustness check.

## Repository layout

```
data.xls                      Raw (de-identified, coded) survey data, 1 sheet, 1,166 x 34
scripts/                      Phased pipeline, 01–26 + build/figure scripts
  REPRODUCE_ALL.py            <- run this (see below)
scripts/27_task7_k7_diagnostics_cv.py   K=7 diagnostics + 5-fold CV (supplementary)
FULL_PIPELINE.py              Alternative single-file reproduction (same outputs)
scripts/k7_primary_analyses.py          K=7-primary evidence (gap decomposition, predictors, duplicate refit)
results/                      Frozen computational evidence package (all CSVs the
                              manuscripts cite). Delete this folder to regenerate
                              everything from scratch.
paper1_final_manuscript_k7.html   Primary submission-ready manuscript (12 figures, embedded)
paper1_final_manuscript.html      Main-manuscript variant (10 figures, embedded)
paper1_full_manuscript.html       Full-length manuscript: RQs, gaps, detailed
                                  methodology, results, discussion (5 tables, 10 figures)
TODO.md                       Phase checklist
```

## Quick start

```bash
git clone https://github.com/<OWNER>/<REPO>.git
cd <REPO>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full reproduction (writes results/, figures, all three manuscripts)
python3 scripts/REPRODUCE_ALL.py
```

Runtime: roughly 1–2 hours on a modern laptop. The heavy cost is
`n_init=1000` Gaussian-mixture refits, 200-replicate bootstraps, and 5-fold
cross-validation. Everything is seeded (`random_state=42`) — reruns reproduce
the frozen `results/` bit-for-bit up to floating-point noise.

### Fast paths

```bash
# Rebuild only the figures + manuscripts from the frozen evidence (~2 min)
python3 scripts/generate_draft_figures.py && python3 scripts/build_k7_manuscript.py
python3 scripts/generate_figures_paper1.py && python3 scripts/build_paper1_manuscript.py
python3 scripts/build_full_manuscript.py   # needs results/draft_figures/ from the first line

# Or the single-file variant
python3 FULL_PIPELINE.py          # requires data.xls in the working directory
```

## What "reproduce" means here

- Every number in the three HTML manuscripts is substituted at build time from a
  CSV in `results/` — nothing in the text is hand-typed.
- Scripts read the selected K, sample N, duplicate counts, bootstrap count,
  and extended-K range **dynamically from the frozen result files**; no
  constants are hardcoded in the analysis code.
- `results/paper1_final/` was frozen when the evidence package was sealed.
  Phase scripts validate their fresh output against these frozen files and
  fail loudly on mismatch (see `scripts/k7_primary_analyses.py` validation
  blocks, `scripts/11_final_verification.py`, `scripts/22_claim_audit.py`).

## Methods summary

| Step | Method |
|---|---|
| Measurement | Mean-score composites; Cronbach's alpha per construct |
| Gap | GAP = z(INT) − z(BE); distribution + variance decomposition |
| LPA | `sklearn.mixture.GaussianMixture`, full covariance, `n_init=1000`, `max_iter=500`, `reg_covar=1e-6`, seed 42, K = 2…10 |
| Selection | BIC among numerically well-behaved fits (positive-log-likelihood Ks excluded); cross-validated held-out log-likelihood |
| Profiles | Descriptives, Welch tests + Cohen's d, directional stability (200 bootstraps) |
| Predictors | Multinomial logistic regression, Benjamini–Hochberg FDR (α = 0.05), VIF diagnostics |
| Robustness | K = 6 competitor, diagonal vs full covariance, duplicate-row refit (N = 1,124), seed sensitivity |

## Known caveats (stated in the manuscripts)

- Profile P4's directional stability is near chance (59%) — reported as such.
- The diagonal-covariance specification attains a lower BIC at some K; this is
  disclosed and shown in Figures 5–6.
- K ≥ 8 fits are numerically degenerate (positive log-likelihood) and are
  excluded from selection.
- Predictor analysis exhibits severe multicollinearity (max VIF reported in
  Table S outputs); cross-sectional design — no causal claims.

## Data

`data.xls` contains de-identified, coded survey responses (Likert 1–5 item
responses and coded demographics). No direct identifiers are present. Items
marked `[AUTHOR INPUT REQUIRED]` / `[LITERATURE SUPPORT NEEDED]` in the
manuscripts flag content only the authors can supply (ethics approval,
sampling frame, citations).

## Requirements

Python ≥ 3.10. Exact versions in `requirements.txt` (tested on Python 3.14,
pandas 3.0.5, scikit-learn 1.9.0).
