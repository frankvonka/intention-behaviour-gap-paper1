# Reviewer-Fix Pass — Task 5: Matching Distance Definition (Source Verified)

## Status: FIXED BY DOCUMENTATION

## Definition (from source code)

Verified verbatim from `scripts/14_k6_stability.py`, function `match_profiles(ref_means, boot_means)` (lines 88–103):

```python
def match_profiles(ref_means, boot_means):
    """
    One-to-one matching of bootstrap profiles to reference profiles
    using minimum Euclidean distance on (mean z_INT, mean z_BE).
    Hungarian algorithm (scipy.optimize.linear_sum_assignment).
    Returns (assignment, cost) where assignment[ref_profile] = boot_profile.
    """
    cost_mat = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost_mat[i, j] = np.linalg.norm(ref_means[i] - boot_means[j])
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    assignment = np.empty(K, dtype=int)
    assignment[row_ind] = col_ind
    total_cost = float(cost_mat[row_ind, col_ind].sum())
    return assignment, total_cost
```

### Precise definition

1. **Profile representation.** Each profile is represented by its 2-dimensional mean vector  
   **`(mean z_INT, mean z_BE)`** — the component means in the standardized latent space
   (`GaussianMixture.means_`, shape `(K, 2)`).

2. **Pairwise cost.** For every reference profile `i` and candidate bootstrap profile `j`,
   the cost is the **Euclidean (L2) distance** between the two 2D mean vectors:
   ```
   cost(i, j) = || ref_means[i] − boot_means[j] ||₂
               = sqrt( (Δmean_z_INT)² + (Δmean_z_BE)² )
   ```
   i.e. the Pythagorean distance in the (z_INT, z_BE) plane.

3. **Aggregation.** The cost matrix `C[i,j]` is `K×K`. The optimal one-to-one assignment is
   found with the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`).

4. **Total matching cost.** After the optimal assignment is found, the reported scalar
   "matching cost" for a given bootstrap replication/split is the **sum of the K selected
   pairwise Euclidean distances**:
   ```
   total_matching_cost = Σ_{i=0}^{K−1} cost(i, assignment[i])
   ```
   It is a *sum* of K Euclidean distances, not a mean, not a maximum, and not a
   correlation-based cost (no Procrustes, no Wasserstein, no symmetric difference).

5. **Direction convention.** `assignment[ref_profile] = boot_profile`. The matched bootstrap
   profile for reference profile `i` is `assignment[i]`. Bootstrap means are reordered by
   this assignment before any downstream stability aggregation (profile means, sizes), so
   that "profile p" consistently refers to the *same original profile* across all B=200
   bootstrap samples.

## Where it is used

- **Bootstrap stability** (Section 1, `scripts/14_k6_stability.py`): `match_profiles(ref_means, res["means"])`
  produces `matching_cost` per bootstrap replication. Reported mean = 2.599 over B=200
  (see Section 9 of the frozen results).
- **Split-sample stability** (Section 5): `match_profiles(halves["half_1"]["means"], halves["half_2"]["means"])`
  produces the split-sample matching cost = 2.713 (see Section 10 of the frozen results).
- Profiles are re-standardized *within each bootstrap/sample* before fitting (line 134–136),
  then matched to the *original* reference means which themselves were computed on the
  full-sample standardized indicators — see lines 116–122.

## Important caveats for interpretation

- The distance is in **standardized (z-score) units** of the two indicators, so the
  matching cost is scale-free and comparable across bootstrap replications (the within-sample
  standardization makes the per-replication metrics unitless).
- The cost is a *sum of K=6 Euclidean distances*; absolute magnitude is therefore not
  directly interpretable as a per-profile distance without dividing by K. For comparison
  purposes across B, the sum is consistent and monotonically ordered.
- Matching is **deterministic given the fit** (it is an argmin of a fixed cost matrix via
  `linear_sum_assignment`); any variation in total matched cost across replications
  reflects variation in the *estimated means* from bootstrap sampling, not variation in the
  matching procedure itself.
- The procedure is **symmetric in role**: reference ← matched-to bootstrap. It is *not*
  a goodness-of-fit test; it is a profile-alignment cost.

## Reproducibility

```
scipy.optimize.linear_sum_assignment
numpy.linalg.norm(ref_means[i] - boot_means[j])   # Euclidean, axis=None
```
Fixed by Hungarian assignment on the K×K Euclidean cost matrix of mean (z_INT, z_BE) vectors.
Total = sum of selected diagonals.

## Verification status

- ✅ Definition read verbatim from `scripts/14_k6_stability.py:match_profiles`.
- ✅ Matches frozen reports: bootstrap mean cost ≈ 2.599, split cost ≈ 2.713 (Section 9/10).
- ✅ Consistent with README.md of Phase 14 ("match-to-reference by minimum Euclidean
      distance on the 2D vector (mean z_INT, mean z_BE) … Hungarian algorithm").
