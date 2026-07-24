# Routing-policy proposal: Raise the Tier1/Tier2 criticality-spread threshold from 2.0 to 8.0

- **Proposal id:** `2026-07-24-raise-criticality-threshold`
- **Policy:** `CRITICALITY_SPREAD_THRESHOLD` -> 8.0
- **Verdict:** ACCEPTED -- non-regression: cost delta 0, physical-error exposure delta -698.7

## What this changes
- budget_router.CRITICALITY_SPREAD_THRESHOLD, the max/mean criticality spread at or above which route() selects Tier 2 (HW_CERTIFIED) over Tier 1 (HW_STANDARD), moves from 2.0 to 8.0.

## What it unlocks
- Problems with a moderately skewed criticality spectrum (roughly 2.0-8.0) that do not need surface-code ECC protection stop paying the Tier 2 pipeline path (ECC distance selection, patch-budget allocation) for no measured benefit against the current ledger.

## What it does not unlock
- Does not change backend selection, shot counts, or any other routing constant. Does not touch problems with spread below 2.0 (already Tier 1) or above 8.0 (still routed to Tier 2). Does not widen the set of gated policies beyond CRITICALITY_SPREAD_THRESHOLD.

## Ledger-backed replay
- **Baseline** total estimated cost: 700 credits, physical-error exposure: 6289
- **Proposed** total estimated cost: 700 credits, physical-error exposure: 5590

| scenario | baseline tier/backend | proposed tier/backend | changed |
|---|---|---|---|
| flat_small | T1/ibm_marrakesh | T1/ibm_marrakesh | no |
| flat_large | T1/ibm_marrakesh | T1/ibm_marrakesh | no |
| skewed_small | T2/ibm_marrakesh | T1/ibm_marrakesh | yes |
| skewed_large | T2/ibm_marrakesh | T2/ibm_marrakesh | no |
| skewed_high_fidelity | T2/ibm_marrakesh | T2/ibm_marrakesh | no |
