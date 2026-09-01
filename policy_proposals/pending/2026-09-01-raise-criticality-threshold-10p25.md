# Routing-policy proposal: Raise the Tier1/Tier2 criticality-spread threshold from 8.0 to 10.25

- **Proposal id:** `2026-09-01-raise-criticality-threshold-10p25`
- **Policy:** `CRITICALITY_SPREAD_THRESHOLD` -> 10.25
- **Verdict:** ACCEPTED -- accepted: all previously meeting scenarios still meet; cost delta 0, physical-error exposure delta 0

## What this changes
- budget_router.CRITICALITY_SPREAD_THRESHOLD moves from 8.0 to 10.25. This is the max/mean criticality spread at or above which route() selects Tier 2 (HW_CERTIFIED) over Tier 1 (HW_STANDARD).

## What it unlocks
- Aligns the live threshold with the largest value that keeps every DEFAULT_SCENARIOS case that is currently HW_CERTIFIED still HW_CERTIFIED under the same ledger-adjusted fleet. Replay shows no change in DEFAULT_SCENARIOS outcomes and no cost increase.

## What it does not unlock
- Does not change backend selection, shot counts, or any other routing constant. Does not move any DEFAULT_SCENARIOS scenario between tiers. Does not widen the set of gated policies beyond CRITICALITY_SPREAD_THRESHOLD.

## Ledger-backed replay
- **Baseline** total estimated cost: 700 credits, physical-error exposure: 5590
- **Proposed** total estimated cost: 700 credits, physical-error exposure: 5590

| scenario | fidelity_target | baseline tier/backend | proposed tier/backend | baseline cost | proposed cost | meets? |
|---|---:|---|---|---:|---:|---|
| flat_small | 0.9 | T1/ibm_marrakesh | T1/ibm_marrakesh | 50 | 50 | yes |
| flat_large | 0.95 | T1/ibm_marrakesh | T1/ibm_marrakesh | 200 | 200 | yes |
| skewed_small | 0.9 | T1/ibm_marrakesh | T1/ibm_marrakesh | 50 | 50 | yes |
| skewed_large | 0.97 | T2/ibm_marrakesh | T2/ibm_marrakesh | 200 | 200 | yes |
| skewed_high_fidelity | 0.995 | T2/ibm_marrakesh | T2/ibm_marrakesh | 200 | 200 | yes |
