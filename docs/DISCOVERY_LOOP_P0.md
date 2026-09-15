# Discovery Loop P0 (in progress)

One-liner: LIMEN improves its own encodings and routing policies from run history, with a certificate trail per improvement.

## Acceptance
Every `run_route_request` writes a RouterMemory row when `results_dir` is set, unless `memory=None` opts out.

## Gap
Recording exists (`_record_route_outcome`) but default `memory=None` skips the ledger.

## Done when
Tests prove default+results_dir writes; explicit None does not; no QPU; no merge.
