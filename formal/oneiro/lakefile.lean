import Lake
open Lake DSL

package oneiro_lean where
  -- No external dependencies; pure Lean 4

@[default_target]
lean_lib Oneiro where
  -- Build all modules under Oneiro/
  roots := #[`Oneiro]
