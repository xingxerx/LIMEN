"""Tests for the co-design history feedback loop: CoDesignResult
serialization and codesign_from_history seeding a fresh loop from the
best prior run on the same backend, the same way
limen.router.history/informed_fleet folds run history into cost/queue
estimates instead of starting from static defaults every time.

Skipped cleanly if limen_core Rust extension is not installed (matches
tests/test_codesign.py).
"""

import pathlib
import tempfile

import pytest

pytest.importorskip("limen_core", reason="limen_core Rust extension not installed")

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.codesign.solver import (
    CoDesignResult,
    codesign_from_history,
    run_codesign,
    save_codesign_result,
)

_TRIVIAL_QUBO = {("q0", "q0"): -1.0, ("q1", "q1"): -1.0, ("q0", "q1"): 2.0}


def _make_encoding(qubo: dict = _TRIVIAL_QUBO):
    return compile_lexicographic(from_qubo_dict(qubo), default_hardware_graph(8))


def test_codesign_result_to_dict_from_dict_round_trips():
    encoding = _make_encoding()
    result = run_codesign(
        encoding, target_kappa=0.5, max_iterations=5, runs_per_iteration=200, seed=0
    )
    restored = CoDesignResult.from_dict(result.to_dict())

    assert restored.kappa == result.kappa
    assert restored.kappa_std == result.kappa_std
    assert restored.iterations == result.iterations
    assert restored.chain_strength_history == result.chain_strength_history
    assert restored.confidence_history == result.confidence_history
    assert restored.converged == result.converged
    assert restored.encoding.to_dict() == result.encoding.to_dict()
    assert restored.score.kappa == result.score.kappa


def test_codesign_from_history_falls_back_to_fresh_run_when_no_history():
    with tempfile.TemporaryDirectory() as tmp:
        encoding = _make_encoding()
        result = codesign_from_history(
            pathlib.Path(tmp),
            encoding,
            backend_name="dwave_sim",
            target_kappa=0.5,
            max_iterations=3,
            runs_per_iteration=100,
            seed=0,
        )
        assert isinstance(result, CoDesignResult)


def test_codesign_from_history_seeds_from_best_prior_run():
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = pathlib.Path(tmp)
        encoding = _make_encoding()

        worse = CoDesignResult(
            encoding=encoding,
            score=run_codesign(
                encoding, target_kappa=0.5, max_iterations=1, runs_per_iteration=50, seed=1
            ).score,
            kappa=0.2,
            kappa_std=0.0,
            iterations=1,
            chain_strength_history=[3.5],
            confidence_history=[0.2],
            converged=False,
        )
        better = CoDesignResult(
            encoding=encoding,
            score=worse.score,
            kappa=0.9,
            kappa_std=0.0,
            iterations=1,
            chain_strength_history=[7.25],
            confidence_history=[0.9],
            converged=True,
        )
        save_codesign_result(results_dir, "dwave_sim", worse)
        save_codesign_result(results_dir, "dwave_sim", better)
        # A record for a different backend must never be picked.
        other_backend = CoDesignResult(
            encoding=encoding,
            score=worse.score,
            kappa=0.99,
            kappa_std=0.0,
            iterations=1,
            chain_strength_history=[99.0],
            confidence_history=[0.99],
            converged=True,
        )
        save_codesign_result(results_dir, "ionq_forte", other_backend)

        result = codesign_from_history(
            results_dir,
            encoding,
            backend_name="dwave_sim",
            target_kappa=0.99,
            max_iterations=1,
            runs_per_iteration=50,
            seed=0,
        )
        # First iteration's recorded chain strength is the *starting*
        # encoding's, before any in-loop recompilation -- so this proves
        # codesign_from_history seeded from the higher-kappa ("better")
        # record's final chain strength (7.25), not "worse" (3.5) or the
        # unrelated "ionq_forte" backend's (99.0).
        assert result.chain_strength_history[0] == 7.25
