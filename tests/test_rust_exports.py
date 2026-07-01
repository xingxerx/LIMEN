"""Guard test: the built limen_core extension must expose every symbol
the Python package tries to import from it.

Every Rust-accelerated path in limen/ follows the same pattern: try to
import a limen_core symbol, silently fall back to pure Python if the
import fails. That fallback is deliberate for environments without the
extension — but it also means an *outdated* build of limen_core (one
missing newer exports) degrades performance silently instead of failing
loudly. This happened in practice: a stale wheel shipped without
simulate_qubo_runs / logical_failure_probability / build_ecc_lookup_table,
and the validator and ECC certifier ran their Python fallbacks for weeks
undetected.

This test scans limen/ source for limen_core usage and asserts each
referenced symbol resolves against the installed extension. It skips
entirely when limen_core is not built (the zero-dependency fallback
story stays intact); it fails when limen_core is present but stale.
"""

import re
import unittest
from pathlib import Path

LIMEN_SRC = Path(__file__).resolve().parent.parent / "limen"

# `from limen_core import a, b as c` (possibly parenthesized/multiline)
_FROM_IMPORT = re.compile(
    r"from limen_core import \(?([\w\s,]+?)\)?$", re.MULTILINE
)
# `limen_core.name` or `limen_core.submodule.name` attribute chains,
# in code or docstrings (a doc reference to a nonexistent symbol is
# stale documentation, which this test intentionally also catches).
_ATTR_CHAIN = re.compile(r"\blimen_core\.([A-Za-z_]\w*)(?:\.([A-Za-z_]\w*))?")


def _referenced_symbols() -> tuple[set[str], set[tuple[str, str]]]:
    """Collect limen_core symbols referenced anywhere under limen/.

    Returns:
        (top_level_names, (submodule, attr) pairs). A bare
        `limen_core.sub` reference contributes ("sub", "") meaning the
        submodule itself must exist.
    """
    top_level: set[str] = set()
    chained: set[tuple[str, str]] = set()

    for path in LIMEN_SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in _FROM_IMPORT.finditer(text):
            for clause in match.group(1).split(","):
                name = clause.split(" as ")[0].strip()
                if name:
                    top_level.add(name)
        for match in _ATTR_CHAIN.finditer(text):
            first, second = match.group(1), match.group(2)
            if second is not None:
                chained.add((first, second))
            else:
                top_level.add(first)

    return top_level, chained


class TestRustExportsCurrent(unittest.TestCase):

    def setUp(self):
        try:
            import limen_core
        except ImportError:
            self.skipTest(
                "limen_core not built; pure-Python fallbacks are in use "
                "by design"
            )
        self.limen_core = limen_core

    def test_scan_finds_known_dependencies(self):
        # Sanity-check the scanner itself: if the regexes rot, this
        # catches it before the main assertion silently passes on an
        # empty set.
        top_level, chained = _referenced_symbols()
        self.assertIn("simulate_qubo_runs", top_level)
        self.assertIn("exact_ising_norm", top_level)
        self.assertIn(("ecc", "select_patches"), chained)
        self.assertIn(("cutting", "reconstruct_expectation"), chained)

    def test_every_referenced_symbol_exists(self):
        top_level, chained = _referenced_symbols()

        missing: list[str] = []
        for name in sorted(top_level):
            if not hasattr(self.limen_core, name):
                missing.append(f"limen_core.{name}")
        for sub, attr in sorted(chained):
            submodule = getattr(self.limen_core, sub, None)
            if submodule is None:
                missing.append(f"limen_core.{sub}")
            elif not hasattr(submodule, attr):
                missing.append(f"limen_core.{sub}.{attr}")

        self.assertEqual(
            missing,
            [],
            "limen_core is built but missing symbols the Python package "
            f"references: {missing}. The installed extension is stale — "
            "rebuild it with `maturin develop --release`.",
        )


if __name__ == "__main__":
    unittest.main()
