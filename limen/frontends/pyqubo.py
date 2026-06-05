"""PyQUBO frontend adapter for LIMEN.

Converts compiled PyQUBO models and raw QUBO dicts into a LogicalGraph
from limen.core.ir. PyQUBO is imported lazily inside from_pyqubo() so
this module loads cleanly even when pyqubo is not installed.
"""

from limen.core.ir import Interaction, LogicalGraph, Variable


def _build_graph(
    qubo_dict: dict[tuple[str, str], float],
    metadata: dict,
) -> LogicalGraph:
    """Build and validate a LogicalGraph from a raw QUBO dict.

    Args:
        qubo_dict: Mapping of (variable_name, variable_name) pairs to weights.
        metadata: Metadata dict to attach to the graph.

    Returns:
        A validated LogicalGraph.

    Raises:
        ValueError: If the constructed graph fails validation.
    """
    names: set[str] = set()
    for i, j in qubo_dict:
        names.add(i)
        names.add(j)

    variables = [Variable(name=n, domain="binary") for n in sorted(names)]
    interactions = [
        Interaction(i=min(i, j), j=max(i, j), weight=float(w))
        for (i, j), w in sorted(
            ((( min(k), max(k) ), v) for k, v in qubo_dict.items())
        )
    ]

    graph = LogicalGraph(variables=variables, interactions=interactions, metadata=metadata)

    errors = graph.validate()
    if errors:
        raise ValueError(f"LogicalGraph validation failed: {errors}")

    return graph


def from_pyqubo(model, feed_dict: dict | None = None) -> LogicalGraph:
    """Convert a compiled PyQUBO model into a LogicalGraph.

    Args:
        model: A compiled PyQUBO model (result of ``Model.compile()``).
        feed_dict: Optional mapping of placeholder names to values,
            forwarded to ``model.to_qubo()``.

    Returns:
        A validated LogicalGraph with metadata ``{"source": "pyqubo",
        "feed_dict": <feed_dict>}``.

    Raises:
        ImportError: If pyqubo is not installed.
        ValueError: If the resulting graph fails validation.
    """
    try:
        import pyqubo as _pyqubo  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(
            "pyqubo is required to use from_pyqubo(). "
            "Install it with: pip install pyqubo"
        ) from exc

    resolved_feed = feed_dict or {}
    qubo, _ = model.to_qubo(feed_dict=resolved_feed)

    metadata = {"source": "pyqubo", "feed_dict": resolved_feed}
    return _build_graph(qubo, metadata)


def from_qubo_dict(
    qubo_dict: dict[tuple[str, str], float],
    metadata: dict | None = None,
) -> LogicalGraph:
    """Convert a raw QUBO dict into a LogicalGraph.

    Args:
        qubo_dict: Mapping of ``(variable_name, variable_name)`` tuples
            to float weights.
        metadata: Optional extra metadata merged into the graph's metadata
            alongside ``{"source": "qubo_dict"}``.

    Returns:
        A validated LogicalGraph.

    Raises:
        ValueError: If the resulting graph fails validation.
    """
    combined: dict = {"source": "qubo_dict"}
    if metadata:
        combined.update(metadata)

    return _build_graph(qubo_dict, combined)
