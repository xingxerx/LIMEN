use petgraph::graph::{NodeIndex, UnGraph};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Stoer-Wagner global minimum cut, run to bisection.
///
/// petgraph (checked up to 0.8.3) has no Stoer-Wagner implementation --
/// only max-flow/min-spanning-tree/etc -- so this is a from-scratch port
/// of the textbook algorithm (Stoer & Wagner, 1997) onto petgraph's
/// `UnGraph` as the input representation. It replaces
/// `limen.distributed.partition.partition_graph`'s lexicographic
/// variable-name chunking: two variables joined by a heavy QUBO
/// interaction now tend to land in the same partition instead of being
/// split apart by alphabetical name order.
///
/// Runs one minimum-cut-of-the-phase pass at a time, always keeping the
/// globally lightest cut seen and merging the two vertices that phase's
/// cut separated, until only one (merged) vertex remains -- O(n^3) but
/// exact, appropriate for the partition sizes (tens to low hundreds of
/// variables) this router targets.
fn stoer_wagner(graph: &UnGraph<(), f64>) -> Option<(f64, Vec<NodeIndex>)> {
    let n = graph.node_count();
    if n < 2 {
        return None;
    }

    // Dense adjacency matrix indexed by node.index(); merged "super
    // vertices" accumulate the original node indices they absorbed so
    // the winning cut can be reported in terms of the original graph.
    let mut weights = vec![vec![0.0_f64; n]; n];
    for edge in graph.edge_references_all() {
        let (a, b, w) = edge;
        weights[a][b] += w;
        weights[b][a] += w;
    }

    let mut merged_into: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    // Plain index-scanned arrays rather than a HashMap: this codebase
    // requires partitioning to be a pure, deterministic function of its
    // input (see limen.router.budget_router.route's docstring), and
    // HashMap iteration order is not guaranteed, which would make the
    // max-connectivity tie-break (and therefore the result, on graphs
    // with equal-weight ties) nondeterministic across runs.
    let mut active: Vec<bool> = vec![true; n];
    let mut active_count = n;

    let mut best_cut_weight = f64::INFINITY;
    let mut best_group: Vec<usize> = Vec::new();

    while active_count > 1 {
        let start = (0..n).find(|&i| active[i]).expect("active_count > 1");
        let mut in_a = vec![false; n];
        in_a[start] = true;
        let mut connection = vec![0.0_f64; n];
        for v in 0..n {
            if active[v] && v != start {
                connection[v] = weights[start][v];
            }
        }

        let mut n_in_a = 1;
        let mut prev = start;
        let mut last = start;
        while n_in_a < active_count {
            let mut sel: Option<usize> = None;
            let mut best_w = f64::NEG_INFINITY;
            for v in 0..n {
                // First-max-wins scan (lowest index on a tie) keeps this
                // deterministic.
                if active[v] && !in_a[v] && connection[v] > best_w {
                    best_w = connection[v];
                    sel = Some(v);
                }
            }
            let s = sel.expect("a candidate remains while n_in_a < active_count");
            prev = last;
            last = s;
            in_a[s] = true;
            n_in_a += 1;
            for v in 0..n {
                if active[v] && !in_a[v] {
                    connection[v] += weights[s][v];
                }
            }
        }

        let cut_of_phase: f64 = (0..n)
            .filter(|&v| active[v] && v != last)
            .map(|v| weights[last][v])
            .sum();
        if cut_of_phase < best_cut_weight {
            best_cut_weight = cut_of_phase;
            best_group = merged_into[last].clone();
        }

        // Merge `last` into `prev` (standard Stoer-Wagner vertex contraction).
        for v in 0..n {
            if active[v] && v != prev && v != last {
                weights[prev][v] += weights[last][v];
                weights[v][prev] += weights[v][last];
            }
        }
        let absorbed = std::mem::take(&mut merged_into[last]);
        merged_into[prev].extend(absorbed);
        active[last] = false;
        active_count -= 1;
    }

    if best_group.is_empty() {
        // Never found more than one active vertex, i.e. n == 1; guarded above.
        return None;
    }
    Some((
        best_cut_weight,
        best_group.into_iter().map(NodeIndex::new).collect(),
    ))
}

trait EdgeReferencesAll {
    fn edge_references_all(&self) -> Vec<(usize, usize, f64)>;
}

impl EdgeReferencesAll for UnGraph<(), f64> {
    fn edge_references_all(&self) -> Vec<(usize, usize, f64)> {
        use petgraph::visit::EdgeRef;
        self.edge_references()
            .map(|e| (e.source().index(), e.target().index(), *e.weight()))
            .collect()
    }
}

/// Bisect a weighted undirected graph via Stoer-Wagner global min-cut.
///
/// # Arguments
/// * `edges`   - `[(i, j, weight)]`, 0-based node indices, one entry per
///   undirected edge (do not list both `(i, j)` and `(j, i)`). Weights
///   should be non-negative -- Stoer-Wagner assumes non-negative
///   capacities, same as its max-flow relatives.
/// * `n_nodes` - Total node count; isolated nodes (no incident edges)
///   are included and end up on whichever side the algorithm places
///   them (arbitrarily, since they contribute no cut weight either way).
///
/// # Returns
/// `(cut_weight, side_a)`: `side_a` lists the node indices on one side
/// of the minimum cut; the complement (`side_b`) is every other index
/// in `0..n_nodes`.
///
/// # Errors
/// Returns a `ValueError` if `n_nodes < 2` (no bisection is possible).
#[pyfunction]
pub fn stoer_wagner_bisect(
    edges: Vec<(usize, usize, f64)>,
    n_nodes: usize,
) -> PyResult<(f64, Vec<usize>)> {
    if n_nodes < 2 {
        return Err(PyValueError::new_err("n_nodes must be >= 2 to bisect"));
    }
    let mut graph = UnGraph::<(), f64>::with_capacity(n_nodes, edges.len());
    let indices: Vec<NodeIndex> = (0..n_nodes).map(|_| graph.add_node(())).collect();
    for (i, j, w) in edges {
        graph.add_edge(indices[i], indices[j], w);
    }
    let (cut_weight, side_a) =
        stoer_wagner(&graph).ok_or_else(|| PyValueError::new_err("no cut found"))?;
    Ok((
        cut_weight,
        side_a.into_iter().map(|n| n.index()).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_triangles_joined_by_one_light_edge() {
        // Two tight triangles {0,1,2} and {3,4,5} joined by a single
        // weight-1 edge (2-3): the min cut must isolate exactly one
        // triangle from the other at weight 1.
        let mut graph = UnGraph::<(), f64>::new_undirected();
        let n: Vec<NodeIndex> = (0..6).map(|_| graph.add_node(())).collect();
        for &(a, b) in &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)] {
            graph.add_edge(n[a], n[b], 10.0);
        }
        graph.add_edge(n[2], n[3], 1.0);

        let (cut_weight, side_a) = stoer_wagner(&graph).expect("cut found");
        assert!((cut_weight - 1.0).abs() < 1e-9);
        let mut side_a_idx: Vec<usize> = side_a.iter().map(|n| n.index()).collect();
        side_a_idx.sort();
        assert!(side_a_idx == vec![0, 1, 2] || side_a_idx == vec![3, 4, 5]);
    }

    #[test]
    fn single_edge() {
        let mut graph = UnGraph::<(), f64>::new_undirected();
        let n: Vec<NodeIndex> = (0..2).map(|_| graph.add_node(())).collect();
        graph.add_edge(n[0], n[1], 5.0);
        let (cut_weight, side_a) = stoer_wagner(&graph).expect("cut found");
        assert!((cut_weight - 5.0).abs() < 1e-9);
        assert_eq!(side_a.len(), 1);
    }
}
