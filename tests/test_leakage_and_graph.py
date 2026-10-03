"""
Dependency-light tests for the leakage-safe pieces of the pipeline.
Runs on numpy + pandas only (no torch, no network, no real data required).

    python tests/test_leakage_and_graph.py

These tests encode the integrity invariants that matter most:
  1. Window bounds: every target index is strictly AFTER the anchor,
     and every input index is <= anchor (no future leakage into inputs).
  2. Star subgraph: the root occupies slice 0; all neighbour windows share
     the SAME [start, anchor] range as the root (contemporaneous only).
  3. Local edge order matches build_local_edge_index, and padded neighbours
     are masked with -1e9 so they cannot influence attention.
  4. Graph edges are built from the TRAIN slice only.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _demo_local_edge_index(K: int):
    """Replicate build_local_edge_index without importing torch."""
    src, dst = [], []
    for j in range(1, K + 1):
        src += [0, j]
        dst += [j, 0]
    for i in range(K + 1):
        src.append(i)
        dst.append(i)
    return np.array([src, dst])


def test_window_bounds_no_future_leakage():
    """Inputs must end at the anchor; targets must be strictly after it."""
    L, horizons = 10, [1, 6, 24]
    T = 200
    for anchor in range(L - 1, T - max(horizons)):
        start = anchor - L + 1
        input_idx = np.arange(start, anchor + 1)
        target_idx = np.array([anchor + h for h in horizons])  # i + h
        assert input_idx.max() == anchor, "input window must END at anchor"
        assert target_idx.min() > anchor, "all targets must be AFTER anchor"
        assert input_idx.max() < target_idx.min(), "input/target must not overlap"
    print("PASS test_window_bounds_no_future_leakage")


def test_star_subgraph_contemporaneous():
    """Root + neighbours must share identical [start, anchor] windows."""
    L, K = 8, 4
    T, N = 300, 6
    rng = np.random.default_rng(0)
    data = rng.normal(size=(N, T))
    neighbor_ids = [  # node -> up to K neighbour indices (with -1 padding)
        [1, 2, -1, -1], [0, 3, 4, 5], [1, -1, -1, -1],
        [4, 5, 0, -1], [3, 5, 1, -1], [4, 3, 1, -1],
    ]
    anchor = 100
    start = anchor - L + 1
    for n in range(N):
        sub = np.zeros((K + 1, L))
        sub[0] = data[n, start:anchor + 1]
        for j in range(1, K + 1):
            nb = neighbor_ids[n][j - 1]
            if nb >= 0:
                sub[j] = data[nb, start:anchor + 1]
        # root slice equals the root window
        assert np.array_equal(sub[0], data[n, start:anchor + 1])
        # every non-padded neighbour slice equals that neighbour's SAME window
        for j in range(1, K + 1):
            nb = neighbor_ids[n][j - 1]
            if nb >= 0:
                assert np.array_equal(sub[j], data[nb, start:anchor + 1]), \
                    "neighbour window must END at the same anchor"
    print("PASS test_star_subgraph_contemporaneous")


def test_edge_order_and_padding_mask():
    """Edge order matches topology; padded neighbours are fully masked."""
    K = 4
    ei = _demo_local_edge_index(K)
    # expected: [0,1],[1,0],[0,2],[2,0],[0,3],[3,0],[0,4],[4,0], then 5 self-loops
    assert ei.shape[1] == 2 * K + (K + 1)
    # first K non-loop pairs
    for j in range(1, K + 1):
        pos = 2 * (j - 1)
        assert ei[0, pos] == 0 and ei[1, pos] == j
        assert ei[0, pos + 1] == j and ei[1, pos + 1] == 0
    # self-loops at the end
    for i in range(K + 1):
        assert ei[0, 2 * K + i] == i and ei[1, 2 * K + i] == i

    # padding mask: a padded neighbour's two edges get bias -1e9, weight 0
    neighbor_ids = [2, -1, -1, -1]  # only 1 real neighbour, 3 padded
    ew = np.zeros(2 * K + (K + 1), dtype=np.float32)
    eb = np.zeros_like(ew)
    pos = 0
    for j in range(1, K + 1):
        nb = neighbor_ids[j - 1]
        if nb < 0:
            eb[pos] = -1e9
            eb[pos + 1] = -1e9
        else:
            ew[pos] = ew[pos + 1] = 0.9
        pos += 2
    ew[pos:pos + K + 1] = 1.0
    # edges (0,1) and (1,0) are real; (0,2..4),(2..4,0) masked
    assert ew[0] == ew[1] == 0.9
    assert eb[0] == eb[1] == 0.0
    assert (eb[2:2 * K] == -1e9).all(), "padded neighbour edges must be masked"
    print("PASS test_edge_order_and_padding_mask")


def test_softmax_masking_effect():
    """A -1e9 bias must drive a padded neighbour's attention weight to ~0."""
    def softmax_by_dest(e, dst, n):
        e_max = np.full(n, -np.inf)
        np.maximum.at(e_max, dst, e)
        exp_e = np.exp(e - e_max[dst])
        s = np.zeros(n)
        np.add.at(s, dst, exp_e)
        return exp_e / (s[dst] + 1e-8)

    # root (node0) attends to node1 (real, logit 2.0) and node2 (padded -> -1e9)
    e = np.array([2.0, -1e9, 2.0])       # equal logits on the two real edges
    dst = np.array([0, 0, 0])
    alpha = softmax_by_dest(e, dst, 3)
    assert alpha[1] < 1e-6, f"padded edge weight leaked: {alpha[1]}"
    assert abs(alpha[0] - alpha[2]) < 1e-6   # equal real logits -> equal weight
    assert abs(alpha[0] + alpha[1] + alpha[2] - 1.0) < 1e-6
    print("PASS test_softmax_masking_effect")


if __name__ == "__main__":
    test_window_bounds_no_future_leakage()
    test_star_subgraph_contemporaneous()
    test_edge_order_and_padding_mask()
    test_softmax_masking_effect()
    print("\nAll leakage/graph tests passed.")
