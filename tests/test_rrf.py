"""Unit tests for RRF fusion — verifies research §4 formula."""
from reporag.retrieval.rrf import rrf_fuse, top_k


def test_rrf_single_list():
    result = rrf_fuse([["a", "b", "c"]], k=60)
    assert result["a"] > result["b"] > result["c"]


def test_rrf_two_lists_agree():
    """Doc appearing top in both lists should score highest."""
    result = rrf_fuse([["x", "y", "z"], ["x", "z", "y"]], k=60)
    assert result["x"] > result["y"]
    assert result["x"] > result["z"]


def test_rrf_formula_exact():
    """Verify exact formula: RRF(d) = Σ 1/(k + r_m(d)), rank is 0-indexed +1."""
    result = rrf_fuse([["a"]], k=60)
    expected = 1.0 / (60 + 0 + 1)
    assert abs(result["a"] - expected) < 1e-9


def test_rrf_k60_standard():
    """k=60 is the standardized constant per research §4."""
    r1 = rrf_fuse([["a", "b"]], k=60)
    r2 = rrf_fuse([["a", "b"]], k=1)
    # With larger k, ranks are more smoothed — gap between a and b smaller
    gap_k60 = r1["a"] - r1["b"]
    gap_k1 = r2["a"] - r2["b"]
    assert gap_k60 < gap_k1


def test_rrf_union_of_lists():
    """Docs appearing in only one list still get a score."""
    result = rrf_fuse([["a", "b"], ["c", "d"]], k=60)
    assert len(result) == 4
    assert all(v > 0 for v in result.values())


def test_top_k():
    result = rrf_fuse([["a", "b", "c", "d"]], k=60)
    top = top_k(result, 2)
    assert len(top) == 2
    assert top[0][0] == "a"
    assert top[0][1] > top[1][1]


def test_rrf_empty_lists():
    result = rrf_fuse([[], []], k=60)
    assert result == {}


def test_rrf_differs_from_pure_dense_or_sparse():
    """RRF result order must differ from either individual list alone."""
    dense = ["a", "b", "c", "d"]
    sparse = ["d", "c", "b", "a"]
    fused = rrf_fuse([dense, sparse], k=60)
    top = [doc_id for doc_id, _ in top_k(fused, 4)]
    # Neither pure dense nor pure sparse order
    assert top != dense
    assert top != sparse
