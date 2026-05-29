"""Unit tests for MemoryStore — verifies persistence and FTS5 recall."""
from pathlib import Path

import pytest
from reporag.memory.store import MemoryStore


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test_memory.db")


def test_remember_returns_id(store: MemoryStore):
    mem_id = store.remember("use JWT for auth with 15min expiry")
    assert isinstance(mem_id, str)
    assert len(mem_id) == 16


def test_remember_and_recall(store: MemoryStore):
    store.remember("decided to use RRF k=60 for fusion", tags=["rrf", "retrieval"], category="decision")
    results = store.recall("RRF fusion")
    assert len(results) > 0
    assert "RRF" in results[0]["content"] or "rrf" in results[0]["content"].lower()


def test_recall_by_tag(store: MemoryStore):
    store.remember("auth uses JWT", tags=["auth", "jwt"], category="decision")
    store.remember("DB uses postgres", tags=["db"], category="decision")
    results = store.recall("decision", tags=["jwt"])
    assert all("jwt" in r["tags"] for r in results)


def test_recall_by_category(store: MemoryStore):
    store.remember("discovered memory leak in indexer", category="discovery")
    store.remember("use lancedb for vectors", category="decision")
    results = store.recall("indexer", category="discovery")
    assert all(r["category"] == "discovery" for r in results)


def test_persistence_across_instances(tmp_path: Path):
    """Memory survives store restart — core acceptance criterion."""
    db = tmp_path / "persist.db"
    s1 = MemoryStore(db)
    mem_id = s1.remember("BM25 params: k1=1.2, b=0.75", tags=["bm25"])
    s1.close()

    s2 = MemoryStore(db)
    result = s2.get_by_id(mem_id)
    assert result is not None
    assert "BM25" in result["content"]
    s2.close()


def test_recall_limit(store: MemoryStore):
    for i in range(20):
        store.remember(f"memory entry number {i}", tags=["test"])
    results = store.recall("memory entry", limit=5)
    assert len(results) <= 5


def test_delete(store: MemoryStore):
    mem_id = store.remember("temp note")
    assert store.delete(mem_id) is True
    assert store.get_by_id(mem_id) is None


def test_delete_nonexistent(store: MemoryStore):
    assert store.delete("nonexistent_id") is False


def test_tags_preserved(store: MemoryStore):
    store.remember("test content", tags=["a", "b", "c"])
    results = store.recall("test content")
    assert results[0]["tags"] == ["a", "b", "c"]
