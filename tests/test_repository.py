import pytest

import repository.repository as repo
from repository.repository import to_vector_literal


def test_to_vector_literal():
    assert to_vector_literal([0.1, 0.2]) == "[0.1,0.2]"


def test_to_vector_literal_empty():
    assert to_vector_literal([]) == "[]"


class FakeResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class FakeVoyage:
    """Reemplaza al cliente de Voyage: cada texto 'x' * i devuelve el vector [i]."""

    def __init__(self):
        self.calls = []
        self.models = []

    def embed(self, texts, model=None, **kwargs):
        self.calls.append(list(texts))
        self.models.append(model)
        return FakeResult([[float(len(t))] for t in texts])


def test_embed_texts_batches_and_keeps_order(monkeypatch):
    fake = FakeVoyage()
    monkeypatch.setattr(repo, "vo", fake)
    texts = ["x" * i for i in range(1, 251)]

    result = repo.embed_texts(texts)

    assert [len(c) for c in fake.calls] == [128, 122]
    assert result == [[float(i)] for i in range(1, 251)]


def test_embed_texts_uses_voyage_4_lite(monkeypatch):
    fake = FakeVoyage()
    monkeypatch.setattr(repo, "vo", fake)

    repo.embed_texts(["hola"])

    assert fake.models == ["voyage-4-lite"]


def test_embed_texts_empty_makes_no_call(monkeypatch):
    fake = FakeVoyage()
    monkeypatch.setattr(repo, "vo", fake)

    assert repo.embed_texts([]) == []
    assert fake.calls == []


def row(id_, content=None, meta=None, score=0.0):
    return (id_, content or id_.upper(), meta or {"source": "s", "chunk": id_}, score)


@pytest.fixture
def branches(monkeypatch):
    """Vectorial: a, b. Palabras: b, c. embed_texts cuenta cuántas veces se llama."""
    calls = {"embed": 0}

    def fake_embed(texts):
        calls["embed"] += 1
        return [[0.0] for _ in texts]

    monkeypatch.setattr(repo, "embed_texts", fake_embed)
    monkeypatch.setattr(repo, "vector_rows", lambda emb, n: [row("a"), row("b")][:n])
    monkeypatch.setattr(repo, "lexical_rows", lambda q, n: [row("b"), row("c")][:n])
    return calls


def test_candidate_rows_is_deduplicated_union(branches):
    rows = repo.candidate_rows("pregunta", 10)

    assert [r[0] for r in rows] == ["a", "b", "c"]
    assert branches["embed"] == 1


class FakeItem:
    def __init__(self, index, document, relevance_score):
        self.index = index
        self.document = document
        self.relevance_score = relevance_score


class FakeRerank:
    def __init__(self, order):
        self.order = order
        self.calls = []

    def rerank(self, query, docs, model=None, top_k=None):
        self.calls.append({"query": query, "docs": list(docs), "model": model, "top_k": top_k})
        picked = self.order[:top_k]
        items = [FakeItem(i, docs[i], 1.0 - 0.1 * pos) for pos, i in enumerate(picked)]
        return type("R", (), {"results": items})()


def test_rerank_search_returns_reranked_chunks_in_chroma_shape(branches, monkeypatch):
    fake = FakeRerank(order=[2, 0, 1])  # c, a, b
    monkeypatch.setattr(repo, "vo", fake)

    r = repo.rerank_search("pregunta", k=2, n_each=10)

    assert r["documents"] == [["C", "A"]]
    assert r["metadatas"] == [[{"source": "s", "chunk": "c"}, {"source": "s", "chunk": "a"}]]
    assert r["scores"] == [[pytest.approx(1.0), pytest.approx(0.9)]]
    assert fake.calls[0]["model"] == "rerank-2.5"
    assert fake.calls[0]["top_k"] == 2
    assert fake.calls[0]["docs"] == ["A", "B", "C"]


def test_rerank_search_without_candidates_does_not_call_rerank(monkeypatch):
    fake = FakeRerank(order=[])
    monkeypatch.setattr(repo, "vo", fake)
    monkeypatch.setattr(repo, "candidate_rows", lambda q, n: [])

    r = repo.rerank_search("pregunta")

    assert r == {"documents": [[]], "metadatas": [[]], "scores": [[]]}
    assert fake.calls == []


def test_hybrid_search_fuses_both_branches(branches):
    r = repo.hybrid_search("pregunta", n_results=3, alpha=0.5)

    assert r["ids"] == [["b", "a", "c"]]  # b está en las dos ramas
    assert r["documents"] == [["B", "A", "C"]]
    assert r["metadatas"][0][0] == {"source": "s", "chunk": "b"}


def test_hybrid_search_respects_n_results(branches):
    r = repo.hybrid_search("pregunta", n_results=1)

    assert len(r["ids"][0]) == 1


def test_hybrid_search_with_no_results(monkeypatch):
    monkeypatch.setattr(repo, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(repo, "vector_rows", lambda emb, n: [])
    monkeypatch.setattr(repo, "lexical_rows", lambda q, n: [])

    assert repo.hybrid_search("pregunta") == {"ids": [[]], "documents": [[]], "metadatas": [[]]}


# --- pool de conexiones ---

class FakePool:
    created = []

    def __init__(self, dsn, **kwargs):
        self.dsn = dsn
        self.kwargs = kwargs
        self.closed = False
        FakePool.created.append(self)

    check_connection = staticmethod(lambda conn: None)

    def close(self):
        self.closed = True


def test_get_pool_is_created_once_and_lazily(monkeypatch):
    FakePool.created = []
    monkeypatch.setattr(repo, "_pool", None)
    monkeypatch.setattr(repo, "ConnectionPool", FakePool)
    monkeypatch.setenv("DB_POOL_MAX", "7")

    assert FakePool.created == []  # importar o configurar no crea el pool
    first = repo.get_pool()
    second = repo.get_pool()

    assert first is second
    assert len(FakePool.created) == 1
    assert first.dsn == repo.DSN
    assert first.kwargs["max_size"] == 7
    assert first.kwargs["check"] is FakePool.check_connection


def test_close_pool_closes_and_resets(monkeypatch):
    fake = FakePool("dsn")
    monkeypatch.setattr(repo, "_pool", fake)

    repo.close_pool()

    assert fake.closed
    assert repo._pool is None


def test_close_pool_without_pool_does_nothing(monkeypatch):
    monkeypatch.setattr(repo, "_pool", None)

    repo.close_pool()  # no debe fallar


class FakeConnection:
    def execute(self, sql):
        self.sql = sql


class WorkingPool:
    def connection(self, timeout=None):
        conn = FakeConnection()

        class Ctx:
            def __enter__(self_inner):
                return conn

            def __exit__(self_inner, *exc):
                return False

        return Ctx()


class BrokenPool:
    def connection(self, timeout=None):
        raise RuntimeError("sin conexión")


def test_ping_true_when_database_answers(monkeypatch):
    monkeypatch.setattr(repo, "get_pool", lambda: WorkingPool())

    assert repo.ping() is True


def test_ping_false_when_database_fails(monkeypatch):
    monkeypatch.setattr(repo, "get_pool", lambda: BrokenPool())

    assert repo.ping() is False
