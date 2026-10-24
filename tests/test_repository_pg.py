import psycopg
import pytest

import repository.repository as repo

FILENAME = "test_repo_add_chunks.txt"
DIM = 1024


def fake_embed(texts):
    """El chunk k recibe el vector unitario e_k: determinista, sin llamar a Voyage."""
    return [[1.0 if j == k else 0.0 for j in range(DIM)] for k in range(len(texts))]


def meta(n):
    return [{"source": FILENAME, "chunk": i} for i in range(n)]


def ids(n):
    return [f"{FILENAME}_chunk_{i}" for i in range(n)]


@pytest.fixture
def db(monkeypatch):
    try:
        conn = psycopg.connect(repo.DSN, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("Postgres no está levantado")
    with conn:
        conn.execute("DELETE FROM documents WHERE filename = %s", (FILENAME,))
    monkeypatch.setattr(repo, "embed_texts", fake_embed)
    yield repo.DSN
    with psycopg.connect(repo.DSN) as conn:
        conn.execute("DELETE FROM documents WHERE filename = %s", (FILENAME,))


def stored(dsn):
    with psycopg.connect(dsn) as conn:
        return conn.execute(
            """
            SELECT c.id, c.content, c.metadata, d.content_type
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.filename = %s
            ORDER BY c.chunk
            """,
            (FILENAME,),
        ).fetchall()


def test_add_chunks_stores_ids_metadata_and_content_type(db):
    repo.add_chunks(FILENAME, ["uno", "dos"], meta(2), ids(2), "text/plain")

    assert stored(db) == [
        (f"{FILENAME}_chunk_0", "uno", {"source": FILENAME, "chunk": 0}, "text/plain"),
        (f"{FILENAME}_chunk_1", "dos", {"source": FILENAME, "chunk": 1}, "text/plain"),
    ]


def test_add_chunks_replaces_previous_version(db):
    repo.add_chunks(FILENAME, ["uno", "dos"], meta(2), ids(2), "text/plain")
    repo.add_chunks(FILENAME, ["nuevo"], meta(1), ids(1), "text/plain")

    assert [r[:2] for r in stored(db)] == [(f"{FILENAME}_chunk_0", "nuevo")]


def test_add_chunks_length_mismatch_fails_before_embedding(db, monkeypatch):
    def explode(texts):
        raise AssertionError("no debería llamar a Voyage")

    monkeypatch.setattr(repo, "embed_texts", explode)

    with pytest.raises(ValueError):
        repo.add_chunks(FILENAME, ["uno", "dos"], meta(2), ids(1), "text/plain")


def test_add_chunks_failure_keeps_old_version(db, monkeypatch):
    repo.add_chunks(FILENAME, ["uno", "dos"], meta(2), ids(2), "text/plain")

    # el segundo vector tiene dimensión incorrecta: la inserción falla
    monkeypatch.setattr(repo, "embed_texts", lambda texts: [[1.0] * DIM, [1.0] * 3])
    with pytest.raises(psycopg.Error):
        repo.add_chunks(FILENAME, ["x", "y"], meta(2), ids(2), "text/plain")

    assert [r[:2] for r in stored(db)] == [
        (f"{FILENAME}_chunk_0", "uno"),
        (f"{FILENAME}_chunk_1", "dos"),
    ]


def unit(k):
    return fake_embed(["a"] * (k + 1))[k]


@pytest.fixture
def three_chunks(db, monkeypatch):
    """Tres chunks con vectores e_0, e_1, e_2 y, después, la consulta devuelve e_1."""
    repo.add_chunks(FILENAME, ["uno", "dos", "tres"], meta(3), ids(3), "text/plain")
    monkeypatch.setattr(repo, "embed_texts", lambda texts: [unit(1)])
    return db


def test_search_has_chroma_shape(three_chunks):
    r = repo.search("lo que sea", n_results=3)

    for key in ("documents", "metadatas", "distances"):
        assert len(r[key]) == 1
        assert len(r[key][0]) == len(r["documents"][0])


def test_search_closest_first(three_chunks):
    r = repo.search("lo que sea", n_results=1)

    assert r["documents"][0] == ["dos"]
    assert r["metadatas"][0] == [{"source": FILENAME, "chunk": 1}]
    assert r["distances"][0][0] == pytest.approx(0, abs=1e-6)


def test_search_respects_n_results_and_orders_by_distance(three_chunks):
    r = repo.search("lo que sea", n_results=3)

    distances = r["distances"][0]
    assert len(r["documents"][0]) == 3
    assert distances == sorted(distances)


@pytest.fixture
def invented(db):
    """Dos chunks con palabras inventadas, para no chocar con el corpus real."""
    repo.add_chunks(
        FILENAME,
        ["las zanfordias del norte", "las quimeras del sur"],
        meta(2),
        ids(2),
        "text/plain",
    )
    return db


def found_ids(rows):
    return [r[0] for r in rows]


def test_lexical_rows_matches_by_stem(invented):
    assert found_ids(repo.lexical_rows("zanfordia", 4)) == [f"{FILENAME}_chunk_0"]


def test_lexical_rows_no_match_returns_empty(invented):
    assert repo.lexical_rows("palabrainexistente", 4) == []


def test_lexical_rows_only_stopwords_returns_empty(invented):
    assert repo.lexical_rows("el de la", 4) == []


def test_lexical_rows_matches_any_term(invented):
    assert set(found_ids(repo.lexical_rows("zanfordia quimera", 4))) == {
        f"{FILENAME}_chunk_0",
        f"{FILENAME}_chunk_1",
    }


def test_lexical_rows_respects_n(invented):
    assert len(repo.lexical_rows("zanfordia quimera", 1)) == 1


def test_lexical_rows_returns_content_metadata_and_rank(invented):
    chunk_id, content, metadata, rank = repo.lexical_rows("zanfordia", 4)[0]

    assert content == "las zanfordias del norte"
    assert metadata == {"source": FILENAME, "chunk": 0}
    assert rank > 0
