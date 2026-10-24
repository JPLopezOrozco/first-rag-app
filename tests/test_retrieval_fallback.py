import logging

import service.query as q


def test_default_retrieve_uses_rerank_when_it_works(monkeypatch, caplog):
    monkeypatch.setattr(q, "rerank_search", lambda query, n: {"documents": [["rerank"]], "metadatas": [[{}]]})

    with caplog.at_level(logging.WARNING):
        result = q.default_retrieve("pregunta", 4)

    assert result["documents"] == [["rerank"]]
    assert caplog.records == []


def test_default_retrieve_falls_back_to_hybrid_and_logs_the_error(monkeypatch, caplog):
    def broken(query, n):
        raise RuntimeError("voyage caído")

    monkeypatch.setattr(q, "rerank_search", broken)
    monkeypatch.setattr(q, "hybrid_search", lambda query, n, alpha: {"documents": [["hibrido"]], "metadatas": [[{}]]})

    with caplog.at_level(logging.WARNING):
        result = q.default_retrieve("pregunta", 4)

    assert result["documents"] == [["hibrido"]]
    assert len(caplog.records) == 1
    assert "reranker" in caplog.records[0].getMessage().lower()
    assert caplog.records[0].exc_info is not None
