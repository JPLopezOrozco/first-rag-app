from types import SimpleNamespace

import app.service.query as q
from app.schema import QueryRequest

HITS = {
    "documents": [["uno", "dos", "tres", "cuatro"]],
    "metadatas": [[{"source": "a.txt", "chunk": i} for i in range(4)]],
}


def fake_llm(reply):
    return lambda question, context: SimpleNamespace(content=[SimpleNamespace(text=reply)])


def run(monkeypatch, reply, retrieve=lambda question, n: HITS):
    monkeypatch.setattr(q, "need_retrieval", lambda question: True)
    monkeypatch.setattr(q, "llm_call", fake_llm(reply))
    return q.answer(QueryRequest(question="pregunta"), retrieve=retrieve)


def test_sources_carry_the_citation_number_they_answer_to(monkeypatch):
    response = run(monkeypatch, "Dato [2] y otro [4].")

    assert [(s.ref, s.text, s.chunk) for s in response.sources] == [(2, "dos", 1), (4, "cuatro", 3)]


def test_citation_numbers_stay_even_when_earlier_hits_are_not_cited(monkeypatch):
    response = run(monkeypatch, "Solo esto [3].")

    assert [s.ref for s in response.sources] == [3]


def test_citing_a_number_that_does_not_exist_adds_no_source(monkeypatch):
    response = run(monkeypatch, "Dato [9] y [1].")

    assert [s.ref for s in response.sources] == [1]


def test_answer_without_citations_is_refused(monkeypatch):
    response = run(monkeypatch, "Algo sin citar.")

    assert response.answer == q.NOT_FOUND
    assert response.sources == []


def test_not_found_reply_has_no_sources(monkeypatch):
    response = run(monkeypatch, q.NOT_FOUND)

    assert response.sources == []


def test_no_hits_returns_not_found_without_calling_the_model(monkeypatch):
    def explode(question, context):
        raise AssertionError("no debería llamar al modelo")

    monkeypatch.setattr(q, "need_retrieval", lambda question: True)
    monkeypatch.setattr(q, "llm_call", explode)
    empty = {"documents": [[]], "metadatas": [[]]}

    response = q.answer(QueryRequest(question="pregunta"), retrieve=lambda question, n: empty)

    assert response.answer == q.NOT_FOUND
