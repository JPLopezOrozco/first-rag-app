import pytest
from fastapi.testclient import TestClient

import main
from schema import QueryResponse
from security import RateLimiter

KEY = "clave-de-prueba"
HEADERS = {"X-API-Key": KEY}
QUESTION = {"question": "¿Quién fundó el club?"}


# --- RateLimiter (puro, con reloj falso) ---

class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_limiter_allows_up_to_limit_then_blocks():
    limiter = RateLimiter(limit=2, window=60, clock=Clock())

    assert limiter.hit("a") == 0
    assert limiter.hit("a") == 0
    assert limiter.hit("a") > 0


def test_limiter_reports_seconds_until_retry():
    clock = Clock()
    limiter = RateLimiter(limit=1, window=60, clock=clock)
    limiter.hit("a")

    clock.now = 10
    assert limiter.hit("a") == pytest.approx(50)


def test_limiter_allows_again_after_window():
    clock = Clock()
    limiter = RateLimiter(limit=1, window=60, clock=clock)
    limiter.hit("a")

    clock.now = 60
    assert limiter.hit("a") == 0


def test_limiter_keys_are_independent():
    limiter = RateLimiter(limit=1, window=60, clock=Clock())
    limiter.hit("a")

    assert limiter.hit("b") == 0


# --- Endpoints (servicios reemplazados por falsos: sin base ni API) ---

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.setattr(main, "answer", lambda query: QueryResponse(answer="ok", sources=[]))
    monkeypatch.setattr(main, "process_document", lambda data, name, content_type: 3)
    for limiter in (main.question_limiter, main.document_limiter):
        limiter._hits.clear()
    return TestClient(main.app)


def upload(client, content=b"hola mundo", headers=HEADERS):
    return client.post(
        "/documents", files={"file": ("a.txt", content, "text/plain")}, headers=headers
    )


def test_question_without_key_is_401(client):
    assert client.post("/question", json=QUESTION).status_code == 401


def test_question_with_wrong_key_is_401(client):
    r = client.post("/question", json=QUESTION, headers={"X-API-Key": "otra"})

    assert r.status_code == 401


def test_question_with_key_is_ok(client):
    r = client.post("/question", json=QUESTION, headers=HEADERS)

    assert r.status_code == 200
    assert r.json()["answer"] == "ok"


def test_endpoints_closed_when_api_key_not_configured(client, monkeypatch):
    monkeypatch.delenv("API_KEY")

    assert client.post("/question", json=QUESTION, headers=HEADERS).status_code == 503
    assert upload(client).status_code == 503


def test_documents_without_key_is_401(client):
    assert upload(client, headers={}).status_code == 401


def test_documents_with_key_is_ok(client):
    r = upload(client)

    assert r.status_code == 200
    assert r.json() == {"filename": "a.txt", "chunks": 3}


def test_question_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(main.question_limiter, "limit", 2)

    assert client.post("/question", json=QUESTION, headers=HEADERS).status_code == 200
    assert client.post("/question", json=QUESTION, headers=HEADERS).status_code == 200
    r = client.post("/question", json=QUESTION, headers=HEADERS)

    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0


def test_upload_over_limit_is_413_in_handler(client, monkeypatch):
    # el tamaño declarado entra en la holgura del middleware, pero el archivo excede el tope
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)

    assert upload(client, content=b"x" * 150).status_code == 413


def test_upload_at_limit_is_accepted(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)

    assert upload(client, content=b"x" * 100).status_code == 200


def test_oversized_declared_body_is_413_before_reading(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)

    assert upload(client, content=b"x" * 100_000).status_code == 413
