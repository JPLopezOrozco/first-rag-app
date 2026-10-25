import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.schema import QueryResponse
from app.security import RateLimiter, client_ip

KEY = "clave-de-prueba"
HEADERS = {"X-API-Key": KEY}
QUESTION = {"question": "¿Quién fundó el club?"}


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


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.setattr(main, "answer", lambda query: QueryResponse(answer="ok", sources=[]))
    monkeypatch.setattr(main, "process_document", lambda data, name, content_type: 3)
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)
    for limiter in (main.question_limiter, main.global_question_limiter, main.document_limiter):
        limiter._hits.clear()
    return TestClient(main.app)


def upload(client, content=b"hola mundo", headers=HEADERS):
    return client.post(
        "/documents", files={"file": ("a.txt", content, "text/plain")}, headers=headers
    )


def test_question_is_public(client):
    r = client.post("/question", json=QUESTION)

    assert r.status_code == 200
    assert r.json()["answer"] == "ok"


def test_question_does_not_need_api_key_configured(client, monkeypatch):
    monkeypatch.delenv("API_KEY")

    assert client.post("/question", json=QUESTION).status_code == 200


def test_documents_closed_when_api_key_not_configured(client, monkeypatch):
    monkeypatch.delenv("API_KEY")

    assert upload(client).status_code == 503


def test_documents_without_key_is_401(client):
    assert upload(client, headers={}).status_code == 401


def test_documents_with_key_is_ok(client):
    r = upload(client)

    assert r.status_code == 200
    assert r.json() == {"filename": "a.txt", "chunks": 3}


def test_question_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(main.question_limiter, "limit", 2)

    assert client.post("/question", json=QUESTION).status_code == 200
    assert client.post("/question", json=QUESTION).status_code == 200
    r = client.post("/question", json=QUESTION)

    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0


def test_upload_over_limit_is_413_in_handler(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)

    assert upload(client, content=b"x" * 150).status_code == 413


def test_upload_at_limit_is_accepted(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)

    assert upload(client, content=b"x" * 100).status_code == 200


def test_oversized_declared_body_is_413_before_reading(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)

    assert upload(client, content=b"x" * 100_000).status_code == 413


def ask(client, ip=None):
    headers = {"X-Forwarded-For": ip} if ip else {}
    return client.post("/question", json=QUESTION, headers=headers).status_code


def test_forwarded_header_is_ignored_by_default(client, monkeypatch):
    monkeypatch.setattr(main.question_limiter, "limit", 2)

    codes = [ask(client, f"10.0.0.{i}") for i in range(3)]

    assert codes == [200, 200, 429]


def test_each_forwarded_ip_has_its_own_limit_behind_a_proxy(client, monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.setattr(main.question_limiter, "limit", 1)

    assert ask(client, "203.0.113.1") == 200
    assert ask(client, "203.0.113.1") == 429
    assert ask(client, "203.0.113.2") == 200


def test_spoofed_leading_forwarded_entries_do_not_escape_the_limit(client, monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.setattr(main.question_limiter, "limit", 1)

    assert ask(client, "1.1.1.1, 203.0.113.9") == 200
    assert ask(client, "2.2.2.2, 203.0.113.9") == 429


def test_global_limit_stops_many_ips(client, monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.setattr(main.global_question_limiter, "limit", 2)

    codes = [ask(client, f"198.51.100.{i}") for i in range(3)]

    assert codes == [200, 200, 429]


def test_blocked_ip_does_not_spend_the_global_budget(client, monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.setattr(main.question_limiter, "limit", 1)
    monkeypatch.setattr(main.global_question_limiter, "limit", 2)

    assert ask(client, "198.51.100.1") == 200
    for _ in range(5):
        assert ask(client, "198.51.100.1") == 429

    assert ask(client, "198.51.100.2") == 200


class FakeRequest:
    def __init__(self, forwarded="", host="9.9.9.9"):
        self.headers = {"x-forwarded-for": forwarded} if forwarded else {}
        self.client = type("C", (), {"host": host})()


def test_client_ip_uses_socket_peer_without_trusted_proxy(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)

    assert client_ip(FakeRequest("1.2.3.4")) == "9.9.9.9"


def test_client_ip_takes_the_entry_added_by_the_nth_proxy(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")

    assert client_ip(FakeRequest("6.6.6.6, 1.2.3.4, 10.0.0.1")) == "1.2.3.4"


def test_client_ip_falls_back_when_the_header_is_shorter_than_the_hops(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")

    assert client_ip(FakeRequest("1.2.3.4")) == "9.9.9.9"


def test_limiter_prunes_idle_keys_when_it_grows():
    clock = Clock()
    limiter = RateLimiter(limit=1, window=60, clock=clock, max_keys=3)
    for key in ("a", "b", "c"):
        limiter.hit(key)

    clock.now = 120
    limiter.hit("d")
    limiter.hit("e")

    assert set(limiter._hits) == {"d", "e"}


def test_front_page_is_served_with_security_headers(client):
    r = client.get("/")

    assert r.status_code == 200
    assert "Taladro" in r.text
    assert r.headers["content-security-policy"].startswith("default-src 'self'")
    assert r.headers["x-content-type-options"] == "nosniff"


def test_static_assets_are_served(client):
    assert client.get("/styles.css").status_code == 200
    assert client.get("/app.js").status_code == 200
    assert client.get("/fonts/barlow-latin-400-normal.woff2").status_code == 200
    assert client.get("/favicon.svg").status_code == 200


def test_static_mount_does_not_shadow_api_routes(client, monkeypatch):
    monkeypatch.setattr(main, "check_health", lambda: {"status": "ok", "database": "ok"})

    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/no-existe").status_code == 404


def test_api_docs_do_not_get_the_strict_csp(client):
    assert "content-security-policy" not in client.get("/openapi.json").headers
