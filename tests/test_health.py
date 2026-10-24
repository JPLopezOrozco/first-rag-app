from fastapi.testclient import TestClient

import main


def test_health_ok_without_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(main, "check_health", lambda: {"status": "ok", "database": "ok"})

    r = TestClient(main.app).get("/health")

    assert r.status_code == 200
    assert r.json() == {"status": "ok", "database": "ok"}


def test_health_503_when_database_is_down(monkeypatch):
    monkeypatch.setattr(main, "check_health", lambda: {"status": "degraded", "database": "error"})

    r = TestClient(main.app).get("/health")

    assert r.status_code == 503
    assert r.json()["database"] == "error"
