from repository.repository import ping


def check_health() -> dict:
    database_ok = ping()
    return {
        "status": "ok" if database_ok else "degraded",
        "database": "ok" if database_ok else "error",
    }
