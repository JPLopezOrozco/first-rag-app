import hmac
import math
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)) -> str:
    expected = os.getenv("API_KEY")
    if not expected:
        raise HTTPException(503, "API_KEY no está configurada en el servidor")
    if key is None or not hmac.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(401, "API key ausente o inválida")
    return key


class RateLimiter:

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic, max_keys: int = 10_000):
        self.limit = limit
        self.window = window
        self.clock = clock
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> float:
        now = self.clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return self.window - (now - hits[0])
            hits.append(now)
            if len(self._hits) > self.max_keys:
                self._prune(now)
            return 0.0

    def _prune(self, now: float) -> None:
        for key in [k for k, h in self._hits.items() if not h or now - h[-1] >= self.window]:
            del self._hits[key]


def rate_limit(limiter: RateLimiter):

    def dependency(key: str = Depends(require_api_key)) -> None:
        wait = limiter.hit(key)
        if wait > 0:
            raise HTTPException(
                429,
                "Demasiadas solicitudes, probá de nuevo en unos segundos",
                headers={"Retry-After": str(math.ceil(wait))},
            )

    return dependency


def client_ip(request: Request) -> str:
    hops = int(os.getenv("TRUSTED_PROXY_HOPS", "0"))
    if hops > 0:
        forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(forwarded) >= hops:
            return forwarded[-hops]
    return request.client.host if request.client else "desconocida"


def public_rate_limit(per_ip: RateLimiter, overall: RateLimiter):

    def dependency(request: Request) -> None:
        for limiter, key in ((per_ip, client_ip(request)), (overall, "todos")):
            wait = limiter.hit(key)
            if wait > 0:
                raise HTTPException(
                    429,
                    "Demasiadas solicitudes, probá de nuevo en unos segundos",
                    headers={"Retry-After": str(math.ceil(wait))},
                )

    return dependency
