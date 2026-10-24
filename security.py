import hmac
import math
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)) -> str:
    """Valida el header X-API-Key contra la variable de entorno API_KEY."""
    expected = os.getenv("API_KEY")
    if not expected:
        raise HTTPException(503, "API_KEY no está configurada en el servidor")
    if key is None or not hmac.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(401, "API key ausente o inválida")
    return key


class RateLimiter:
    """Ventana deslizante en memoria: como máximo `limit` pedidos por clave cada `window` segundos."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic):
        self.limit = limit
        self.window = window
        self.clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> float:
        """Registra un pedido. Devuelve 0 si se permite o los segundos que faltan para poder reintentar."""
        now = self.clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return self.window - (now - hits[0])
            hits.append(now)
            return 0.0


def rate_limit(limiter: RateLimiter):
    """Dependencia de FastAPI: exige la API key y aplica el límite de `limiter` a esa clave."""

    def dependency(key: str = Depends(require_api_key)) -> None:
        wait = limiter.hit(key)
        if wait > 0:
            raise HTTPException(
                429,
                "Demasiadas solicitudes, probá de nuevo en unos segundos",
                headers={"Retry-After": str(math.ceil(wait))},
            )

    return dependency
