import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from service.health import check_health
from service.ingest import process_document
from service.query import answer
from schema import QueryRequest, QueryResponse
from security import RateLimiter, rate_limit
from starlette.concurrency import run_in_threadpool
from exceptions.exceptions import UnsupportedFileType, EmptyDocumentError

load_dotenv()

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ENVELOPE_BYTES = 64 * 1024  # holgura para los encabezados del multipart

question_limiter = RateLimiter(int(os.getenv("QUESTION_RATE_PER_MIN", "20")))
document_limiter = RateLimiter(int(os.getenv("DOCUMENT_RATE_PER_MIN", "5")))

app = FastAPI()


@app.middleware("http")
async def reject_oversized_bodies(request: Request, call_next):
    # Corta antes de leer el cuerpo cuando el cliente declara un tamaño excesivo.
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES + ENVELOPE_BYTES:
        return JSONResponse(
            {"detail": "El cuerpo de la solicitud supera el tamaño máximo permitido"},
            status_code=413,
        )
    return await call_next(request)


@app.get("/health")
async def health():
    # Sin clave de API ni límite: lo consultan las plataformas para saber si la app está viva.
    result = await run_in_threadpool(check_health)
    return JSONResponse(result, status_code=200 if result["status"] == "ok" else 503)


@app.post("/documents", dependencies=[Depends(rate_limit(document_limiter))])
async def ingest_documents(file: UploadFile = File(...)):

    # Se lee un byte de más para detectar el exceso sin cargar el archivo entero.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "El archivo supera el tamaño máximo permitido")

    try:
        n = await run_in_threadpool(process_document, data, file.filename, file.content_type)
    except EmptyDocumentError as e:
        raise HTTPException(422, str(e))
    except UnsupportedFileType as e:
        raise HTTPException(415, str(e))
    return {"filename": file.filename, "chunks": n}


@app.post(
    "/question",
    response_model=QueryResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(rate_limit(question_limiter))],
)
async def query_request(query: QueryRequest) -> QueryResponse:
    return await run_in_threadpool(answer, query)
