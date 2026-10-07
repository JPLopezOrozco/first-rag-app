from fastapi import FastAPI, UploadFile, File, HTTPException
from service.ingest import process_document
from service.query import answer_question, answer_hybrid
from schema import QueryRequest, QueryResponse
from starlette.concurrency import run_in_threadpool
from exceptions.exceptions import UnsupportedFileType, EmptyDocumentError

app = FastAPI()



@app.post("/documents")
async def ingest_documents(file: UploadFile = File(...)):
    
    data = await file.read()
    
    try:
        n = await run_in_threadpool(process_document, data, file.filename, file.content_type)
    except ValueError as e:
        raise HTTPException(415, str(e))
    except EmptyDocumentError as e:
        raise HTTPException(422, str(e))
    except UnsupportedFileType as e:
        raise HTTPException(415, str(e))
    return {f"filename:{file.filename}, chunks: {n}"}


@app.post("/question", response_model=QueryResponse, response_model_exclude_none=True)
async def query_request(query:QueryRequest)->QueryResponse:    
    return await run_in_threadpool(answer_hybrid, query)