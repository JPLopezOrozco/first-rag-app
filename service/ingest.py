import io
from pypdf import PdfReader
from docx import Document
from repository.repository import add_chunks
import tiktoken
from exceptions.exceptions import EmptyDocumentError, UnsupportedFileType


_enc = tiktoken.get_encoding("cl100k_base")


PERMITTED = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


def read_document(data:bytes, content_type:str)->str:
    
    if content_type not in PERMITTED:
        raise UnsupportedFileType("File not permitted")
    
    ext = PERMITTED[content_type]
    
    if ext == ".pdf":
        content = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    elif ext == ".txt":
        content = data.decode("utf-8", errors="replace")
    elif ext ==".docx":
        content = "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    
    
    return content


def chunking_overlap(text: str, chunk_size: int=500, overlap: int=100)->list[str]:
    
    if overlap >= chunk_size:
        raise ValueError("Overlap debe ser menor a chunk size")
    
    tokens = _enc.encode(text)
    step = chunk_size - overlap
    chunks = []
    
    for start in range(0, len(tokens), step):
        window = tokens[start:start + chunk_size]
        chunks.append(_enc.decode(window))
        if start + chunk_size >= len(tokens):
            break
    
    return chunks
    
def process_document(data:bytes, filename: str,content_type:str)->int:
    content = read_document(data, content_type)
    if not content.strip():
        raise EmptyDocumentError("No se pudo extraer texto")
    chunks = chunking_overlap(content)
    metadata = [{"source": filename, "chunk":i } for i in range(len(chunks))]
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    
    add_chunks(filename, chunks, metadata, ids)
    
    return len(chunks)

        

    
    
        

    
    