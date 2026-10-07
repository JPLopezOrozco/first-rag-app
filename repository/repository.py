import chromadb 
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import re, unicodedata
from rank_bm25 import BM25Okapi
import os



load_dotenv()
embedding_fn = embedding_functions.VoyageAIEmbeddingFunction(
    api_key=os.environ["VOYAGE_API_KEY"],
    model_name="voyage-4-lite",
)

client = chromadb.PersistentClient(path="chromadb")

collection = client.get_or_create_collection(
    name="documents_collection",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(filename: str, chunks: list[str], metadata: list[dict], ids: list[str]) -> None:
    collection.delete(where={"source": filename})
    collection.upsert(documents=chunks, metadatas=metadata, ids=ids)



def search(query: str, n_results: int = 4) -> dict:
    return collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    
    
    
def tokenize(s: str) -> list[str]:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.findall(r"\w+", s)

_bm25 = None
_ids: list[str] = []

def _build_bm25():
    global _bm25, _ids
    todos = collection.get(include=["documents"])
    _ids = todos["ids"]
    _bm25 = BM25Okapi([tokenize(d) for d in todos["documents"]])
    

def vector_ids(query, n):
    r = collection.query(query_texts=[query], n_results=n)
    return r["ids"][0]

def bm25_ids(query, n):
    if _bm25 is None:
        _build_bm25()
    scores = _bm25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [_ids[i] for i in top if scores[i] > 0] 

def rrf(rankings, weights, k=60):
    scores = {}
    for ranking, w in zip(rankings, weights):
        for pos, id_ in enumerate(ranking, 1):
            scores[id_] = scores.get(id_, 0) + w / (k + pos)
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_search(query:str, n_results: int = 4, alpha: float = 0.5, pool: int =20)->dict:
    fused = rrf(
        [vector_ids(query, pool), bm25_ids(query, pool)],
        [alpha, 1 - alpha],
    )[:n_results]
    
    got = collection.get(ids=fused, include=["documents", "metadatas"])
    by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
    return {
        "ids": [fused],
        "documents": [[by_id[i][0] for i in fused]],
        "metadatas": [[by_id[i][1] for i in fused]]
    }

    
    
    